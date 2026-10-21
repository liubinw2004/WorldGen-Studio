/* ==========================================================================
   WorldGen Studio — client
   ========================================================================== */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const PRESETS = [
  { label: '日式庭院', text: 'a serene japanese garden at dusk, koi pond, stone lanterns, maple leaves, soft golden light, HDR' },
  { label: '赛博街道', text: 'a rainy neon cyberpunk street at night, holographic signs, wet asphalt reflections, dense atmosphere' },
  { label: '雪山湖泊', text: 'a mirror-still alpine lake at sunrise, snow peaks, low mist, pastel sky, ultra wide panorama' },
  { label: '古图书馆', text: 'a vast ancient library with towering bookshelves, dust in sunbeams, warm candlelight, deep perspective' },
  { label: '海底遗迹', text: 'sunken ruins under clear tropical water, shafts of light, coral growth, ambient blue depth' },
  { label: '沙漠黄昏', text: 'endless sand dunes at golden hour, long shadows, wind ripples, cinematic wide shot' },
  { label: '悬浮岛屿', text: 'floating islands above a sea of clouds, waterfalls into the void, dreamlike sci-fi scale' },
  { label: '北欧木屋', text: 'a cozy scandinavian cabin interior, fireplace glow, snow outside the window, warm wood tones' },
  { label: '太空舱', text: 'interior of a modern space station observation deck, earth below, cool panel lighting' },
  { label: '竹林小径', text: 'a bamboo forest path with morning fog, dappled light, quiet green tones, tall vertical structure' },
];

const S = {
  mode: 't2s',
  image: null,
  resolution: 1024,
  sr: '',              // '' | 'ultrasharp' | 'realesrgan'
  srReady: {},         // key -> bool, from /api/system
  job: null,
  jobId: null,
  cursor: 0,
  es: null,
  logCount: 0,
  autoScroll: true,
  timer: null,
  lastParams: null,
  history: [],
};

/* ─── helpers ─────────────────────────────────────────────────────────── */
function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('#toastWrap').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 260ms, transform 260ms';
    el.style.opacity = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => el.remove(), 280);
  }, 3600);
}

function fmtElapsed(sec) {
  if (sec == null) return '—';
  const s = Math.max(0, sec);
  if (s < 60) return s.toFixed(1) + 's';
  return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
}

function fmtBytes(n) {
  if (!n) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(1)) + ' ' + u[i];
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const p = n => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/* ─── view switching ──────────────────────────────────────────────────── */
const VIEWS = ['empty', 'running', 'done', 'error'];
function showView(name) {
  for (const v of VIEWS) {
    const el = $('#view' + v[0].toUpperCase() + v.slice(1));
    if (el) el.hidden = v !== name;
  }
  document.body.dataset.view = name;
}

/* ─── mode / params ───────────────────────────────────────────────────── */
function setMode(mode) {
  S.mode = mode;
  $$('.mode').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  $('.mode-switch').dataset.mode = mode;
  $('#imageGroup').hidden = mode !== 'i2s';

  const seg = $('#resSeg');
  $$('button', seg).forEach(b => {
    const res = +b.dataset.res;
    b.disabled = mode === 'i2s' && res !== 1024;
  });

  // i2s already yields a 2048-wide panorama, so an upscaler would be a no-op.
  // Disabling the switch is honest; leaving it on would silently do nothing.
  $$('#srSeg button').forEach(b => {
    const k = b.dataset.sr;
    if (!k) return;
    b.disabled = mode === 'i2s' || S.srReady[k] === false;
  });
  if (mode === 'i2s' && S.sr) setSR('');

  if (mode === 'i2s') {
    setResolution(1024);
    $('#resNote').textContent = '图像模式锁定 1024';
    $('#ctaHint').textContent = '预计 70–90 秒 · 图像会补全为 360° 全景';
    $('#prompt').placeholder = '可选：描述场景氛围，例如 a quiet forest at dawn, misty light';
  } else {
    $('#resNote').textContent = '全景长边';
    $('#ctaHint').textContent = '预计 60–90 秒 · 1024 分辨率留有余量';
    $('#prompt').placeholder = '例如：a serene japanese garden at dusk, koi pond, warm lanterns, cinematic lighting, HDR';
  }
  updateSrHint();
}

function setResolution(res) {
  S.resolution = res;
  $$('#resSeg button').forEach(b => b.classList.toggle('active', +b.dataset.res === res));
  updateSrHint();
}

/* ─── super-resolution switch ─────────────────────────────────────────── */
const SR_LABEL = { ultrasharp: 'UltraSharp', realesrgan: 'Real-ESRGAN' };

function setSR(key) {
  if (key && S.srReady[key] === false) {
    toast(SR_LABEL[key] + ' 的权重文件缺失，无法启用', 'error');
    return;
  }
  S.sr = key || '';
  $$('#srSeg button').forEach(b => b.classList.toggle('active', b.dataset.sr === S.sr));
  updateSrHint();
}

function srTargetWidth() {
  // 2x, capped at 2048 the same way the backend caps it.
  return Math.min(S.resolution * 2, 2048);
}

function updateSrHint() {
  const hint = $('#srHint');
  const note = $('#srNote');
  if (S.mode === 'i2s') {
    note.textContent = '图像模式不需要';
    hint.innerHTML = '图像补全出的全景本身就是 <b>2048</b> 宽，已到超分上限，因此图像模式不使用超分';
    return;
  }
  if (!S.sr) {
    note.textContent = '全景超分 · 2×';
    const dbl = Math.min(S.resolution * 2, 2048);
    hint.innerHTML = S.resolution >= 2048
      ? '当前分辨率已足够高，超分不会再有收益'
      : `关闭时按 <b>${S.resolution}</b> 直接重建；开启后先低分辨率扩散，再放大到 <b>${dbl}</b>，点数约 4 倍`;
    return;
  }
  const tw = srTargetWidth();
  const pts = ((tw * tw) / 4 / 10000).toFixed(0);
  note.textContent = SR_LABEL[S.sr] + ' · ' + Math.round(tw / S.resolution) + '×';
  hint.innerHTML = S.resolution >= 2048
    ? `全景已是 <b>${S.resolution}</b>，超分将自动跳过（达到上限 2048）`
    : `扩散仍按 <b>${S.resolution}</b> 跑，完成后放大到 <b>${tw}</b> 再重建 · 约 <b>${pts} 万</b>个高斯点 · 显存不叠加`;
}

/* ─── prompt ──────────────────────────────────────────────────────────── */
const promptEl = () => $('#prompt');

function renderChips() {
  const box = $('#chips');
  box.innerHTML = '';
  PRESETS.forEach(p => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = p.label;
    b.title = p.text;
    b.onclick = () => {
      promptEl().value = p.text;
      updateCharCount();
      promptEl().focus();
    };
    box.appendChild(b);
  });
}

function updateCharCount() {
  const n = promptEl().value.length;
  $('#charCount').textContent = n + ' 字';
}

/* ─── upload ──────────────────────────────────────────────────────────── */
async function uploadFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    toast('请选择图片文件', 'error');
    return;
  }
  const fd = new FormData();
  fd.append('file', file);
  $('#dzLoading').hidden = false;
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '上传失败');
    S.image = j;
    const img = $('#dzPreview');
    img.src = j.url + '?t=' + Date.now();
    img.hidden = false;
    $('#dzInner').hidden = true;
    $('#btnClearImage').hidden = false;
    toast(`已载入 ${j.width}×${j.height} 照片`, 'ok');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    $('#dzLoading').hidden = true;
  }
}

function clearImage() {
  S.image = null;
  $('#dzPreview').hidden = true;
  $('#dzPreview').src = '';
  $('#dzInner').hidden = false;
  $('#btnClearImage').hidden = true;
  $('#fileInput').value = '';
}

/* ─── stage timeline ──────────────────────────────────────────────────── */
const STAGES = [
  ['init', '装载模型'],
  ['think', '理解输入'],
  ['diffuse', '扩散生成'],
  ['upscale', '超分增强'],
  ['reconstruct', '重建三维'],
  ['ready', '场景就绪'],
];

function buildStageList() {
  const ul = $('#stageList');
  ul.innerHTML = '';
  // The upscale step only exists when the switch is on; showing it greyed out
  // for every run would just add noise.
  STAGES.filter(([k]) => k !== 'upscale' || S.sr).forEach(([k, label]) => {
    const li = document.createElement('li');
    li.dataset.stage = k;
    li.innerHTML = `<i class="sd"></i><span>${label}</span>`;
    ul.appendChild(li);
  });
}

function markStages(key) {
  const lis = $$('#stageList li');
  const idx = lis.findIndex(li => li.dataset.stage === key);
  if (idx < 0) return;
  lis.forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('active', i === idx);
  });
}

/* ─── progress ────────────────────────────────────────────────────────── */
const CIRC = 2 * Math.PI * 86;

function setProgress(pct, label) {
  const c = $('#ringFill');
  const p = Math.max(0, Math.min(100, pct || 0));
  c.style.strokeDashoffset = String(CIRC * (1 - p / 100));
  $('#ringPct').innerHTML = Math.round(p) + '<span>%</span>';
  if (label) $('#ringLabel').textContent = label;
}

/* ─── console ─────────────────────────────────────────────────────────── */
function pushLog(text, level = 'info', at = null) {
  const body = $('#consoleBody');
  const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
  const line = document.createElement('div');
  line.className = 'ln ' + (level || 'info');
  const secs = at != null ? at : (S.job ? (Date.now() / 1000 - S.job.started_at) : 0);
  const ts = '[' + Math.round(secs) + 's]';
  line.innerHTML = `<span class="ts">${ts}</span>`;
  line.appendChild(document.createTextNode(text));
  body.appendChild(line);
  S.logCount++;
  if (S.logCount > 320) {
    body.removeChild(body.firstElementChild);
    S.logCount--;
  }
  if (S.autoScroll && (nearBottom || S.logCount < 4)) {
    body.scrollTop = body.scrollHeight;
  }
}

/* ─── timer ───────────────────────────────────────────────────────────── */
function startTimer() {
  stopTimer();
  S.timer = setInterval(() => {
    if (!S.job) return;
    const running = S.job.status === 'running';
    const el = running
      ? (Date.now() / 1000 - S.job.started_at)
      : (S.job.elapsed || 0);
    $('#runElapsed').textContent = fmtElapsed(el);
  }, 200);
}
function stopTimer() {
  if (S.timer) clearInterval(S.timer);
  S.timer = null;
}

/* ─── SSE ─────────────────────────────────────────────────────────────── */
function connectEvents(since = 0) {
  if (S.es) { S.es.close(); S.es = null; }
  const es = new EventSource('/api/events?since=' + since);
  S.es = es;

  es.onmessage = ev => {
    let d;
    try { d = JSON.parse(ev.data); } catch { return; }
    handleEvent(d);
  };

  es.onerror = () => {
    // Reconnect with the cursor we reached, so no event is lost.
    setTimeout(() => {
      if (S.es === es) connectEvents(S.cursor);
    }, 1500);
  };
}

function handleEvent(d) {
  switch (d.kind) {
    case 'reset': {
      const job = d.job;
      const fresh = job.id !== S.jobId;
      S.job = job;
      if (fresh) {
        S.jobId = job.id;
        // Keep the left panel in sync with the job being shown (matters after
        // a reload, where the panel would otherwise always say "t2s").
        if (job.mode && job.mode !== S.mode) setMode(job.mode);
        $('#consoleBody').innerHTML = '';
        S.logCount = 0;
        buildStageList();
        markStages('init');
        setProgress(0, '准备中');
        startTimer();
      }
      if (job.status === 'running') {
        showView('running');
        $('#btnGenerate').classList.add('running');
        $('#ctaIdle').hidden = true;
        $('#ctaBusy').hidden = false;
      } else if (job.status === 'succeeded') {
        showDone({ elapsed: job.elapsed, out_dir: job.out_dir, splat_bytes: 0 });
      }
      break;
    }
    case 'stage': {
      if (S.job) { S.job.stage = d.stage; S.job.stage_label = d.label; }
      markStages(d.stage);
      if (d.label) $('#ringLabel').textContent = d.label;
      break;
    }
    case 'message':
      if (S.job) S.job.message = d.text;
      $('#ringLabel').textContent = d.text;
      break;
    case 'progress':
      if (S.job) { S.job.step = d.step; S.job.total_steps = d.total; S.job.pct = d.pct; }
      setProgress(d.pct);
      $('#runStep').textContent = d.step + ' / ' + d.total;
      break;
    case 'log':
      pushLog(d.text, d.level, d.t);
      break;
    case 'done':
      onDone(d);
      break;
  }
  S.cursor = (d.i != null ? d.i : S.cursor) + 1;
}

/* ─── lifecycle ───────────────────────────────────────────────────────── */
function resetCta() {
  $('#btnGenerate').classList.remove('running');
  $('#ctaIdle').hidden = false;
  $('#ctaBusy').hidden = true;
}

function onDone(d) {
  stopTimer();
  resetCta();
  if (S.job) {
    S.job.status = d.status;
    S.job.elapsed = d.elapsed;
    S.job.out_dir = d.out_dir;
    S.job.splat = d.splat;
  }

  if (d.status === 'succeeded') {
    setProgress(100, '场景就绪');
    markStages('ready');
    showDone(d);
    toast('场景生成完成', 'ok');
    loadHistory();
  } else if (d.status === 'cancelled') {
    showView('empty');
    renderStats();
    toast('已停止生成');
  } else {
    showError(d.error || '生成失败');
  }
}

function showDone(d) {
  const job = S.job || {};
  $('#doneBadge').textContent = job.mode || S.mode;
  $('#donePrompt').textContent = job.prompt || '';
  $('#donePrompt').title = job.prompt || '';

  const meta = [];
  if (job.resolution) meta.push(`分辨率 <b>${job.resolution}</b>`);
  if (d.elapsed) meta.push(`耗时 <b>${fmtElapsed(d.elapsed)}</b>`);
  if (d.splat_bytes) meta.push(`体积 <b>${fmtBytes(d.splat_bytes)}</b>`);
  meta.push(`输出 <b>${(d.out_dir || '').replace(/^D:\\mm\\outputs\\/, 'outputs\\')}</b>`);
  $('#doneMeta').innerHTML = meta.join('<span style="opacity:.3">·</span>');

  showView('done');

  const loading = $('#viewerLoading');
  loading.hidden = false;
  const frame = $('#viewer');
  frame.onload = () => { setTimeout(() => { loading.hidden = true; }, 600); };
  frame.src = 'http://localhost:8080/?darkMode=1&t=' + Date.now();
  setTimeout(() => { loading.hidden = true; }, 15000);
}

function showError(msg) {
  $('#errText').textContent = msg;
  const log = $('#consoleBody').innerText || '';
  $('#errLog').textContent = log.slice(-2400) || '(无日志)';
  showView('error');
}

/* ─── actions ─────────────────────────────────────────────────────────── */
async function generate() {
  if ($('#btnGenerate').classList.contains('running')) {
    await cancel();
    return;
  }
  if (S.job && S.job.status === 'running') return;

  const prompt = promptEl().value.trim();
  if (S.mode === 't2s' && !prompt) {
    toast('先写一句场景描述', 'error');
    promptEl().focus();
    return;
  }
  if (S.mode === 'i2s' && !S.image) {
    toast('先选择一张照片', 'error');
    return;
  }

  const btn = $('#btnGenerate');
  btn.disabled = true;
  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: S.mode,
        prompt: prompt || 'a beautiful scene',
        image: S.image ? S.image.path : null,
        resolution: S.resolution,
        sr: S.sr || null,
      }),
    });
    const j = await r.json();
    if (!j.ok) {
      toast(j.error || '无法启动生成', 'error');
      return;
    }
    // A Chinese prompt is translated server-side; show the English that FLUX
    // actually receives so the result is explainable.
    const zhPrompt = j.job && j.job.prompt_zh;
    if (zhPrompt) {
      $('#transText').textContent = j.job.prompt;
      $('#transNote').hidden = false;
    } else {
      $('#transNote').hidden = true;
    }
    S.lastParams = {
      mode: S.mode,
      prompt: promptEl().value,
      image: S.image,
      resolution: S.resolution,
      sr: S.sr,
    };
    // UI switches immediately; the SSE reset event carries the full job.
    $('#consoleBody').innerHTML = '';
    S.logCount = 0;
    buildStageList();
    markStages('init');
    setProgress(0, '准备中');
    showView('running');
    btn.classList.add('running');
    $('#ctaIdle').hidden = true;
    $('#ctaBusy').hidden = false;
  } catch (e) {
    toast('请求失败：' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function cancel() {
  try {
    await fetch('/api/cancel', { method: 'POST' });
  } catch { /* ignore */ }
}

async function reveal(path) {
  const r = await fetch('/api/reveal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  const j = await r.json();
  if (!j.ok) toast(j.error || '无法打开文件夹', 'error');
}

/* ─── history ─────────────────────────────────────────────────────────── */
async function loadHistory() {
  try {
    const r = await fetch('/api/history');
    const j = await r.json();
    S.history = j.items || [];
    renderHistory();
  } catch { /* ignore */ }
}

function renderHistory() {
  const list = $('#historyList');
  $('#histCount').textContent = S.history.length;
  $('#historyEmpty').hidden = S.history.length > 0;
  list.innerHTML = '';

  if (!S.history.length) return;

  S.history.forEach((it, idx) => {
    const card = document.createElement('button');
    card.className = 'hist-card';

    const thumb = document.createElement('div');
    thumb.className = 'hist-thumb';
    if (it.thumb) {
      const im = document.createElement('img');
      im.src = it.thumb;
      im.alt = '';
      thumb.appendChild(im);
    } else {
      thumb.innerHTML = `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9L12 3.5z" stroke="currentColor" stroke-width="1.4"/></svg>`;
    }

    const body = document.createElement('div');
    body.className = 'hist-body';
    body.innerHTML = `
      <div class="hist-title">${escapeHtml(it.prompt_zh || it.prompt || '(无描述)')}</div>
      <div class="hist-sub">
        <span class="tag">${it.mode}</span>
        <span>${it.resolution}${it.sr ? ' → ' + (it.sr_result || '2×') : ''}</span>
        <span>${it.size}</span>
        <span>${fmtTime(it.created)}</span>
      </div>`;
    card.append(thumb, body);
    card.onclick = () => openDetail(idx);
    list.appendChild(card);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function openDetail(idx) {
  const it = S.history[idx];
  if (!it) return;
  const rows = [
    ['模式', it.mode === 'i2s' ? '图像 → 场景' : '文本 → 场景'],
    ['描述', it.prompt_zh || it.prompt || '—'],
    ...(it.prompt_zh ? [['实际输入', it.prompt]] : []),
    ['分辨率', it.resolution + (it.sr_result ? ` → ${it.sr_result}` : '')],
    ['画质增强', it.sr ? (SR_LABEL[it.sr] || it.sr) : '关闭'],
    ['高斯点', it.gaussians ? it.gaussians.toLocaleString() : '—'],
    ['文件', it.size],
    ['耗时', it.elapsed ? fmtElapsed(it.elapsed) : '—'],
    ['时间', fmtTime(it.created)],
    ['路径', it.dir],
  ];
  $('#systemBody').innerHTML =
    rows.map(([k, v]) => `<div class="sys-row"><span>${k}</span><b>${escapeHtml(String(v))}</b></div>`).join('') +
    `<div style="display:flex;gap:8px;margin-top:6px">
       <button class="tool-btn" id="mReveal">打开文件夹</button>
       <button class="tool-btn primary" id="mDownload">下载 splat.ply</button>
     </div>`;
  $('#modalTitle').textContent = '场景详情';
  $('#systemModal').hidden = false;
  $('#mReveal').onclick = () => reveal(it.dir);
  $('#mDownload').onclick = () => {
    window.location.href = '/api/download?path=' + encodeURIComponent(it.dir + '\\splat.ply');
  };
}

/* ─── GPU ─────────────────────────────────────────────────────────────── */
function paintGpu(g) {
  if (!g || !g.ok || !g.total) {
    $('#gpuName').textContent = 'GPU 未检测到';
    $('#gpuDetail').textContent = 'nvidia-smi 不可用';
    return;
  }
  const pct = Math.round((g.used / g.total) * 100);
  const circ = 94.2;
  const arc = $('#gpuArc');
  arc.style.strokeDashoffset = String(circ * (1 - pct / 100));
  arc.style.stroke = pct > 94 ? '#f0736f' : pct > 82 ? '#e0a34a' : '#5eead4';
  $('#gpuPct').textContent = pct;
  $('#gpuName').textContent = (g.name || '').replace(/NVIDIA GeForce\s*/i, '');
  $('#gpuDetail').textContent = `${(g.used / 1024).toFixed(1)} / ${(g.total / 1024).toFixed(0)} GB · ${g.util}%`;
  $('#runVram').textContent = `${(g.used / 1024).toFixed(1)} GB`;
}

async function pollState() {
  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    paintGpu(j.gpu);
    if (j.job && !S.job && j.job.status === 'running') {
      S.job = j.job;
      S.jobId = j.job.id;
      connectEvents(j.job.event_base);
    }
  } catch { /* ignore */ }
}

/* ─── stats on empty view ─────────────────────────────────────────────── */
function renderStats() {
  const g = S.gpu || {};
  const items = [
    ['模型', 'FLUX.1 + DA-2'],
    ['显存', g.total ? (g.total / 1024).toFixed(0) + ' GB' : '—'],
    ['离线', '本地缓存'],
    ['输出', 'splat.ply'],
  ];
  $('#statsRow').innerHTML = items
    .map(([k, v]) => `<div class="stat-card"><b>${v}</b><span>${k}</span></div>`)
    .join('');
}

/* ─── system modal ────────────────────────────────────────────────────── */
async function openSystem() {
  try {
    const r = await fetch('/api/system');
    const j = await r.json();
    const rows = [
      ['GPU', j.gpu.name || '—'],
      ['显存', j.gpu.total ? `${(j.gpu.used / 1024).toFixed(1)} / ${(j.gpu.total / 1024).toFixed(1)} GB` : '—'],
      ['Python', j.python],
      ['离线模式', j.offline ? '已启用（D 盘缓存）' : '关闭'],
      ['输出目录', j.outputs],
    ];
    let html = rows
      .map(([k, v]) => `<div class="sys-row"><span>${k}</span><b>${escapeHtml(String(v))}</b></div>`)
      .join('');
    html += '<div style="margin-top:4px;font-size:11px;color:var(--txt-3)">模型组件</div>';
    html += j.models.map(m =>
      `<div class="sys-row"><span>${m.label}</span><b class="${m.ready ? 'sys-ok' : 'sys-bad'}">${m.ready ? '就绪' : '缺失'}</b></div>`
    ).join('');
    if (j.sr && j.sr.length) {
      html += '<div style="margin-top:10px;font-size:11px;color:var(--txt-3)">超分模型</div>';
      html += j.sr.map(m =>
        `<div class="sys-row"><span>${escapeHtml(m.label)}</span><b class="${m.ready ? 'sys-ok' : 'sys-bad'}">${m.ready ? '就绪' : '缺失'}</b></div>`
      ).join('');
    }
    if (j.translator) {
      html += '<div style="margin-top:10px;font-size:11px;color:var(--txt-3)">中文输入</div>';
      html += `<div class="sys-row"><span>${escapeHtml(j.translator.label)}</span>`
        + `<b class="${j.translator.ready ? 'sys-ok' : 'sys-bad'}">${j.translator.ready ? '就绪' : '缺失'}</b></div>`;
    }
    $('#systemBody').innerHTML = html;
    $('#modalTitle').textContent = '运行环境';
    $('#systemModal').hidden = false;
  } catch (e) {
    toast('无法读取系统信息', 'error');
  }
}

/* ─── wiring ──────────────────────────────────────────────────────────── */
async function loadSrAvailability() {
  try {
    const j = await (await fetch('/api/system')).json();
    (j.sr || []).forEach(m => { S.srReady[m.key] = m.ready; });
    $$('#srSeg button').forEach(b => {
      const k = b.dataset.sr;
      if (!k) return;
      b.disabled = S.srReady[k] === false || S.mode === 'i2s';
      b.title = S.srReady[k] === false
        ? (SR_LABEL[k] + ' 权重文件缺失')
        : (S.mode === 'i2s' ? '图像模式的全景已是 2048 宽' : '');
    });
    if (S.sr && S.srReady[S.sr] === false) setSR('');
    updateSrHint();
  } catch { /* the backend validates again on submit */ }
}

function init() {
  buildStageList();
  renderChips();
  updateCharCount();
  renderStats();
  showView('empty');
  loadSrAvailability();

  // mode switch
  $$('.mode').forEach(b => b.onclick = () => setMode(b.dataset.mode));

  // prompt
  promptEl().addEventListener('input', updateCharCount);
  promptEl().addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') generate();
  });
  $('#btnSurprise').onclick = () => {
    const p = PRESETS[Math.floor(Math.random() * PRESETS.length)];
    promptEl().value = p.text;
    updateCharCount();
  };

  // resolution
  $$('#resSeg button').forEach(b => b.onclick = () => {
    if (!b.disabled) setResolution(+b.dataset.res);
  });

  // super-resolution switch
  $$('#srSeg button').forEach(b => b.onclick = () => setSR(b.dataset.sr || ''));

  // upload
  const dz = $('#dropzone');
  dz.onclick = () => $('#fileInput').click();
  dz.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') $('#fileInput').click(); };
  $('#fileInput').onchange = e => { if (e.target.files[0]) uploadFile(e.target.files[0]); };
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.remove('dragging');
  }));
  dz.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  $('#btnClearImage').onclick = e => { e.stopPropagation(); clearImage(); };

  // generate / cancel
  $('#btnGenerate').onclick = generate;

  // done view actions
  $('#btnExternal').onclick = () => window.open('http://localhost:8080', '_blank');
  $('#btnDownload').onclick = () => {
    const dir = (S.job && S.job.out_dir) || (S.lastDone && S.lastDone.out_dir);
    if (dir) window.location.href = '/api/download?path=' + encodeURIComponent(dir + '\\splat.ply');
  };
  $('#btnReveal').onclick = () => {
    const dir = (S.job && S.job.out_dir) || (S.lastDone && S.lastDone.out_dir);
    if (dir) reveal(dir);
  };
  $('#btnAgain').onclick = () => {
    showView('empty');
    renderStats();
    promptEl().focus();
  };

  // error view
  $('#btnErrBack').onclick = () => { showView('empty'); renderStats(); };
  $('#btnErrRetry').onclick = () => {
    setResolution(1024);
    if (S.mode === 'i2s') setMode('i2s');
    showView('empty');
    generate();
  };

  // console auto scroll
  $('#btnScrollLock').onclick = () => {
    S.autoScroll = !S.autoScroll;
    $('#btnScrollLock').textContent = '自动滚动 · ' + (S.autoScroll ? '开' : '关');
  };

  // system modal
  $('#btnSystem').onclick = openSystem;
  $('#btnCloseSystem').onclick = () => { $('#systemModal').hidden = true; };
  $('#systemModal').onclick = e => {
    if (e.target.id === 'systemModal') $('#systemModal').hidden = true;
  };

  // keyboard: esc closes modal / cancels nothing
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') $('#systemModal').hidden = true;
  });

  // boot
  loadHistory();
  pollState().then(() => connectEvents(0));
  setInterval(async () => {
    try {
      const r = await fetch('/api/state');
      const j = await r.json();
      S.gpu = j.gpu;
      paintGpu(j.gpu);
    } catch { /* ignore */ }
  }, 2500);

  S.gpu = null;
}

document.addEventListener('DOMContentLoaded', init);
