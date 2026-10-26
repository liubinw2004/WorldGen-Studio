/* World viewer: renders generated Gaussian scenes with Three.js and lets the
   user teleport between them. The point cloud is drawn as size-attenuated
   disc sprites rather than real anisotropic splats -- at 0.5 M points per
   scene the visual gap is small and the frame cost stays low enough to leave
   room for the day/night and wind shaders that make the world feel alive. */
(function () {
  'use strict';

  var canvas = document.getElementById('stage');
  var fade = document.getElementById('fade');
  var elName = document.getElementById('sceneName');
  var elSites = document.getElementById('sites');
  var elStat = document.getElementById('stat');
  var elDay = document.getElementById('day');
  var btnCycle = document.getElementById('btnCycle');
  var btnWind = document.getElementById('btnWind');

  var renderer, scene, camera, points, material;
  var scenes = [];
  var current = -1;
  var yaw = 0, pitch = 0;
  var keys = {};
  var dragging = false, lastX = 0, lastY = 0;
  var clock = null;
  var light = 0.62, autoCycle = true, windOn = true, elapsed = 0;
  var frames = 0, fpsAt = 0, fps = 0;
  var loadToken = 0;
  var debugOnce = true;
  var DEBUG = /[?&]debug=1/.test(location.search);
  var raycaster = new THREE.Raycaster();
  var pointer = new THREE.Vector2();
  var downX = 0, downY = 0;
  /* 'cloud' = walk the Gaussian splat, 'video' = play a rendered 360 mp4 */
  var mode = 'cloud';

  var SH_C0 = 0.28209479177387814;
  var WALK_RADIUS = 0.55;
  var TYPE_BYTES = {
    float32: 4, float: 4, double: 8, uchar: 1, uint8: 1, char: 1, int8: 1,
    int32: 4, uint32: 4, uint: 4, int: 4, short: 2, ushort: 2
  };

  /* ------------------------------------------------------------------ ply */

  function parsePly(buf) {
    var u8 = new Uint8Array(buf);
    var dec = new TextDecoder('ascii');
    var head = dec.decode(u8.subarray(0, Math.min(u8.length, 32768)));
    var end = head.indexOf('end_header');
    if (end < 0) throw new Error('不是有效的 PLY 文件');
    var nl = head.indexOf('\n', end);
    var lines = head.slice(0, nl).split('\n');

    var count = 0, elem = '', props = [];
    for (var i = 0; i < lines.length; i++) {
      var t = lines[i].trim().split(/\s+/);
      if (t[0] === 'element') { elem = t[1]; if (elem === 'vertex') count = parseInt(t[2], 10); }
      else if (t[0] === 'property' && elem === 'vertex') {
        // `property float x` is 3 tokens; `property list uchar int ...` is 5.
        var nm = t[1] === 'list' ? t[4] : t[2];
        if (nm) props.push({ type: t[1] === 'list' ? t[3] : t[1], name: nm });
      }
    }
    if (!count || !props.length) throw new Error('PLY 里没有顶点数据');

    var stride = 0, off = {};
    for (var p = 0; p < props.length; p++) {
      off[props[p].name] = stride;
      stride += TYPE_BYTES[props[p].type] || 4;
    }
    var need = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
    for (var n = 0; n < need.length; n++) {
      if (off[need[n]] === undefined) throw new Error('PLY 缺少属性 ' + need[n]);
    }

    var base = nl + 1;
    var dv = new DataView(buf, base);
    if (base + count * stride > u8.length) {
      count = Math.floor((u8.length - base) / stride);
    }

    var pos = new Float32Array(count * 3);
    var col = new Float32Array(count * 3);
    var siz = new Float32Array(count);
    var hasScale = off.scale_0 !== undefined;

    for (var k = 0; k < count; k++) {
      var b = k * stride;
      // PLY is OpenCV-style (+y down). Rotate 180 deg about x for a y-up world,
      // which is a pure rotation and therefore does not mirror the scene.
      pos[k * 3] = dv.getFloat32(b + off.x, true);
      pos[k * 3 + 1] = -dv.getFloat32(b + off.y, true);
      pos[k * 3 + 2] = -dv.getFloat32(b + off.z, true);

      for (var c = 0; c < 3; c++) {
        col[k * 3 + c] = dv.getFloat32(b + off['f_dc_' + c], true) * SH_C0 + 0.5;
      }
      siz[k] = hasScale ? Math.exp(dv.getFloat32(b + off.scale_0, true)) : 0.03;
    }

    return { count: count, position: pos, color: col, size: siz };
  }

  /* ---------------------------------------------------------------- world */

  var VERT = [
    'attribute vec3 aColor;',
    'attribute float aSize;',
    'uniform float uTime;',
    'uniform float uLight;',
    'uniform float uWind;',
    'uniform float uProj;',
    'uniform vec2 uRes;',
    'varying vec3 vColor;',
    'varying float vLift;',
    'void main() {',
    '  vec3 p = position;',
    '  float lift = clamp((p.y - 0.15) / 5.0, 0.0, 1.0);',
    '  float sway = sin(uTime * 1.5 + p.z * 0.9 + p.y * 0.4) * 0.5',
    '             + sin(uTime * 2.7 + p.x * 1.7) * 0.5;',
    '  float amp = lift * uWind * mix(0.005, 0.035, uLight);',
    '  p.x += sway * amp;',
    '  p.z += cos(uTime * 1.9 + p.x * 0.8) * amp;',
    '  vec4 mv = modelViewMatrix * vec4(p, 1.0);',
    '  float d = max(-mv.z, 0.05);',
    '  gl_Position = projectionMatrix * mv;',
    // The source cloud is a regular equirect lattice; drawing it verbatim
    // produces a visible moire grid. A sub-pixel jitter per point breaks the
    // lattice without moving any geometry.
    '  float h1 = fract(sin(dot(position, vec3(12.9898, 78.233, 37.719))) * 43758.5453);',
    '  float h2 = fract(sin(dot(position, vec3(39.346, 11.135, 83.155))) * 24634.6345);',
    '  gl_Position.xy += (vec2(h1, h2) - 0.5) * 2.1 * (2.0 / uRes) * gl_Position.w;',
    '  gl_PointSize = clamp(aSize * uProj * 4.2 / d, 1.6, 54.0);',
    '  vColor = aColor;',
    '  vLift = lift;',
    '}'
  ].join('\n');

  var FRAG = [
    'uniform float uLight;',
    'varying vec3 vColor;',
    'varying float vLift;',
    'void main() {',
    '  vec2 d = gl_PointCoord - 0.5;',
    '  float r2 = dot(d, d);',
    '  if (r2 > 0.25) discard;',
    '  float a = exp(-r2 * 5.5);',
    '  a *= 1.0 - smoothstep(0.16, 0.25, r2);',
    '  float lum = dot(vColor, vec3(0.299, 0.587, 0.114));',
    '  vec3 night = vec3(0.78, 0.88, 1.22);',
    '  vec3 gold  = vec3(1.32, 1.10, 0.80);',
    '  vec3 day   = vec3(1.05, 1.03, 0.97);',
    '  vec3 tint = mix(night, day, smoothstep(0.0, 0.85, uLight));',
    // warm sources (lanterns, lit windows) keep their glow after dark
    '  tint = mix(tint, gold, smoothstep(0.42, 0.95, lum) * (1.0 - uLight) * 0.85);',
    '  vec3 c = vColor * tint;',
    '  c = mix(c, c * 1.22 + 0.03, smoothstep(0.5, 1.0, lum) * (1.0 - uLight));',
    '  c *= mix(0.92, 1.02, uLight);',
    '  gl_FragColor = vec4(c, a * 0.94);',
    '}'
  ].join('\n');

  function ensureWorld() {
    if (renderer) return;
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: false, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setClearColor(0x070a0d, 1);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(62, 1, 0.01, 500);
    camera.position.set(0, 0, 0);

    material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uLight: { value: light },
        uWind: { value: 1 },
        uProj: { value: 600 },
        uRes: { value: new THREE.Vector2(1280, 800) }
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NormalBlending
    });

    clock = new THREE.Clock();
    makePortal();
    resize();
  }

  function resize() {
    if (!renderer) return;
    var w = canvas.clientWidth || window.innerWidth;
    var h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (material) {
      material.uniforms.uProj.value = (h * renderer.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI / 180) / 2));
      material.uniforms.uRes.value.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio());
    }
  }

  function disposePoints() {
    if (!points) return;
    scene.remove(points);
    points.geometry.dispose();
    points = null;
  }

  function applyScene(data) {
    disposePoints();
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(data.position, 3));
    geo.setAttribute('aColor', new THREE.BufferAttribute(data.color, 3));
    geo.setAttribute('aSize', new THREE.BufferAttribute(data.size, 1));
    geo.computeBoundingSphere();
    points = new THREE.Points(geo, material);
    points.frustumCulled = false;
    scene.add(points);
  }

  /* --------------------------------------------------------------- portal */

  var portal = null, portalLabel = null, portalGlow = null;

  function makeLabel(text) {
    var c = document.createElement('canvas');
    c.width = 512; c.height = 128;
    var ctx = c.getContext('2d');
    ctx.font = '500 44px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';
    ctx.fillStyle = '#9ff5dd';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(94,234,212,0.75)';
    ctx.shadowBlur = 18;
    ctx.fillText(text, 256, 64);
    var tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, depthTest: false
    }));
    sp.scale.set(0.62, 0.155, 1);
    return sp;
  }

  function updatePortal() {
    if (!portal || !scenes.length) return;
    var next = (current + 1) % scenes.length;
    if (portalLabel) {
      portal.remove(portalLabel);
      portalLabel.material.map.dispose();
      portalLabel.material.dispose();
    }
    portalLabel = makeLabel(scenes.length > 1 ? '前往 · ' + sceneLabel(scenes[next], next) : '唯一场景');
    portalLabel.position.set(0, 0.72, 0);
    portal.add(portalLabel);
  }

  function makePortal() {
    var ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.52, 0.028, 10, 64),
      new THREE.MeshBasicMaterial({
        color: 0x5eead4, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
      })
    );
    var veil = new THREE.Mesh(
      new THREE.CircleGeometry(0.5, 56),
      new THREE.MeshBasicMaterial({
        color: 0x2bd6a8, transparent: true, opacity: 0.10,
        side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
      })
    );
    ring.add(veil);
    portalGlow = veil;
    portal = ring;
    portal.position.set(0.82, 0.02, -2.5);
    portal.userData.hit = true;
    scene.add(portal);
    updatePortal();
  }

  function tryPortal(e) {
    if (mode !== 'cloud') return;
    if (!portal || scenes.length < 2) return;
    var rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    if (raycaster.intersectObject(portal, true).length) {
      teleport((current + 1) % scenes.length);
    }
  }

  /* ---------------------------------------------------------------- views */

  function sceneLabel(s, i) {
    var t = (s.prompt_zh || s.prompt || '').trim();
    if (!t) return '场景 ' + (i + 1);
    return t.length > 18 ? t.slice(0, 18) + '…' : t;
  }

  function buildSites() {
    elSites.innerHTML = '';
    scenes.forEach(function (s, i) {
      var b = document.createElement('button');
      b.className = 'site' + (i === current ? ' active' : '');
      b.type = 'button';
      var n = document.createElement('span');
      n.className = 'n';
      n.textContent = sceneLabel(s, i);
      var m = document.createElement('span');
      m.className = 'm';
      m.textContent = (i === current ? '所在 · ' : '') + (s.mode === 'i2s' ? '图像' : '文本');
      b.appendChild(n);
      b.appendChild(m);
      b.onclick = function () {
        if (mode === 'video') {
          if (i === current) { exitVideo(); return; }
        }
        if (i !== current) teleport(i);
      };
      elSites.appendChild(b);
    });
  }

  function markActive() {
    var kids = elSites.children;
    for (var i = 0; i < kids.length; i++) {
      kids[i].classList.toggle('active', i === current);
      var m = kids[i].querySelector('.m');
      if (m) m.textContent = (i === current ? '所在 · ' : '') + (scenes[i].mode === 'i2s' ? '图像' : '文本');
    }
  }

  function teleport(i) {
    if (i < 0 || i >= scenes.length) return;
    if (mode === 'video') exitVideo();
    var token = ++loadToken;
    fade.classList.add('show');
    elStat.textContent = '正在传送…';

    setTimeout(function () {
      fetch('/api/world/ply/' + encodeURIComponent(scenes[i].id))
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.arrayBuffer();
        })
        .then(function (buf) {
          if (token !== loadToken) return;
          var data = parsePly(buf);
          applyScene(data);
          current = i;
          yaw = 0; pitch = 0;
          camera.position.set(0, 0, 0);
          elName.textContent = sceneLabel(scenes[i], i);
          markActive();
          elStat.textContent = data.count.toLocaleString() + ' 个高斯点';
          updatePortal();
          fade.classList.remove('show');
        })
        .catch(function (err) {
          if (token !== loadToken) return;
          elStat.textContent = '传送失败：' + err.message;
          fade.classList.remove('show');
        });
    }, 180);
  }

  /* --------------------------------------------------------------- input */

  canvas.addEventListener('pointerdown', function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    downX = e.clientX; downY = e.clientY;
    canvas.classList.add('dragging');
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    yaw -= dx * 0.0042;
    pitch = Math.max(-1.35, Math.min(1.35, pitch - dy * 0.0034));
  });
  function endDrag() { dragging = false; canvas.classList.remove('dragging'); }
  canvas.addEventListener('pointerup', function (e) {
    endDrag();
    // a click, not a drag: 5 px of slack absorbs trackpad jitter
    if (Math.hypot(e.clientX - downX, e.clientY - downY) < 5) tryPortal(e);
  });
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });

  window.addEventListener('keydown', function (e) {
    // Esc leaves the 360 player; a second Esc is a no-op (use the top-left
    // link to return to the generation UI, since leaving the tab on a stray
    // keypress would be hostile).
    if (e.key === 'Escape' && mode === 'video') { exitVideo(); return; }
    keys[e.key.toLowerCase()] = true;
    if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].indexOf(e.key.toLowerCase()) >= 0) e.preventDefault();
  });
  window.addEventListener('keyup', function (e) { keys[e.key.toLowerCase()] = false; });
  window.addEventListener('blur', function () { keys = {}; endDrag(); });
  window.addEventListener('resize', resize);

  elDay.addEventListener('input', function () {
    light = parseFloat(elDay.value);
    autoCycle = false;
    btnCycle.classList.remove('on');
  });
  btnCycle.addEventListener('click', function () {
    autoCycle = !autoCycle;
    btnCycle.classList.toggle('on', autoCycle);
  });
  btnWind.addEventListener('click', function () {
    windOn = !windOn;
    btnWind.classList.toggle('on', windOn);
  });

  /* ---------------------------------------------------------------- loop */

  function step() {
    requestAnimationFrame(step);
    if (!renderer) return;

    var dt = Math.min(clock.getDelta(), 0.1);
    elapsed += dt;

    if (autoCycle) {
      // 48 s cycle, starting at dusk so lanterns read immediately
      light = 0.5 - 0.5 * Math.cos((elapsed / 48) * Math.PI * 2);
      elDay.value = light.toFixed(2);
    }

    var speed = 0.9 * dt;
    if (mode === 'cloud') {
      if (keys['w'] || keys['arrowup']) camera.position.z -= speed * Math.cos(yaw);
      if (keys['w'] || keys['arrowup']) camera.position.x -= speed * Math.sin(yaw);
      if (keys['s'] || keys['arrowdown']) camera.position.z += speed * Math.cos(yaw);
      if (keys['s'] || keys['arrowdown']) camera.position.x += speed * Math.sin(yaw);
      if (keys['a'] || keys['arrowleft']) camera.position.x -= speed * Math.cos(yaw);
      if (keys['a'] || keys['arrowleft']) camera.position.z += speed * Math.sin(yaw);
      if (keys['d'] || keys['arrowright']) camera.position.x += speed * Math.cos(yaw);
      if (keys['d'] || keys['arrowright']) camera.position.z -= speed * Math.sin(yaw);

      // keep the viewer inside the shell that actually has geometry
      var r = Math.hypot(camera.position.x, camera.position.z);
      if (r > WALK_RADIUS) {
        camera.position.x *= WALK_RADIUS / r;
        camera.position.z *= WALK_RADIUS / r;
      }
    }

    camera.lookAt(
      camera.position.x + Math.sin(yaw) * Math.cos(pitch),
      camera.position.y + Math.sin(pitch),
      camera.position.z - Math.cos(yaw) * Math.cos(pitch)
    );

    if (portal && mode === 'cloud') {
      var pulse = 0.5 + 0.5 * Math.sin(elapsed * 2.1);
      portal.scale.setScalar(1 + 0.028 * pulse);
      portal.material.opacity = 0.58 + 0.32 * pulse;
      if (portalGlow) portalGlow.material.opacity = 0.05 + 0.11 * pulse;
    }

    material.uniforms.uTime.value = elapsed;
    material.uniforms.uLight.value = light;
    material.uniforms.uWind.value = windOn ? 1 : 0;

    renderer.render(scene, camera);

    if (debugOnce && points && DEBUG) {
      debugOnce = false;
      var n = points ? points.geometry.attributes.position.count : 0;
      var gl = renderer.getContext();
      document.title = 'dbg pts=' + n +
        ' drawn=' + renderer.info.render.points +
        ' calls=' + renderer.info.render.calls +
        ' uProj=' + material.uniforms.uProj.value.toFixed(1) +
        ' size=' + canvas.width + 'x' + canvas.height +
        ' gl=' + (gl ? gl.getParameter(gl.VERSION) : 'none');
    }

    frames++;
    if (elapsed - fpsAt > 0.5) {
      fps = Math.round(frames / (elapsed - fpsAt));
      frames = 0; fpsAt = elapsed;
      if (current >= 0 && points && mode === 'cloud') {
        elStat.textContent = points.geometry.attributes.position.count.toLocaleString() + ' 点 · ' + fps + ' fps';
      }
    }
  }

  /* ----------------------------------------------------------- pano video */

  var elVPath = document.getElementById('vPath');
  var elVHint = document.getElementById('vHint');
  var elVSec = document.getElementById('vSec');
  var elVSize = document.getElementById('vSize');
  var elVFps = document.getElementById('vFps');
  var elVDay = document.getElementById('vDay');
  var elVGo = document.getElementById('vGo');
  var elVProg = document.getElementById('vProg');
  var elVBar = document.getElementById('vBar');
  var elVTxt = document.getElementById('vTxt');
  var elVCancel = document.getElementById('vCancel');
  var elVErr = document.getElementById('vErr');
  var elVList = document.getElementById('vList');
  var elVPlayer = document.getElementById('vPlayer');
  var elVNow = document.getElementById('vNow');
  var elVDownload = document.getElementById('vDownload');
  var elVBack = document.getElementById('vBack');
  var videoEl = document.getElementById('panoVideo');

  var PATH_HINTS = {
    // `still` depends on the 昼夜过渡 switch, so vHintFor() fills it in.
    still: '',
    orbit: '偏航转一圈，首尾帧相接，可无缝循环',
    handheld: '轻微低频抖动，加一点呼吸感',
    dolly: '沿 +z 推进 0.4 m 再回落，产生视差（较慢，上限 0.8 m）',
    crane: '垂直起降 0.4 m 再回落（较慢，上限 0.8 m）'
  };
  var PATH_SLOW = { dolly: true, crane: true };

  var videoJob = null;
  var videoTimer = null;
  /* Read the initial state off the markup: the product default is 开, so a
     first-time user gets a clip that actually moves instead of a near-still. */
  var videoDayOn = elVDay.classList.contains('on');
  var videoUrl = '';
  var panoMesh = null;
  var videoTex = null;

  function vErr(msg) {
    elVErr.textContent = msg || '';
  }

  function vShowProgress(on, done, total) {
    elVProg.hidden = !on;
    if (!on) return;
    var pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
    elVBar.style.width = pct.toFixed(1) + '%';
    elVTxt.textContent = total > 0
      ? '已渲染 ' + done + ' / ' + total + ' 帧 · ' + pct.toFixed(0) + '%'
      : '正在加载点云…';
  }

  function vStopPoll() {
    if (videoTimer) { clearInterval(videoTimer); videoTimer = null; }
  }

  /* A 360 player is an inside-out sphere with the clip as its texture. The
     mesh is mirrored on x so the texture reads correctly from the inside, then
     rotated -90 deg about y so u=0.5 (the pano centre, i.e. the splat's +z)
     lands on -z, which is where the point-cloud camera starts looking. */
  function ensurePanoMesh() {
    if (panoMesh) return;
    var geo = new THREE.SphereGeometry(60, 64, 32);
    geo.scale(-1, 1, 1);
    videoTex = new THREE.VideoTexture(videoEl);
    videoTex.colorSpace = THREE.SRGBColorSpace;
    videoTex.minFilter = THREE.LinearFilter;
    videoTex.magFilter = THREE.LinearFilter;
    videoTex.generateMipmaps = false;
    panoMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      map: videoTex,
      depthTest: false,
      depthWrite: false
    }));
    panoMesh.rotation.y = -Math.PI / 2;
    panoMesh.frustumCulled = false;
    panoMesh.visible = false;
    scene.add(panoMesh);
  }

  function enterVideo(url, label) {
    if (!url) return;
    ensureWorld();
    ensurePanoMesh();
    videoUrl = url;

    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.loop = true;
    if (videoEl.src.indexOf(url) < 0) videoEl.src = url;
    var p = videoEl.play();
    if (p && p.catch) p.catch(function () { /* user gesture / codec */ });

    panoMesh.visible = true;
    if (points) points.visible = false;
    if (portal) portal.visible = false;
    if (portalLabel) portalLabel.visible = false;
    mode = 'video';

    yaw = 0; pitch = 0;
    camera.position.set(0, 0, 0);
    camera.fov = 78;
    camera.updateProjectionMatrix();
    renderer.setClearColor(0x000000, 1);

    elVPlayer.hidden = false;
    elVNow.textContent = label || '全景视频';
    elVDownload.href = url;
    elVDownload.setAttribute('download', url.split('/').pop() || 'pano.mp4');
    elName.textContent = '全景视频 · 360°';
    elStat.textContent = '拖动环视 · 循环播放';
    markVideoItems();
  }

  function exitVideo() {
    if (mode !== 'video') return;
    mode = 'cloud';
    if (panoMesh) panoMesh.visible = false;
    try { videoEl.pause(); } catch (e) { /* ignore */ }
    if (points) points.visible = true;
    if (portal) portal.visible = true;
    if (portalLabel) portalLabel.visible = true;

    yaw = 0; pitch = 0;
    camera.position.set(0, 0, 0);
    camera.fov = 62;
    camera.updateProjectionMatrix();
    renderer.setClearColor(0x070a0d, 1);

    elVPlayer.hidden = true;
    markVideoItems();
    if (current >= 0 && scenes[current]) {
      elName.textContent = sceneLabel(scenes[current], current);
      elStat.textContent = points
        ? points.geometry.attributes.position.count.toLocaleString() + ' 个高斯点'
        : '—';
    } else {
      elName.textContent = '世界漫游';
      elStat.textContent = '—';
    }
  }

  function markVideoItems() {
    var kids = elVList.querySelectorAll('.vitem');
    for (var i = 0; i < kids.length; i++) {
      kids[i].classList.toggle('on', !!videoUrl && kids[i].getAttribute('data-url') === videoUrl
        && mode === 'video');
    }
  }

  function renderVideoList(items) {
    elVList.innerHTML = '';
    if (!items || !items.length) {
      var p = document.createElement('p');
      p.className = 'vempty';
      p.textContent = '还没有生成的视频';
      elVList.appendChild(p);
      return;
    }
    items.forEach(function (it) {
      var label = (it.path_label || it.path || '视频') +
        ' · ' + (it.seconds || 0) + 's · ' +
        (it.width || 0) + '×' + (it.height || 0);
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'vitem';
      b.setAttribute('data-url', it.url);
      var n = document.createElement('span');
      n.className = 'vn';
      n.textContent = label;
      var m = document.createElement('span');
      m.className = 'vm';
      m.textContent = (it.size || '') + ' · ' + String(it.created || '').replace('T', ' ');
      b.appendChild(n);
      b.appendChild(m);
      b.onclick = function () { enterVideo(it.url, label); };
      elVList.appendChild(b);
    });
    markVideoItems();
  }

  function loadVideos() {
    fetch('/api/world/videos')
      .then(function (r) { return r.json(); })
      .then(function (d) { if (d && d.ok) renderVideoList(d.items || []); })
      .catch(function () { /* the panel simply stays empty */ });
  }

  function vPoll() {
    if (!videoJob) return;
    fetch('/api/world/video/' + encodeURIComponent(videoJob))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || !d.ok) throw new Error((d && d.error) || '任务状态丢失');
        vShowProgress(true, d.frames_done || 0, d.frames_total || 0);
        if (d.state === 'done') {
          vStopPoll();
          vShowProgress(false);
          elVGo.disabled = false;
          loadVideos();
          enterVideo(d.url, vLabel());
        } else if (d.state === 'error') {
          vStopPoll();
          vShowProgress(false);
          elVGo.disabled = false;
          vErr(d.error || '渲染失败');
        } else if (d.state === 'cancelled') {
          vStopPoll();
          vShowProgress(false);
          elVGo.disabled = false;
          vErr('已取消渲染');
        }
      })
      .catch(function (err) {
        vStopPoll();
        vShowProgress(false);
        elVGo.disabled = false;
        vErr(err.message);
      });
  }

  function vStart() {
    vErr('');
    if (!scenes.length || current < 0) {
      vErr('还没有可用的场景，先生成一个。');
      return;
    }
    var size = parseInt(elVSize.value, 10) || 1024;
    var body = {
      scene: scenes[current].id,
      path: elVPath.value,
      seconds: parseFloat(elVSec.value) || 6,
      fps: parseInt(elVFps.value, 10) || 24,
      width: size,
      height: size / 2,
      wind: windOn ? 0.6 : 0
    };
    // Off: hold whatever the day/night slider shows. On: dusk -> midnight.
    body.light_from = videoDayOn ? 0.9 : light;
    body.light_to = videoDayOn ? 0.2 : light;

    elVGo.disabled = true;
    vShowProgress(true, 0, 0);
    fetch('/api/world/video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
      .then(function (r) {
        return r.json().then(function (d) { return { status: r.status, d: d }; });
      })
      .then(function (res) {
        if (!res.d || !res.d.ok) throw new Error((res.d && res.d.error) || '启动失败');
        videoJob = res.d.job;
        vShowProgress(true, 0, res.d.frames_total || 0);
        vStopPoll();
        videoTimer = setInterval(vPoll, 600);
      })
      .catch(function (err) {
        vShowProgress(false);
        elVGo.disabled = false;
        vErr(err.message);
      });
  }

  function vCancelJob() {
    if (!videoJob) return;
    fetch('/api/world/video/' + encodeURIComponent(videoJob) + '/cancel', { method: 'POST' })
      .catch(function () { /* the poll reports the final state */ });
  }

  function vHintFor(p) {
    if (p === 'still') {
      return videoDayOn ? '机位不动，昼夜渐变 + 风' : '机位不动，只有风';
    }
    return PATH_HINTS[p] || '';
  }

  function vSyncHint() {
    var p = elVPath.value;
    elVHint.textContent = vHintFor(p);
    elVHint.classList.toggle('slow', !!PATH_SLOW[p]);
  }

  function vLabel() {
    var opt = elVPath.options[elVPath.selectedIndex];
    return (opt ? opt.text : '全景') + ' · ' + elVSec.value + 's';
  }

  elVPath.addEventListener('change', vSyncHint);
  elVGo.addEventListener('click', vStart);
  elVCancel.addEventListener('click', vCancelJob);
  elVBack.addEventListener('click', exitVideo);
  elVDay.addEventListener('click', function () {
    videoDayOn = !videoDayOn;
    elVDay.classList.toggle('on', videoDayOn);
    elVDay.textContent = videoDayOn ? '开' : '关';
    vSyncHint();
  });

  /* ---------------------------------------------------------------- boot */

  function boot() {
    ensureWorld();
    vSyncHint();
    // Allow deep links like /world?light=0.9 for stills and for sharing a mood.
    var q = new URLSearchParams(location.search);
    if (q.has('light')) {
      var v = parseFloat(q.get('light'));
      if (isFinite(v)) {
        light = Math.max(0, Math.min(1, v));
        autoCycle = false;
        elDay.value = light.toFixed(2);
        btnCycle.classList.remove('on');
      }
    }
    step();
    loadVideos();
    fetch('/api/world/scenes')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        scenes = (d.items || []).filter(function (s) { return s.bytes > 100000; });
        if (!scenes.length) {
          elStat.textContent = '还没有可漫游的场景，先生成一个。';
          return;
        }
        buildSites();
        var start = 0;
        if (q.has('scene')) {
          var want = q.get('scene');
          for (var i = 0; i < scenes.length; i++) {
            if (scenes[i].id === want) { start = i; break; }
          }
        }
        teleport(start);
      })
      .catch(function (err) {
        elStat.textContent = '读取场景失败：' + err.message;
      });

    // Deep link: /world?play=1 (newest clip) or /world?play=<file name>.
    // Waiting for `current >= 0` avoids racing the point-cloud load, which
    // would otherwise overwrite the player state.
    if (q.has('play')) {
      var want = q.get('play') || '1';
      var tries = 0;
      (function tryPlay() {
        tries++;
        fetch('/api/world/videos')
          .then(function (r) { return r.json(); })
          .then(function (d) {
            var items = (d && d.items) || [];
            var pick = null;
            for (var k = 0; k < items.length; k++) {
              if (want === '1' || items[k].name === want || items[k].url === want) {
                pick = items[k];
                break;
              }
            }
            if (pick && current >= 0) {
              enterVideo(pick.url, (pick.path_label || '全景视频') +
                ' · ' + (pick.seconds || 0) + 's');
            } else if (tries < 25) {
              setTimeout(tryPlay, 300);
            }
          })
          .catch(function () { /* no clips yet */ });
      })();
    }
  }

  boot();
})();
