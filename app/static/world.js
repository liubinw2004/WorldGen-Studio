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
      b.onclick = function () { if (i !== current) teleport(i); };
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

    camera.lookAt(
      camera.position.x + Math.sin(yaw) * Math.cos(pitch),
      camera.position.y + Math.sin(pitch),
      camera.position.z - Math.cos(yaw) * Math.cos(pitch)
    );

    if (portal) {
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
      if (current >= 0 && points) {
        elStat.textContent = points.geometry.attributes.position.count.toLocaleString() + ' 点 · ' + fps + ' fps';
      }
    }
  }

  /* ---------------------------------------------------------------- boot */

  function boot() {
    ensureWorld();
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
  }

  boot();
})();
