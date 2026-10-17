// AR try-on de estrellitas (POC). MediaPipe Face + Hand Landmarker + canvas.
import { FaceLandmarker, HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const css = getComputedStyle(document.documentElement);
const token = (name) => css.getPropertyValue(name).trim();
const STAR_MM = parseFloat(token('--ar-star-size-mm'));
const IPD_MM = parseFloat(token('--ar-ipd-mm'));
const COLORS = {
  black: token('--black-900'),
  pink: token('--primary-pink-500'),
  yellow: token('--stars-yellow'),
  blue: token('--stars-blue'),
  'light-blue': token('--stars-light-blue'),
};
const WHITE = token('--white-100');
const PINK = token('--primary-pink-500');
// Dedo: dejarlo quieto sobre la cara DWELL_MS para pegar una estrella.
const DWELL_MS = 700;
const STILL_MM = 6;    // tolerancia de movimiento mientras se mantiene
const SPACING_MM = 8;  // no pegar encima de otra estrella
const INDEX_TIP = 8;   // punta del índice en HandLandmarker
const IRIS_R = 468, IRIS_L = 473; // centros de iris
const FACE_POINTS = 468;          // landmarks de malla (sin iris)
// Contorno de la cara (FACEMESH_FACE_OVAL de MediaPipe), para no pegar fuera de la piel.
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const insideFace = (pts, x, y) => {
  let inside = false;
  for (let i = 0, j = FACE_OVAL.length - 1; i < FACE_OVAL.length; j = i++) {
    const [xi, yi] = pts[FACE_OVAL[i]], [xj, yj] = pts[FACE_OVAL[j]];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
};

const $ = (id) => document.getElementById(id);
const canvas = $('stage'), ctx = canvas.getContext('2d'), video = $('cam');

// Estrella del parche real: 5 puntas, gordita, esquinas redondeadas.
// Vértices de una estrella de radio exterior 1 centrada en 0,0.
const STAR_PTS = Array.from({ length: 10 }, (_, i) => {
  const r = i % 2 ? 0.5 : 1, a = -Math.PI / 2 + (i * Math.PI) / 5;
  return [r * Math.cos(a), r * Math.sin(a)];
});
const ROUND = 0.22; // grosor del trazo redondeado relativo al radio

const withAlpha = (hex, a) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

function starPath(c, R) {
  c.beginPath();
  STAR_PTS.forEach(([x, y], i) => (i ? c.lineTo(x * R, y * R) : c.moveTo(x * R, y * R)));
  c.closePath();
}
// Grano fino del hidrocoloide (textura mate tipo gamuza), generado una vez.
const GRAIN = (() => {
  const g = document.createElement('canvas'); g.width = g.height = 64;
  const gc = g.getContext('2d'), img = gc.createImageData(64, 64);
  const n = parseInt(WHITE.slice(1), 16);
  for (let i = 0; i < img.data.length; i += 4) {
    img.data[i] = n >> 16; img.data[i + 1] = (n >> 8) & 255; img.data[i + 2] = n & 255;
    img.data[i + 3] = Math.random() * 40; // motas claras muy sutiles
  }
  gc.putImageData(img, 0, 0);
  return ctx.createPattern(g, 'repeat');
})();

// f = marco local de la piel proyectado a pantalla: origen (x,y), ejes a/b (por mm) y opacidad.
function drawStar(c, f, tilt, color) {
  // R en mm: el ancho punta a punta es 1.902·R y el trazo redondeado suma ROUND·R.
  const R = STAR_MM / (1.902 + ROUND);
  c.save();
  c.setTransform(f.ax * f.ppm, f.ay * f.ppm, f.bx * f.ppm, f.by * f.ppm, f.x, f.y);
  c.rotate(tilt);
  // Acabado mate como la foto real: sin reflejo, bordes levemente difusos sobre la piel.
  c.globalAlpha = 0.94 * f.alpha;
  c.fillStyle = c.strokeStyle = color;
  c.shadowColor = color; c.shadowBlur = R * f.ppm * 0.12; // en px de pantalla
  c.lineJoin = 'round'; c.lineWidth = ROUND * R;
  starPath(c, R); c.fill(); c.stroke();
  c.shadowBlur = 0;
  // Textura + una luz difusa casi imperceptible.
  c.save();
  starPath(c, R); c.clip();
  GRAIN.setTransform(new DOMMatrix().scale(1 / f.ppm));
  c.fillStyle = GRAIN; c.fillRect(-R * 1.5, -R * 1.5, R * 3, R * 3);
  const soft = c.createLinearGradient(0, -R, 0, R);
  soft.addColorStop(0, withAlpha(WHITE, 0.07));
  soft.addColorStop(1, withAlpha(WHITE, 0));
  c.fillStyle = soft; c.fillRect(-R * 1.5, -R * 1.5, R * 3, R * 3);
  c.restore();
  c.restore();
}

// Íconos de los swatches con la misma geometría.
document.querySelectorAll('.ar-swatch').forEach((b) => {
  const d = STAR_PTS.map(([x, y], i) => `${i ? 'L' : 'M'}${(x * 10).toFixed(2)} ${(y * 10).toFixed(2)}`).join('') + 'Z';
  b.innerHTML = `<svg viewBox="-12 -12 24 24"><path d="${d}" fill="currentColor" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/></svg>`;
});

// Estado: cada estrella queda pegada a un landmark de la malla (sin offset → nunca se despega de la piel).
let color = 'black';
const stars = [
  { idx: 108, tilt: -0.2, color: 'black' }, // frente
  { idx: 425, tilt: 0.25, color: 'black' }, // mejilla
];
let face = null; // { pts: [[x,y,z]...] en px de canvas, ppm, xAxis, down, headN }
let view = { cw: 0, ch: 0, ox: 0, oy: 0, dw: 0, dh: 0 };

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
}
addEventListener('resize', resize); resize();

function layout() {
  const cw = canvas.width, ch = canvas.height, vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(cw / vw, ch / vh);
  view = { cw, ch, dw: vw * s, dh: vh * s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2 };
}

// Vectores 3D mínimos.
const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
const scale = (p, k) => [p[0] * k, p[1] * k, p[2] * k];
const norm = (p) => scale(p, 1 / (Math.hypot(...p) || 1));

// Landmark normalizado → px de canvas (vista espejo). z de MediaPipe va en la escala de x.
const toCanvas = (l) => [view.cw - (view.ox + l.x * view.dw), view.oy + l.y * view.dh, l.z * view.dw];

function readFace(lms) {
  const pts = lms.map(toCanvas);
  const xAxis = norm(sub(pts[IRIS_R], pts[IRIS_L]));
  const down = norm(sub(pts[152], pts[10])); // frente → mentón
  let headN = norm(cross(xAxis, down));
  if (headN[2] > 0) headN = scale(headN, -1); // normal hacia la cámara (z negativo)
  // Distancia interpupilar en 3D: la escala no cambia al girar la cabeza.
  return { pts, ppm: Math.hypot(...sub(pts[IRIS_R], pts[IRIS_L])) / IPD_MM, xAxis, down, headN };
}

// Plano tangente de la piel en el landmark de la estrella (normal desde sus vecinos 3D).
function skinFrame(s) {
  const { pts, ppm, xAxis, down, headN } = face, p = pts[s.idx];
  if (!s.nb) {
    s.nb = [...Array(FACE_POINTS).keys()].filter((i) => i !== s.idx)
      .sort((i, j) => Math.hypot(...sub(pts[i], p)) - Math.hypot(...sub(pts[j], p))).slice(0, 8);
  }
  const v = s.nb.map((i) => sub(pts[i], p));
  let n = [0, 0, 0];
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) {
    const c = cross(v[i], v[j]);
    n = sub(n, scale(c, dot(c, headN) < 0 ? 1 : -1)); // suma orientada hacia afuera de la cara
  }
  n = Math.hypot(...n) ? norm(n) : headN;
  s.n = s.n ? norm(sub(scale(s.n, 0.6), scale(n, -0.4))) : n; // suavizado anti-temblor
  n = s.n;
  const a = norm(sub(xAxis, scale(n, dot(xAxis, n))));
  let b = cross(n, a);
  if (dot(b, down) < 0) b = scale(b, -1);
  // Se desvanece cuando la piel queda de canto respecto a la cámara.
  const alpha = Math.min(1, Math.max(0, (-n[2] - 0.15) / 0.25));
  return { x: p[0], y: p[1], ax: a[0], ay: a[1], bx: b[0], by: b[1], ppm, alpha };
}

function render() {
  const { cw, ox, oy, dw, dh } = view;
  ctx.setTransform(-1, 0, 0, 1, cw, 0);
  ctx.drawImage(video, ox, oy, dw, dh);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!face) return;
  for (const s of stars) {
    const f = skinFrame(s);
    if (f.alpha > 0) drawStar(ctx, f, s.tilt, COLORS[s.color]);
  }
  drawDwell();
}

// Anillo de progreso en la punta del dedo mientras se mantiene sobre la cara.
function drawDwell() {
  if (!dwell.pos || dwell.fired || !face) return;
  const k = Math.min(1, (performance.now() - dwell.start) / DWELL_MS);
  const r = STAR_MM * 1.5 * face.ppm; // ~16 mm de radio: se ve alrededor del dedo
  ctx.save();
  ctx.lineWidth = r * 0.2; ctx.lineCap = 'round';
  ctx.strokeStyle = withAlpha(COLORS.black, 0.35); // pista oscura para contraste sobre la piel
  ctx.beginPath(); ctx.arc(dwell.pos[0], dwell.pos[1], r, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = PINK;
  ctx.beginPath(); ctx.arc(dwell.pos[0], dwell.pos[1], r, -Math.PI / 2, -Math.PI / 2 + k * Math.PI * 2); ctx.stroke();
  ctx.restore();
}

let landmarker, handmarker, lastT = -1, frame = 0;
let dwell = { pos: null, start: 0, fired: false };

function loop() {
  if (video.readyState >= 2) {
    if (!view.dw) layout();
    const t = performance.now();
    if (t !== lastT) {
      const r = landmarker.detectForVideo(video, t);
      face = r.faceLandmarks?.[0] ? readFace(r.faceLandmarks[0]) : null;
      // Manos cada 2 frames para no castigar el rendimiento en el teléfono.
      if (handmarker && frame++ % 2 === 0) trackFinger(handmarker.detectForVideo(video, t), t);
      lastT = t;
    }
    render();
  }
  requestAnimationFrame(loop);
}

// Punta del índice sobre la cara y quieta DWELL_MS → pega una estrella ahí.
function trackFinger(res, t) {
  const hand = res.landmarks?.[0];
  const tip = hand && toCanvas(hand[INDEX_TIP]);
  if (!face || !tip || !insideFace(face.pts, tip[0], tip[1])) { dwell = { pos: null, start: 0, fired: false }; return; }
  const moved = dwell.pos ? Math.hypot(tip[0] - dwell.pos[0], tip[1] - dwell.pos[1]) / face.ppm : Infinity;
  if (moved > STILL_MM) { dwell = { pos: tip, start: t, fired: false }; return; }
  if (!dwell.fired && t - dwell.start >= DWELL_MS) {
    dwell.fired = true; // no vuelve a pegar hasta que el dedo se mueva
    placeAt(dwell.pos[0], dwell.pos[1]);
  }
}

// Pega una estrella en el landmark más cercano a (px, py), si está dentro de la cara.
function placeAt(px, py) {
  if (!face || !insideFace(face.pts, px, py)) return;
  let best = -1, bd = Infinity;
  for (let i = 0; i < FACE_POINTS; i++) {
    const d = (face.pts[i][0] - px) ** 2 + (face.pts[i][1] - py) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  const p = face.pts[best];
  if (stars.some((s) => Math.hypot(...sub(face.pts[s.idx], p)) / face.ppm < SPACING_MM)) return;
  stars.push({ idx: best, tilt: (Math.random() - 0.5) * 0.6, color });
  navigator.vibrate?.(15);
}

// Tocar la pantalla también sirve.
canvas.addEventListener('pointerdown', (e) => {
  const r = canvas.getBoundingClientRect();
  placeAt(((e.clientX - r.left) * canvas.width) / r.width, ((e.clientY - r.top) * canvas.height) / r.height);
});
document.querySelectorAll('.ar-swatch').forEach((b) => b.addEventListener('click', () => {
  color = b.dataset.color;
  document.querySelectorAll('.ar-swatch').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
}));
$('undo').onclick = () => stars.pop();
$('clear').onclick = () => (stars.length = 0);
$('shot').onclick = () => canvas.toBlob(async (blob) => {
  const file = new File([blob], 'sweet-patch.png', { type: 'image/png' });
  if (navigator.canShare?.({ files: [file] })) return navigator.share({ files: [file] }).catch(() => {});
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: file.name });
  a.click();
});

$('start').onclick = async () => {
  const status = $('status');
  try {
    status.textContent = 'Cargando…';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    const files = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    landmarker = await FaceLandmarker.createFromOptions(files, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO', numFaces: 1,
    });
    // Si el modelo de manos falla, la app sigue funcionando con toques.
    handmarker = await HandLandmarker.createFromOptions(files, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO', numHands: 1,
    }).catch(() => null);
    addEventListener('resize', layout);
    video.addEventListener('resize', layout);
    $('intro').hidden = true; $('toolbar').hidden = false;
    loop();
  } catch (err) {
    status.textContent = location.protocol === 'https:' || location.hostname === 'localhost'
      ? `No se pudo abrir la cámara (${err.name || err}).`
      : 'La cámara necesita HTTPS. Abre esta página desde una URL https.';
  }
};
