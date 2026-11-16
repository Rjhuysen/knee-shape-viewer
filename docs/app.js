/* Knee Shape Viewer: clinical measurements -> predicted femur/tibia shape, desktop + WebXR (HoloLens 2).
 *
 * Model (see export_model.py):  V(x_A) = meanV + C_VA * C_AA^-1 * (x_A - mu_A)
 * All geometry math is done in the SSM frame (mm; x = proximal, -y = lateral, +z = posterior) and converted
 * to Babylon coordinates (y up) only when filling buffers: babylon (X, Y, Z) = (y, x, z) - knee centre.
 */
'use strict';
const B = BABYLON, G = BABYLON.GUI;

// ------------------------------------------------------------------ small vector helpers (SSM frame)
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vnorm = a => Math.hypot(a[0], a[1], a[2]);
const vunit = a => vscale(a, 1 / vnorm(a));
const projOut = (u, n) => vunit(vsub(u, vscale(n, vdot(u, n))));
const angDeg = (a, b) => Math.acos(Math.max(-1, Math.min(1, vdot(a, b)))) * 180 / Math.PI;
const pt = (V, i) => [V[3 * i], V[3 * i + 1], V[3 * i + 2]];

function centroid(V, idx) {
  const c = [0, 0, 0];
  for (const i of idx) { c[0] += V[3 * i]; c[1] += V[3 * i + 1]; c[2] += V[3 * i + 2]; }
  return vscale(c, 1 / idx.length);
}

function solve(A, b) {                       // Gaussian elimination with partial pivoting (A small, dense)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    [M[k], M[p]] = [M[p], M[k]];
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

function sphereCenter(V, idx) {              // least-squares sphere through the points
  const AtA = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], Atb = [0, 0, 0, 0];
  for (const i of idx) {
    const p = pt(V, i), row = [2 * p[0], 2 * p[1], 2 * p[2], 1], rhs = vdot(p, p);
    for (let a = 0; a < 4; a++) { Atb[a] += row[a] * rhs; for (let b = 0; b < 4; b++) AtA[a][b] += row[a] * row[b]; }
  }
  return solve(AtA, Atb).slice(0, 3);
}

function principalAxis(V, idx) {             // first principal axis, pointing proximally (+x)
  const c = centroid(V, idx), C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const i of idx) {
    const d = vsub(pt(V, i), c);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += d[a] * d[b];
  }
  let v = [1, 0, 0];
  for (let it = 0; it < 60; it++) v = vunit([vdot(C[0], v), vdot(C[1], v), vdot(C[2], v)]);
  if (v[0] < 0) v = vscale(v, -1);
  return { c, v };
}

// chi-square quantile (Wilson-Hilferty) for the plausibility indicator
const chi2q = (k, z) => k * Math.pow(1 - 2 / (9 * k) + z * Math.sqrt(2 / (9 * k)), 3);

// ------------------------------------------------------------------ app state
const S = {
  meta: null, meanV: null, CVX: null, faces: null, nM: 0,
  active: new Set(), input: [],             // input values (used for active measurements)
  shown: [],                                // displayed values (input if active, else conditional expectation)
  V: null, mesh: null, plaus: null,
  show: { overlayHKA: true, overlayFlex: true, overlayLen: true, ghost: false, deviation: false, xray: false },
};
const TOGGLES = [
  ['overlayHKA', 'HKA axes', '#ffd54a'], ['overlayFlex', 'Flexion axes', '#4dd0e1'],
  ['overlayLen', 'FL / TL lengths', '#f48fb1'], ['ghost', 'Mean shape ghost', '#7fa7ff'],
  ['deviation', 'Deviation colours', null], ['xray', 'Translucent bones', null],
];

let scene, engine, root, ghost, boneMat, ghostMat, C0, refPosB, refNormB, normalSign = 1;
const overlay = {};
let xrPanel = null, xrHelper = null, inXR = false;

// ------------------------------------------------------------------ model
function predict() {
  const m = S.meta.measurements, nM = S.nM;
  const A = [...S.active].sort((a, b) => a - b);
  const CXX = S.meta.C_XX, d = A.map(j => S.input[j] - m[j].mean);
  const alpha = A.length ? solve(A.map(i => A.map(j => CXX[i][j])), d) : [];
  // conditional expectation of the unticked measurements (same alpha)
  for (let i = 0; i < nM; i++) {
    if (S.active.has(i)) { S.shown[i] = S.input[i]; continue; }
    let v = m[i].mean;
    A.forEach((j, k) => { v += CXX[i][j] * alpha[k]; });
    S.shown[i] = v;
  }
  // Mahalanobis distance of the ticked inputs
  const d2 = A.reduce((s, _, k) => s + d[k] * alpha[k], 0);
  S.plaus = { k: A.length, d: Math.sqrt(Math.max(d2, 0)), q95: chi2q(A.length, 1.645), q999: chi2q(A.length, 3.09) };
  // vertices
  const V = S.V, mv = S.meanV, CVX = S.CVX, n = mv.length;
  for (let r = 0; r < n; r++) {
    let v = mv[r];
    const base = r * nM;
    for (let k = 0; k < A.length; k++) v += CVX[base + A[k]] * alpha[k];
    V[r] = v;
  }
}

function measureMesh(V) {
  const L = S.meta.landmarks;
  const hip = sphereCenter(V, L.HEAD);
  const med = centroid(V, L.MED_EPI), lat = centroid(V, L.LAT_EPI);
  const kneeF = vscale(vadd(med, lat), 0.5), kneeT = centroid(V, L.TPLAT), ankle = centroid(V, L.TDIST);
  const ml = vunit(vsub(lat, med));
  // flexion: shaft axes projected on the sagittal plane (normal = epicondylar axis)
  const fS = principalAxis(V, L.FEM_SHAFT), tS = principalAxis(V, L.TIB_SHAFT);
  const f = projOut(fS.v, ml), t = projOut(tS.v, ml);
  const flex = -Math.sign(vdot(t, vcross(ml, f))) * angDeg(f, t);
  // HKA: mechanical axes projected on the coronal plane
  const fa = vsub(hip, kneeF), ta = vsub(kneeT, ankle);
  const ap = vunit(vcross(ml, vunit(fa)));
  const a = projOut(fa, ap), b = projOut(ta, ap);
  const hka = 180 + S.meta.hkaSign * Math.sign(vdot(vcross(a, b), ap)) * angDeg(a, b);
  // FL: hip centre -> most distal femur point along the mechanical axis; TL: plateau centre -> plafond centre
  const fau = vunit(fa);
  let mn = Infinity;
  for (let i = S.meta.femur[0]; i < S.meta.femur[1]; i++) mn = Math.min(mn, vdot(vsub(pt(V, i), hip), fau));
  return { hip, kneeF, kneeT, ankle, med, lat, ml, ap, fS, tS, f, t, a, b, flex, hka,
           FL: -mn, flEnd: vadd(hip, vscale(fau, mn)), TL: vnorm(vsub(kneeT, ankle)) };
}

// ------------------------------------------------------------------ scene
const toB = p => new B.Vector3(p[1] - C0[1], p[0] - C0[0], p[2] - C0[2]);

function fillPositions(V, out) {
  for (let i = 0; i < V.length / 3; i++) {
    out[3 * i] = V[3 * i + 1] - C0[1]; out[3 * i + 1] = V[3 * i] - C0[0]; out[3 * i + 2] = V[3 * i + 2] - C0[2];
  }
  return out;
}

function emissive(name, hex, alpha = 1) {
  const m = new B.StandardMaterial(name, scene);
  m.disableLighting = true; m.emissiveColor = B.Color3.FromHexString(hex); m.alpha = alpha;
  return m;
}

function makeSeg(name, mat, radius) {
  const m = B.MeshBuilder.CreateCylinder(name, { height: 1, diameter: 1, tessellation: 10 }, scene);
  Object.assign(m, { parent: root, material: mat, isPickable: false, renderingGroupId: 1 });
  m.rotationQuaternion = new B.Quaternion(); m.metadata = { r: radius };
  return m;
}
function setSeg(m, p, q) {
  const a = toB(p), b = toB(q), d = b.subtract(a), len = d.length();
  m.position.copyFrom(a.add(b).scale(0.5));
  m.scaling.set(2 * m.metadata.r, Math.max(len, 1e-3), 2 * m.metadata.r);
  B.Quaternion.FromUnitVectorsToRef(B.Vector3.Up(), d.normalize(), m.rotationQuaternion);
}
function makeBall(name, mat, r) {
  const m = B.MeshBuilder.CreateSphere(name, { diameter: 2 * r, segments: 8 }, scene);
  Object.assign(m, { parent: root, material: mat, isPickable: false, renderingGroupId: 1 });
  return m;
}
function arcPath(c, u, v, r, n = 24) {        // arc from direction u to v (unit vectors), radius r, centre c
  const raw = vsub(v, vscale(u, vdot(u, v)));
  const w = vnorm(raw) > 1e-9 ? vunit(raw) : [0, 0, 0], th = Math.acos(Math.max(-1, Math.min(1, vdot(u, v))));
  const out = [];
  for (let s = 0; s <= n; s++) {
    const t = th * s / n;
    out.push(toB(vadd(c, vscale(vadd(vscale(u, Math.cos(t)), vscale(w, Math.sin(t))), r))));
  }
  return out;
}
function makeArc(name, mat, r) {
  const dummy = Array.from({ length: 25 }, (_, i) => new B.Vector3(i, 0, 0));
  const m = B.MeshBuilder.CreateTube(name, { path: dummy, radius: r, tessellation: 8, updatable: true }, scene);
  Object.assign(m, { parent: root, material: mat, isPickable: false, renderingGroupId: 1 });
  m.metadata = { r };
  return m;
}
function setArc(m, path) {
  B.MeshBuilder.CreateTube(m.name, { path, radius: m.metadata.r, tessellation: 8, instance: m });
}
function makeLabel(name, color) {
  const plane = B.MeshBuilder.CreatePlane(name, { width: 120, height: 30 }, scene);
  Object.assign(plane, { parent: root, isPickable: false, renderingGroupId: 1, billboardMode: B.Mesh.BILLBOARDMODE_ALL });
  const dt = new B.DynamicTexture(name + '_dt', { width: 512, height: 128 }, scene, true);
  dt.hasAlpha = true;
  const mat = new B.StandardMaterial(name + '_m', scene);
  Object.assign(mat, { diffuseTexture: dt, emissiveColor: B.Color3.White(), disableLighting: true,
                       useAlphaFromDiffuseTexture: true, backFaceCulling: false });
  plane.material = mat;
  plane.metadata = {
    set(text) {
      const ctx = dt.getContext();
      ctx.clearRect(0, 0, 512, 128);
      ctx.fillStyle = 'rgba(10,12,16,0.72)';
      ctx.beginPath(); ctx.roundRect(4, 14, 504, 100, 22); ctx.fill();
      ctx.fillStyle = color; ctx.font = 'bold 60px Segoe UI, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 256, 66);
      dt.update();
    },
  };
  return plane;
}

function buildOverlays() {
  const R = 1.6;
  const y = emissive('ovY', '#ffd54a'), c = emissive('ovC', '#4dd0e1'), p = emissive('ovP', '#f48fb1');
  const yDim = emissive('ovY2', '#ffd54a', 0.55);
  overlay.hka = {
    nodes: [], fem: makeSeg('hkaFem', y, R), tib: makeSeg('hkaTib', y, R), ext: makeSeg('hkaExt', yDim, R * 0.7),
    hip: makeBall('hipC', y, 6), knee: makeBall('kneeC', y, 4.5), kneeT: makeBall('kneeTC', y, 4.5), ankle: makeBall('ankC', y, 5),
    arc: makeArc('hkaArc', y, R), label: makeLabel('hkaLbl', '#ffd54a'),
  };
  overlay.flex = {
    fem: makeSeg('flxFem', c, R), tib: makeSeg('flxTib', c, R), epi: makeSeg('flxEpi', c, R),
    med: makeBall('medE', c, 4), lat: makeBall('latE', c, 4), arc: makeArc('flxArc', c, R), label: makeLabel('flxLbl', '#4dd0e1'),
  };
  overlay.len = {
    fl: makeSeg('flLine', p, R), flA: makeSeg('flCapA', p, R), flB: makeSeg('flCapB', p, R), flLabel: makeLabel('flLbl', '#f48fb1'),
    tl: makeSeg('tlLine', p, R), tlA: makeSeg('tlCapA', p, R), tlB: makeSeg('tlCapB', p, R), tlLabel: makeLabel('tlLbl', '#f48fb1'),
  };
  for (const g of Object.values(overlay)) g.nodes = Object.values(g).filter(n => n instanceof B.AbstractMesh);
}

function updateOverlays(mm) {
  const fmt = (v, d = 1) => v.toFixed(d);
  // HKA (coronal plane)
  const o = overlay.hka, fdir = vunit(vsub(mm.hip, mm.kneeF));
  setSeg(o.fem, mm.hip, mm.kneeF); setSeg(o.tib, mm.kneeT, mm.ankle);
  setSeg(o.ext, mm.kneeF, vadd(mm.kneeF, vscale(fdir, -150)));
  o.hip.position = toB(mm.hip); o.knee.position = toB(mm.kneeF); o.kneeT.position = toB(mm.kneeT); o.ankle.position = toB(mm.ankle);
  setArc(o.arc, arcPath(mm.kneeF, vscale(mm.a, -1), vscale(mm.b, -1), 95));
  o.label.position = toB(vadd(vadd(mm.kneeF, vscale(fdir, -120)), vscale(mm.ml, -70)));
  o.label.metadata.set(`HKA ${fmt(mm.hka)}°`);
  // Flexion (sagittal plane): shaft axes, epicondylar axis, arc at the knee
  const f = overlay.flex;
  setSeg(f.fem, vadd(mm.fS.c, vscale(mm.fS.v, 170)), vadd(mm.kneeF, vscale(mm.f, 30)));
  setSeg(f.tib, vadd(mm.kneeF, vscale(mm.t, -30)), vadd(mm.tS.c, vscale(mm.tS.v, -170)));
  setSeg(f.epi, mm.med, mm.lat); f.med.position = toB(mm.med); f.lat.position = toB(mm.lat);
  setArc(f.arc, arcPath(mm.kneeF, vscale(mm.f, -1), vscale(mm.t, -1), 75));
  f.label.position = toB(vadd(vadd(mm.kneeF, vscale(mm.f, -60)), vscale(vcross(mm.ml, mm.f), -110)));
  f.label.metadata.set(`Flexion ${fmt(mm.flex)}°`);
  // Lengths: dimension lines offset medially (+y side) with end caps
  const l = overlay.len, off = vscale(mm.ml, -85), cap = vscale(mm.ml, -20);
  const a1 = vadd(mm.hip, off), a2 = vadd(mm.flEnd, off);
  setSeg(l.fl, a1, a2); setSeg(l.flA, vsub(a1, cap), vadd(a1, cap)); setSeg(l.flB, vsub(a2, cap), vadd(a2, cap));
  l.flLabel.position = toB(vadd(vscale(vadd(a1, a2), 0.5), vscale(mm.ml, -70)));
  l.flLabel.metadata.set(`FL ${fmt(mm.FL)} mm`);
  const b1 = vadd(mm.kneeT, off), b2 = vadd(mm.ankle, off);
  setSeg(l.tl, b1, b2); setSeg(l.tlA, vsub(b1, cap), vadd(b1, cap)); setSeg(l.tlB, vsub(b2, cap), vadd(b2, cap));
  l.tlLabel.position = toB(vadd(vscale(vadd(b1, b2), 0.5), vscale(mm.ml, -70)));
  l.tlLabel.metadata.set(`TL ${fmt(mm.TL)} mm`);
}

function applyVisibility() {
  const sh = S.show;
  overlay.hka.nodes.forEach(n => n.setEnabled(sh.overlayHKA));
  overlay.flex.nodes.forEach(n => n.setEnabled(sh.overlayFlex));
  overlay.len.nodes.forEach(n => n.setEnabled(sh.overlayLen));
  ghost.setEnabled(sh.ghost);
  boneMat.alpha = sh.xray ? 0.45 : 1;
  S.mesh.useVertexColors = sh.deviation;
  boneMat.diffuseColor = sh.deviation ? B.Color3.White() : B.Color3.FromHexString('#e9dfc8');
  document.getElementById('legend').hidden = !sh.deviation;
}

const posBuf = { p: null, n: null, c: null };
function updateMesh() {
  fillPositions(S.V, posBuf.p);
  B.VertexData.ComputeNormals(posBuf.p, S.faces, posBuf.n);
  S.mesh.updateVerticesData(B.VertexBuffer.PositionKind, posBuf.p, false, false);
  S.mesh.updateVerticesData(B.VertexBuffer.NormalKind, posBuf.n, false, false);
  if (S.show.deviation) {
    const c = posBuf.c, n = posBuf.p.length / 3;
    for (let i = 0; i < n; i++) {
      const j = 3 * i;
      const d = normalSign * ((posBuf.p[j] - refPosB[j]) * refNormB[j] + (posBuf.p[j + 1] - refPosB[j + 1]) * refNormB[j + 1] +
                              (posBuf.p[j + 2] - refPosB[j + 2]) * refNormB[j + 2]);
      const t = Math.max(-1, Math.min(1, d / 10));
      const [r, g, b] = t < 0 ? [0.95 + t * 0.72, 0.95 + t * 0.5, 0.95 + t * 0.0] : [0.95 - t * 0.05, 0.95 - t * 0.7, 0.95 - t * 0.72];
      c[4 * i] = r; c[4 * i + 1] = g; c[4 * i + 2] = b; c[4 * i + 3] = 1;
    }
    S.mesh.updateVerticesData(B.VertexBuffer.ColorKind, c, false, false);
  }
  S.mesh.refreshBoundingInfo();
}

// ------------------------------------------------------------------ update cycle
let pending = false;
function refresh() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    predict();
    updateMesh();
    const mm = measureMesh(S.V);
    updateOverlays(mm);
    applyVisibility();
    syncHTML(mm);
    if (xrPanel) xrPanel.sync(mm);
  });
}

function setActive(j, on) {
  if (on) { S.input[j] = S.meta.measurements[j].mean; S.active.add(j); } else S.active.delete(j);   // always start at the mean
  refresh();
}
function setInput(j, v) { S.input[j] = clampToRange(j, v); if (!S.active.has(j)) S.active.add(j); refresh(); }
function preset(name) {
  const m = S.meta.measurements;
  if (name === 'best') S.active = new Set(S.meta.defaultActive.map(k => m.findIndex(x => x.key === k)));
  if (name === 'all') S.active = new Set(m.map((_, i) => i));
  if (name === 'none') S.active = new Set();
  for (const j of S.active) S.input[j] = m[j].mean;
  refresh();
}
function resetValues() { S.meta.measurements.forEach((x, i) => { S.input[i] = x.mean; }); refresh(); }
function plausText() {
  const p = S.plaus;
  if (!p.k) return ['ok', 'No inputs: showing the population-mean shape.'];
  const msg = `Mahalanobis distance of the ${p.k} input${p.k > 1 ? 's' : ''}: ${p.d.toFixed(1)}`;
  if (p.d * p.d > p.q999) return ['bad', `${msg} — far outside the training data, prediction is an extrapolation.`];
  if (p.d * p.d > p.q95) return ['warn', `${msg} — unusual combination (outside 95 % of cases).`];
  return ['ok', `${msg} — within the range of the training data.`];
}
// angles: mean ± 2 SD; lengths/distances (mm): observed range ± 0.5 SD
const sliderRange = m => m.unit === '°' ? [m.mean - 2 * m.sd, m.mean + 2 * m.sd] : [m.min - 0.5 * m.sd, m.max + 0.5 * m.sd];
const clampToRange = (j, v) => { const [lo, hi] = sliderRange(S.meta.measurements[j]); return Math.min(hi, Math.max(lo, v)); };
const fmtVal = (v, m) => `${v.toFixed(m.unit === 'mm' && Math.abs(v) > 100 ? 0 : 1)} ${m.unit}`;

// ------------------------------------------------------------------ HTML panel (desktop)
const html = {};
function buildHTML() {
  const list = document.getElementById('measList');
  html.rows = S.meta.measurements.map((m, j) => {
    const [lo, hi] = sliderRange(m);
    const row = document.createElement('div');
    row.className = 'meas';
    row.innerHTML = `<input type="checkbox" aria-label="use ${m.key}"><div class="lbl" title="${m.name}"><b>${m.key}</b><span>${m.name}</span></div>
      <div class="val"></div><input type="range" min="${lo}" max="${hi}" step="0.1" aria-label="${m.key} value">`;
    const [cb, , val, sl] = row.children;
    cb.addEventListener('change', () => setActive(j, cb.checked));
    sl.addEventListener('input', () => setInput(j, +sl.value));
    list.appendChild(row);
    return { row, cb, val, sl };
  });
  document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => preset(b.dataset.preset)));
  document.getElementById('resetValues').addEventListener('click', resetValues);
  const tg = document.getElementById('toggles');
  for (const [key, label, color] of TOGGLES) {
    const l = document.createElement('label');
    l.innerHTML = `<input type="checkbox" ${S.show[key] ? 'checked' : ''}>${color ? `<span class="sw" style="background:${color}"></span>` : ''}${label}`;
    l.firstChild.addEventListener('change', e => { S.show[key] = e.target.checked; refresh(); });
    tg.appendChild(l);
    html['tg_' + key] = l.firstChild;
  }
  const v = S.meta.validation;
  document.getElementById('validNote').textContent =
    `Mesh landmarks vs the clinical table (305 cases): HKA r = ${v.HKA.r.toFixed(3)} (bias ${v.HKA.bias.toFixed(1)}°), ` +
    `FL r = ${v.FL.r.toFixed(3)} (bias +${v.FL.bias.toFixed(0)} mm), TL r = ${v.TL.r.toFixed(3)} (bias +${v.TL.bias.toFixed(0)} mm). ` +
    `FL/TL use approximate landmarks (hip centre → distal femur, plateau → plafond centre), hence the constant offset.`;
}
function syncHTML(mm) {
  S.meta.measurements.forEach((m, j) => {
    const r = html.rows[j], on = S.active.has(j);
    r.row.classList.toggle('on', on); r.cb.checked = on;
    if (document.activeElement !== r.sl) r.sl.value = S.shown[j];
    r.val.textContent = fmtVal(S.shown[j], m);
  });
  for (const [key] of TOGGLES) html['tg_' + key].checked = S.show[key];
  const [cls, txt] = plausText();
  const pl = document.getElementById('plaus'); pl.className = 'plaus ' + cls; pl.textContent = txt;
  const idx = k => S.meta.measurements.findIndex(x => x.key === k);
  const rows = [['HKA', mm.hka, '°'], ['Flexion', mm.flex, '°'], ['FL', mm.FL, 'mm'], ['TL', mm.TL, 'mm']];
  document.querySelector('#meshTable tbody').innerHTML = rows.map(([k, v, u]) =>
    `<tr><td>${k}</td><td>${S.shown[idx(k)].toFixed(1)} ${u}${S.active.has(idx(k)) ? '' : ' <span style="color:var(--muted)">(exp.)</span>'}</td><td>${v.toFixed(1)} ${u}</td></tr>`).join('');
}

// ------------------------------------------------------------------ XR panel (Babylon GUI on a floating plane)
function buildXRPanel() {
  const W = 0.46, H = 0.66, RES = 1600;
  const handle = B.MeshBuilder.CreateBox('panelHandle', { width: W * 0.6, height: 0.03, depth: 0.02 }, scene);
  handle.material = emissive('handleMat', '#5fb3ff', 0.9);
  handle.isNearGrabbable = true;
  const plane = B.MeshBuilder.CreatePlane('panel', { width: W, height: H }, scene);
  plane.parent = handle; plane.position.y = -H / 2 - 0.025;
  plane.isNearPickable = true;
  const adt = G.AdvancedDynamicTexture.CreateForMesh(plane, Math.round(RES * W / H), RES);
  const drag = new B.SixDofDragBehavior();
  drag.rotateWithMotionController = false;   // keep the panel upright
  handle.addBehavior(drag);

  const bg = new G.Rectangle('bg');
  Object.assign(bg, { background: '#1a1e25ee', color: '#5fb3ff', thickness: 3, cornerRadius: 24 });
  adt.addControl(bg);
  const outer = new G.StackPanel('outer');
  Object.assign(outer, { width: '96%', paddingTop: '20px' });
  outer.verticalAlignment = G.Control.VERTICAL_ALIGNMENT_TOP;
  bg.addControl(outer);

  const btn = (text, w, h, onClick, fs = 34) => {
    const b = G.Button.CreateSimpleButton('b' + text, text);
    Object.assign(b, { width: w, height: h, color: '#e6e8eb', background: '#252b34', cornerRadius: 14, thickness: 2, fontSize: fs });
    b.paddingLeft = b.paddingRight = '6px'; b.paddingTop = b.paddingBottom = '6px';
    b.onPointerUpObservable.add(onClick);
    return b;
  };
  const txt = (t, h, fs, color = '#e6e8eb') => {
    const tb = new G.TextBlock('', t);
    Object.assign(tb, { height: h, fontSize: fs, color, textWrapping: true });
    return tb;
  };

  const title = txt('Knee Shape Viewer', '60px', 44); outer.addControl(title);
  const tabs = new G.StackPanel('tabs'); tabs.isVertical = false; tabs.height = '100px'; outer.addControl(tabs);
  const plaus = txt('', '90px', 28, '#9aa3ad'); outer.addControl(plaus);
  const content = new G.StackPanel('content'); content.width = '100%'; outer.addControl(content);

  let page = 'sliders', sliderPage = 0;
  const PER_PAGE = 6;
  const tabBtns = {};
  for (const [id, label] of [['sliders', 'Sliders'], ['select', 'Inputs'], ['display', 'Display']]) {
    tabBtns[id] = btn(label, '360px', '96px', () => { page = id; render(); refresh(); }, 38);
    tabs.addControl(tabBtns[id]);
  }

  let sliders = [];   // {j, slider, value}
  let syncing = false;
  let lastMM = null;

  function render() {
    content.clearControls(); sliders = [];
    for (const [id, b] of Object.entries(tabBtns)) b.background = id === page ? '#2f5d8a' : '#252b34';
    const meas = S.meta.measurements;
    if (page === 'sliders') {
      const act = [...S.active].sort((a, b) => a - b);
      const pages = Math.max(1, Math.ceil(act.length / PER_PAGE));
      sliderPage = Math.min(sliderPage, pages - 1);
      if (!act.length) content.addControl(txt('No inputs selected.\nOpen "Inputs" to choose measurements.', '200px', 34, '#9aa3ad'));
      for (const j of act.slice(sliderPage * PER_PAGE, (sliderPage + 1) * PER_PAGE)) {
        const m = meas[j], [lo, hi] = sliderRange(m), step = m.unit === 'mm' && m.sd > 10 ? 2 : 0.5;
        const head = new G.Grid(); head.height = '62px'; head.addColumnDefinition(0.62); head.addColumnDefinition(0.38);
        const name = txt(`${m.key}  ·  ${m.name}`, '62px', 30); name.textHorizontalAlignment = G.Control.HORIZONTAL_ALIGNMENT_LEFT;
        const value = txt('', '62px', 36, '#5fb3ff'); value.textHorizontalAlignment = G.Control.HORIZONTAL_ALIGNMENT_RIGHT;
        head.addControl(name, 0, 0); head.addControl(value, 0, 1);
        const row = new G.StackPanel(); row.isVertical = false; row.height = '110px';
        const minus = btn('−', '110px', '100px', () => setInput(j, Math.max(lo, S.shown[j] - step)), 52);
        const plus = btn('+', '110px', '100px', () => setInput(j, Math.min(hi, S.shown[j] + step)), 52);
        const sl = new G.Slider();
        Object.assign(sl, { minimum: lo, maximum: hi, value: S.shown[j], width: '820px', height: '100px', color: '#5fb3ff',
                            background: '#39414d', thumbColor: '#e6e8eb', thumbWidth: '60px', barOffset: '30px', isThumbCircle: true });
        sl.paddingLeft = sl.paddingRight = '16px';
        sl.onValueChangedObservable.add(v => { if (!syncing) setInput(j, v); });
        row.addControl(minus); row.addControl(sl); row.addControl(plus);
        content.addControl(head); content.addControl(row);
        sliders.push({ j, sl, value });
      }
      if (pages > 1) {
        const nav = new G.StackPanel(); nav.isVertical = false; nav.height = '100px';
        nav.addControl(btn('◀ prev', '300px', '90px', () => { sliderPage = (sliderPage + pages - 1) % pages; render(); refresh(); }));
        nav.addControl(txt(`${sliderPage + 1} / ${pages}`, '90px', 32)).width = '200px';
        nav.addControl(btn('next ▶', '300px', '90px', () => { sliderPage = (sliderPage + 1) % pages; render(); refresh(); }));
        content.addControl(nav);
      }
    } else if (page === 'select') {
      content.addControl(txt('Tap to use a measurement as input', '60px', 30, '#9aa3ad'));
      const grid = new G.Grid(); grid.height = `${Math.ceil(meas.length / 3) * 92}px`;
      for (let c = 0; c < 3; c++) grid.addColumnDefinition(1 / 3);
      for (let r = 0; r < Math.ceil(meas.length / 3); r++) grid.addRowDefinition(92, true);
      meas.forEach((m, j) => {
        const b = btn(m.key, '100%', '100%', () => { setActive(j, !S.active.has(j)); render(); }, 32);
        b.background = S.active.has(j) ? '#2f5d8a' : '#252b34';
        grid.addControl(b, Math.floor(j / 3), j % 3);
      });
      content.addControl(grid);
      const pr = new G.StackPanel(); pr.isVertical = false; pr.height = '110px';
      pr.addControl(btn('Best 4', '260px', '100px', () => { preset('best'); render(); }));
      pr.addControl(btn('All 27', '260px', '100px', () => { preset('all'); render(); }));
      pr.addControl(btn('None', '260px', '100px', () => { preset('none'); render(); }));
      pr.addControl(btn('Reset', '260px', '100px', resetValues));
      content.addControl(pr);
    } else {
      const grid = new G.Grid(); grid.height = `${Math.ceil(TOGGLES.length / 2) * 110}px`;
      grid.addColumnDefinition(0.5); grid.addColumnDefinition(0.5);
      for (let r = 0; r < Math.ceil(TOGGLES.length / 2); r++) grid.addRowDefinition(110, true);
      TOGGLES.forEach(([key, label, color], i) => {
        const b = btn(label, '100%', '100%', () => { S.show[key] = !S.show[key]; render(); refresh(); }, 32);
        b.background = S.show[key] ? '#2f5d8a' : '#252b34';
        if (color) b.color = color;
        grid.addControl(b, Math.floor(i / 2), i % 2);
      });
      content.addControl(grid);
      content.addControl(txt('Model size', '60px', 30, '#9aa3ad'));
      const sc = new G.StackPanel(); sc.isVertical = false; sc.height = '110px';
      for (const [l, s] of [['Life size', 1], ['1 : 2', 0.5], ['1 : 4', 0.25]]) sc.addControl(btn(l, '300px', '100px', () => setModelScale(s)));
      content.addControl(sc);
      content.addControl(btn('Bring model and panel in front of me', '100%', '100px', placeForUser));
      content.addControl(txt('Grab the bones to move/rotate them (pinch or air-tap and hold). ' +
                             'Grab the blue bar above this panel to move the panel.', '140px', 28, '#9aa3ad'));
      const mt = txt('', '170px', 30, '#e6e8eb'); mt.name = 'meshText'; content.addControl(mt);
    }
    if (lastMM) sync(lastMM);
  }

  function sync(mm) {
    lastMM = mm;
    syncing = true;
    const meas = S.meta.measurements;
    for (const s of sliders) { s.sl.value = S.shown[s.j]; s.value.text = fmtVal(S.shown[s.j], meas[s.j]); }
    syncing = false;
    plaus.text = plausText()[1];
    const mt = content.getDescendants(false, c => c.name === 'meshText')[0];
    if (mt) mt.text = `On the predicted mesh: HKA ${mm.hka.toFixed(1)}°, Flexion ${mm.flex.toFixed(1)}°\n` +
                      `FL ${mm.FL.toFixed(0)} mm, TL ${mm.TL.toFixed(0)} mm`;
  }

  render();
  handle.setEnabled(false);
  return { handle, sync, render, setPage(p) { page = p; render(); } };
}

function setModelScale(s) { root.scaling.setAll(0.001 * s); }

function placeForUser() {
  const cam = scene.activeCamera;
  const fwd = cam.getDirection(B.Axis.Z); fwd.y = 0; fwd.normalize();
  const right = B.Vector3.Cross(B.Axis.Y, fwd).normalize();
  const head = cam.globalPosition.clone();
  const yaw = Math.atan2(fwd.x, fwd.z);
  root.position = head.add(fwd.scale(1.0)).add(right.scale(0.25)).add(new B.Vector3(0, -0.25, 0));
  root.rotationQuaternion = B.Quaternion.FromEulerAngles(0, yaw, 0);
  if (xrPanel) {
    xrPanel.handle.position = head.add(fwd.scale(0.6)).add(right.scale(-0.35)).add(new B.Vector3(0, 0.05, 0));
    xrPanel.handle.rotationQuaternion = B.Quaternion.FromEulerAngles(0, yaw, 0);
  }
}

// ------------------------------------------------------------------ boot
// The bundle holds only the OLS relationship (mean shape, C_VX, C_XX, measurement stats, landmarks).
// The SSM itself (ssmV, eigenvalues, mode weights, per-case data) is never exported.
function parseBundle(buf) {
  const head = new Uint8Array(buf, 0, 4);
  if (String.fromCharCode(...head) !== 'KSV1') throw new Error('not a knee_shape_model.ksv file');
  const jl = new DataView(buf).getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jl)));
  return { meta, bin: buf.slice(8 + jl) };
}

async function loadModel() {
  const r = await fetch('model/knee_shape_model.ksv');
  if (!r.ok) throw new Error(`model file not found (HTTP ${r.status}); run export_model.py`);
  const { meta, bin } = parseBundle(await r.arrayBuffer());
  S.meta = meta; S.nM = meta.nMeas;
  S.meanV = new Float32Array(bin, meta.bin.meanV[0], meta.bin.meanV[1]);
  S.CVX = new Float32Array(bin, meta.bin.C_VX[0], meta.bin.C_VX[1]);
  S.faces = Array.from(new Uint16Array(bin, meta.bin.faces[0], meta.bin.faces[1]));
  S.V = new Float32Array(S.meanV.length);
  S.input = meta.measurements.map(m => m.mean);
  S.shown = S.input.slice();
  S.active = new Set(meta.defaultActive.map(k => meta.measurements.findIndex(m => m.key === k)));
}

async function main() {
  await loadModel();
  const canvas = document.getElementById('renderCanvas');
  engine = new B.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true, xrCompatible: true });
  scene = new B.Scene(engine);
  scene.clearColor = new B.Color4(0.07, 0.08, 0.1, 1);

  // knee centre of the mean shape = pivot
  const L = S.meta.landmarks;
  C0 = vscale(vadd(centroid(S.meanV, L.MED_EPI), centroid(S.meanV, L.LAT_EPI)), 0.5);

  // predicted mesh = root of everything model-related (grab target)
  const n = S.meanV.length;
  posBuf.p = new Float32Array(n); posBuf.n = new Float32Array(n); posBuf.c = new Float32Array(n / 3 * 4).fill(1);
  // make the faces wind outward
  refPosB = fillPositions(S.meanV, new Float32Array(n));
  refNormB = new Float32Array(n);
  B.VertexData.ComputeNormals(refPosB, S.faces, refNormB);
  let out = 0; const cx = [0, 0, 0];
  for (let i = 0; i < n / 3; i++) for (let a = 0; a < 3; a++) cx[a] += refPosB[3 * i + a] * 3 / n;
  for (let i = 0; i < n / 3; i++) for (let a = 0; a < 3; a++) out += refNormB[3 * i + a] * (refPosB[3 * i + a] - cx[a]);
  if (out < 0) {          // inward: flip winding and recompute
    for (let f = 0; f < S.faces.length; f += 3) [S.faces[f + 1], S.faces[f + 2]] = [S.faces[f + 2], S.faces[f + 1]];
    B.VertexData.ComputeNormals(refPosB, S.faces, refNormB);
  }

  root = new B.Mesh('bones', scene);
  const vd = new B.VertexData();
  vd.positions = refPosB.slice(); vd.indices = S.faces; vd.normals = refNormB.slice(); vd.colors = posBuf.c.slice();
  vd.applyToMesh(root, true);
  S.mesh = root;
  root.scaling.setAll(0.001);
  root.isNearGrabbable = true;
  boneMat = new B.StandardMaterial('bone', scene);
  boneMat.diffuseColor = B.Color3.FromHexString('#e9dfc8');
  boneMat.specularColor = new B.Color3(0.15, 0.15, 0.15);
  boneMat.backFaceCulling = false;
  root.material = boneMat;

  ghost = new B.Mesh('ghost', scene);
  const gd = new B.VertexData(); gd.positions = refPosB; gd.indices = S.faces; gd.normals = refNormB; gd.applyToMesh(ghost);
  ghost.parent = root; ghost.isPickable = false;
  ghostMat = new B.StandardMaterial('ghostMat', scene);
  Object.assign(ghostMat, { diffuseColor: B.Color3.FromHexString('#7fa7ff'), emissiveColor: B.Color3.FromHexString('#1d3566'), alpha: 0.3 });
  ghost.material = ghostMat;

  const cam = new B.ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 2.1, 1.35, B.Vector3.Zero(), scene);
  Object.assign(cam, { minZ: 0.01, lowerRadiusLimit: 0.2, upperRadiusLimit: 5, wheelPrecision: 300, pinchPrecision: 300, panningSensibility: 2000 });
  cam.attachControl(canvas, true);
  const hemi = new B.HemisphericLight('hemi', new B.Vector3(0.3, 1, -0.4), scene);
  hemi.intensity = 0.75; hemi.groundColor = new B.Color3(0.25, 0.25, 0.3);
  const dir = new B.DirectionalLight('dir', new B.Vector3(-0.4, -0.6, 1), scene);
  dir.intensity = 0.6;
  dir.parent = cam;   // headlight-ish

  buildOverlays();
  buildHTML();
  xrPanel = buildXRPanel();
  if (new URLSearchParams(location.search).has('panel')) {     // debug: show the HoloLens panel on desktop
    xrPanel.handle.setEnabled(true);
    xrPanel.handle.position.set(-0.42, 0.33, 0);
    const page = new URLSearchParams(location.search).get('panel');
    if (page) xrPanel.setPage(page);
  }
  await setupXR();
  refresh();
  document.getElementById('loading').remove();
  engine.runRenderLoop(() => scene.render());
  window.addEventListener('resize', () => engine.resize());
  window.__viewer = { S, scene, preset, setInput, setActive, refresh };   // for debugging / tests
}

async function setupXR() {
  const hint = document.getElementById('xrHint');
  let mode = null;
  try {
    if (await B.WebXRSessionManager.IsSessionSupportedAsync('immersive-ar')) mode = 'immersive-ar';
    else if (await B.WebXRSessionManager.IsSessionSupportedAsync('immersive-vr')) mode = 'immersive-vr';
  } catch (e) { /* no WebXR */ }
  if (!mode) {
    hint.textContent = window.isSecureContext ? 'WebXR not available in this browser. Open this page in Edge on the HoloLens 2 to view it in AR.'
                                              : 'WebXR needs HTTPS (or localhost). Open the HTTPS address on the HoloLens.';
    return;
  }
  hint.textContent = `Headset detected (${mode}). Use the goggles button bottom-right to enter.`;
  xrHelper = await scene.createDefaultXRExperienceAsync({
    uiOptions: { sessionMode: mode, referenceSpaceType: 'local-floor' },
    optionalFeatures: true,
    disableTeleportation: true,
  });
  const drag = new B.SixDofDragBehavior();
  drag.rotateWithMotionController = true;
  const pinch = new B.MultiPointerScaleBehavior();
  xrHelper.baseExperience.onStateChangedObservable.add(st => {
    if (st === B.WebXRState.IN_XR) {
      inXR = true;
      scene.clearColor = new B.Color4(0, 0, 0, 0);   // transparent = see-through on HoloLens
      setModelScale(0.5);
      xrPanel.handle.setEnabled(true);
      root.addBehavior(drag); root.addBehavior(pinch);
      setTimeout(placeForUser, 400);
    } else if (st === B.WebXRState.NOT_IN_XR && inXR) {
      inXR = false;
      scene.clearColor = new B.Color4(0.07, 0.08, 0.1, 1);
      root.removeBehavior(drag); root.removeBehavior(pinch);
      xrPanel.handle.setEnabled(false);
      root.position.setAll(0); root.rotationQuaternion = null; root.rotation.setAll(0); setModelScale(1);
    }
  });
}

main().catch(err => {
  console.error(err);
  document.getElementById('loading').textContent = 'Failed to load: ' + err.message;
});
