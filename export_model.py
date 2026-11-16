"""Export the measurement -> shape model for the HoloLens / WebXR viewer.

Writes docs/data/knee_shape_model.ksv (git-ignored: the shape model is NOT published with the app).

Only AGGREGATE statistics are exported (means, SDs, ranges, covariances), no per-case rows.
The viewer predicts a shape with ordinary least squares on any subset A of measurements:

    V(x_A) = mean_V + C_VA C_AA^-1 (x_A - mu_A)

which is exactly the OLS fit (with intercept) of the notebooks' `fit_predict`, on all 305 cases,
using all 310 SSM modes (C_VX = ssmV @ Cov(W, X), so no mode truncation).

Run from the RVH folder:  python HoloLensViewer/export_model.py
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'docs' / 'data'
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data (same as Measurements2SSM_Flexion.ipynb)
S = sio.loadmat(ROOT / 'SSM_femtib.mat')
MEAN, ssmV = S['MEAN'].ravel(), S['ssmV']
NV = MEAN.size // 3
W = sio.loadmat(ROOT / 'Weights_307.mat')['Modes']
X26 = sio.loadmat(ROOT / 'ClinicalMeasurements_307.mat')['results']
F = sio.loadmat(ROOT / 'CheckCorrespondenceFaces/Ffemurtib_lowres.mat')['newFaceIdx'].astype(int) - 1
keep = np.setdiff1d(np.arange(307), [159, 190])            # drop the 2 duplicated cases
W, X26 = W[keep], X26[keep]
n = len(W)

FEM, TIB, FIB = np.arange(0, 2256), np.arange(2256, 4798), np.arange(4798, NV)
FT = np.arange(0, 4798)
M = MEAN.reshape(3, NV).T                                    # x = proximal, -y = lateral, +z = posterior
P = (MEAN[:, None] + ssmV @ W.T).reshape(3, NV, -1).transpose(2, 1, 0)   # n x NV x 3
x = M[:, 0]

# ---------------------------------------------------------------- landmarks (picked once on the SSM mean shape)
def band(idx, lo, hi):
    xi = x[idx]; x0, L = xi.min(), np.ptp(xi); return idx[(xi >= x0 + lo * L) & (xi <= x0 + hi * L)]

def sphere_fit(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]; s = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
    c = s[:3]; return c, np.sqrt(s[3] + c @ c)

FEM_SHAFT, TIB_SHAFT = band(FEM, .3, .7), band(TIB, .3, .7)
distal = FEM[x[FEM] < x[FEM].min() + 45]
MED_EPI = distal[np.argsort(-M[distal, 1])[:15]]
LAT_EPI = distal[np.argsort(M[distal, 1])[:15]]
TPLAT = TIB[x[TIB] > x[TIB].max() - 12]                       # proximal 12 mm of tibia (plateau)
TDIST = TIB[x[TIB] < x[TIB].min() + 12]                       # distal 12 mm of tibia (plafond)
head = FEM[x[FEM] > x[FEM].max() - 25]                        # femoral head: iterative sphere fit
c, r = sphere_fit(M[head])
for _ in range(5):
    head = FEM[np.abs(np.linalg.norm(M[FEM] - c, axis=1) - r) < 0.12 * r]; c, r = sphere_fit(M[head])
HEAD = head
print(f'femoral head: {len(HEAD)} vertices, radius {r:.1f} mm')

# ---------------------------------------------------------------- mesh measurements (mirrored 1:1 in app.js)
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

def long_axis(Pn, idx):
    q = Pn[:, idx] - Pn[:, idx].mean(1, keepdims=True)
    v = np.array([np.linalg.svd(a, full_matrices=False)[2][0] for a in q]); return v * np.sign(v[:, :1])

def flexion(Pn):
    ml = unit(Pn[:, LAT_EPI].mean(1) - Pn[:, MED_EPI].mean(1))
    proj = lambda u: unit(u - (u * ml).sum(1, keepdims=True) * ml)
    f, t = proj(long_axis(Pn, FEM_SHAFT)), proj(long_axis(Pn, TIB_SHAFT))
    ang = np.degrees(np.arccos(np.clip((f * t).sum(1), -1, 1)))
    return -np.sign((t * np.cross(ml, f)).sum(1)) * ang

def centres(Pn):
    hip = np.array([sphere_fit(p[HEAD])[0] for p in Pn])
    return hip, Pn[:, np.r_[MED_EPI, LAT_EPI]].mean(1), Pn[:, TPLAT].mean(1), Pn[:, TDIST].mean(1)

def femur_length(Pn):        # hip centre -> most distal femur point, along the femoral mechanical axis
    hip, kneeF, _, _ = centres(Pn); fa = unit(hip - kneeF)
    return -np.einsum('ijk,ik->ij', Pn[:, FEM] - hip[:, None], fa).min(1)

def tibia_length(Pn):        # tibial plateau centre -> tibial plafond centre
    _, _, kneeT, ankle = centres(Pn); return np.linalg.norm(kneeT - ankle, axis=1)

def hka_raw(Pn):
    hip, kneeF, kneeT, ankle = centres(Pn)
    ml = unit(Pn[:, LAT_EPI].mean(1) - Pn[:, MED_EPI].mean(1))
    fa, ta = hip - kneeF, kneeT - ankle
    ap = unit(np.cross(ml, unit(fa)))                          # coronal-plane normal
    pr = lambda u: unit(u - (u * ap).sum(1, keepdims=True) * ap)
    a, b = pr(fa), pr(ta)
    ang = np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))
    return np.sign((np.cross(a, b) * ap).sum(1)) * ang         # signed deviation from straight

names26 = ['HKA', 'FMA', 'TMAlat', 'FAA', 'TAAlat', 'FMvsA', 'JLCA', 'FNSA', 'FAVA', 'aTEA-PCL', 'sTEA-PCL', 'asTEA',
           'TTA-PCL', 'FTA-PCL', 'FTA-TTA', 'FTA-TAPA', 'TEVA', 'TT-TG', 'PT', 'TI', 'SA', 'MTS', 'LTS', 'FL', 'TL', 'FO']
flex = flexion(P)
X = np.c_[X26, flex]; names = names26 + ['Flexion']
units = ['°'] * 23 + ['mm', 'mm', 'mm', '°']
units[names.index('TT-TG')] = 'mm'
long_names = {   # Table 1 of the student article
    'HKA': 'Hip-knee-ankle angle', 'FMA': 'Femoral mechanical angle', 'TMAlat': 'Tibial mechanical angle',
    'FAA': 'Femoral anatomical angle', 'TAAlat': 'Tibial anatomical angle', 'FMvsA': 'Femoral mechanical vs anatomical angle',
    'JLCA': 'Joint line convergence angle', 'FNSA': 'Femoral neck-shaft angle', 'FAVA': 'Femoral anteversion angle',
    'aTEA-PCL': 'Anatomical TEA vs posterior condylar line', 'sTEA-PCL': 'Surgical TEA vs posterior condylar line',
    'asTEA': 'Anatomical vs surgical TEA', 'TTA-PCL': 'Tibial transverse axis vs posterior condylar line',
    'FTA-PCL': 'Femoral transverse axis vs posterior condylar line', 'FTA-TTA': 'Femoral vs tibial transverse axis',
    'FTA-TAPA': 'Femoral transverse axis vs trochlear AP axis', 'TEVA': 'Tibial external version angle',
    'TT-TG': 'Tibial tuberosity - trochlear groove distance', 'PT': 'Patellar tilt', 'TI': 'Lateral trochlear inclination',
    'SA': 'Sulcus angle', 'MTS': 'Medial tibial slope', 'LTS': 'Lateral tibial slope', 'FL': 'Femoral length',
    'TL': 'Tibial length', 'FO': 'Femoral offset', 'Flexion': 'Sagittal knee flexion (computed on the mesh)'}

# validation of the mesh-based overlays against the measurement table
hr = hka_raw(P); sgn = 1.0 if np.corrcoef(hr, X[:, 0])[0, 1] > 0 else -1.0
mesh_vals = {'HKA': 180 + sgn * hr, 'FL': femur_length(P), 'TL': tibia_length(P),'Flexion': flex}
valid = {}
for k, v in mesh_vals.items():
    t = X[:, names.index(k)]
    valid[k] = {'r': float(np.corrcoef(v, t)[0, 1]), 'mae': float(np.abs(v - t).mean()), 'bias': float((v - t).mean())}
    print(f'{k:8s} mesh vs table: r = {valid[k]["r"]:.3f}, MAE {valid[k]["mae"]:.2f}, bias {valid[k]["bias"]:+.2f}')

# ---------------------------------------------------------------- aggregate statistics
V = P[:, FT].reshape(n, -1)                                    # n x (4798*3), interleaved xyz
mu_X, mu_V = X.mean(0), V.mean(0)
Xc = X - mu_X
C_XX = Xc.T @ Xc / (n - 1)
C_WX = (W - W.mean(0)).T @ Xc / (n - 1)                        # 310 x 27
rows = np.concatenate([FT, FT + NV, FT + 2 * NV])
C_VX_blocks = ssmV[rows] @ C_WX                                # (3*4798) x 27 in [x; y; z] blocks
C_VX = C_VX_blocks.reshape(3, len(FT), -1).transpose(1, 0, 2).reshape(-1, X.shape[1])   # interleaved

# self-check: covariance formula == OLS (notebook fit_predict) for a subset
sub = [names.index(k) for k in ['FL', 'Flexion', 'HKA', 'TL']]
Bols = np.linalg.lstsq(X[:, sub] - mu_X[sub], W - W.mean(0), rcond=None)[0]
pred_ols = ((X[:5, sub] - mu_X[sub]) @ Bols + W.mean(0)) @ ssmV[rows].T + MEAN[rows]
pred_ols = pred_ols.reshape(5, 3, -1).transpose(0, 2, 1).reshape(5, -1)
pred_cov = mu_V + (X[:5, sub] - mu_X[sub]) @ np.linalg.solve(C_XX[np.ix_(sub, sub)], C_VX[:, sub].T)
print('max |cov-formula - OLS| =', np.abs(pred_cov - pred_ols).max(), 'mm')

# ---------------------------------------------------------------- write
meanV32 = mu_V.astype('<f4'); cvx32 = C_VX.astype('<f4'); faces16 = F.astype('<u2')
blob = meanV32.tobytes() + cvx32.tobytes() + faces16.tobytes()
lm = lambda a: [int(i) for i in a]
meta = {
    'description': 'Femur-tibia SSM (305 cases), OLS from clinical measurements to shape. Aggregate statistics only.',
    'nVertices': len(FT), 'nFaces': len(F), 'nMeas': len(names),
    'bin': {'meanV': [0, meanV32.size], 'C_VX': [meanV32.nbytes, cvx32.size],
            'faces': [meanV32.nbytes + cvx32.nbytes, faces16.size]},
    'femur': [0, 2256], 'tibia': [2256, 4798],
    'axes': 'x = proximal, -y = lateral, +z = posterior (mm)',
    'measurements': [{'key': k, 'name': long_names[k], 'unit': u, 'mean': float(mu_X[j]), 'sd': float(np.sqrt(C_XX[j, j])),
                      'min': float(X[:, j].min()), 'max': float(X[:, j].max())}
                     for j, (k, u) in enumerate(zip(names, units))],
    'C_XX': C_XX.tolist(),
    'defaultActive': ['FL', 'Flexion', 'HKA', 'TL'],
    'landmarks': {'HEAD': lm(HEAD), 'MED_EPI': lm(MED_EPI), 'LAT_EPI': lm(LAT_EPI), 'TPLAT': lm(TPLAT),
                  'TDIST': lm(TDIST), 'FEM_SHAFT': lm(FEM_SHAFT), 'TIB_SHAFT': lm(TIB_SHAFT)},
    'hkaSign': sgn,
    'validation': valid,
}
# single bundle: b'KSV1' | uint32 json length | json (space-padded to a multiple of 4) | binary blob
js = json.dumps(meta, separators=(',', ':')).encode()
js += b' ' * (-len(js) % 4)
bundle = b'KSV1' + np.uint32(len(js)).astype('<u4').tobytes() + js + blob
(OUT / 'knee_shape_model.ksv').write_bytes(bundle)
print(f'wrote {OUT / "knee_shape_model.ksv"} ({len(bundle) / 1e6:.2f} MB)  -- keep this file private, it is git-ignored')
