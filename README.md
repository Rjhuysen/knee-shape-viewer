# Knee Shape Viewer

A small WebXR viewer for the HoloLens 2 (it also works in a desktop browser). You set clinical measurements with sliders and
see the femur–tibia shape a statistical shape model (SSM) predicts. You can grab and move the bones, compare them with the
population-mean shape, and show measurement overlays (HKA axes, flexion axes, femur/tibia lengths).

**What is (not) published.** `docs/model/knee_shape_model.ksv` holds only the *OLS relationship*: the mean shape,
faces, the measurement–vertex covariance `C_VX` and the 27×27 measurement covariance `C_XX`, plus per-measurement
mean/SD/range and landmark indices. The statistical shape model itself (`ssmV`, eigenvectors/eigenvalues, mode weights)
and all per-case data are never exported. `ssmV` is used only locally in `export_model.py` to compute `C_VX`.

## Using it

- **HoloLens 2:** open https://rjhuysen.github.io/knee-shape-viewer/ in Edge and press the goggles button (bottom right)
  to enter mixed reality.
- **Desktop:** open the same link, or run `python -m http.server 8000` in `docs/` and open http://localhost:8000
  (WebXR needs HTTPS, so mixed reality only works through the hosted link).
- **Updating the model:** after the SSM or the measurements change, run (from the folder with `SSM_femtib.mat` etc.)
  ```
  python HoloLensViewer/export_model.py
  ```
  then commit and push `docs/model/knee_shape_model.ksv`.

## In mixed reality

- **Panel** (left): *Sliders* holds the active inputs (drag a slider or use −/+), *Inputs* chooses which measurements are
  model inputs, and *Display* has the overlays, the mean-shape ghost, deviation colours, model size, and *Bring model and panel in front of me*.
- **Bones:** grab them with a pinch or air-tap-and-hold to move and rotate them. Use two hands to scale.
- **Panel position:** grab the blue bar above the panel.

## The model

`export_model.py` reproduces the OLS model of `Measurements2SSM_Flexion.ipynb`, fitted on the 305 non-duplicate cases with
26 clinical measurements + Flexion (computed on the mesh). It exports only aggregate statistics:

`V(x_A) = mean_V + C_VA · C_AA⁻¹ · (x_A − μ_A)`

This equals OLS (with intercept) from any chosen subset A of measurements to all 310 SSM modes, so the browser can refit
instantly when you tick or untick inputs. The same algebra gives the *expected* value of every unticked measurement
(conditional mean) and a Mahalanobis plausibility check of the inputs.

Mesh-based overlays use landmarks picked once on the mean shape:

| overlay | definition | agreement with the clinical table (305 cases) |
|---|---|---|
| HKA | hip centre (sphere fit to femoral head) → epicondyle midpoint; tibial plateau centre → plafond centre; coronal plane | r = 0.983, bias −0.2° |
| Flexion | femoral / tibial mid-shaft PCA axes in the sagittal plane (as in the notebook) | identical by construction |
| FL | hip centre → most distal femur point along the mechanical axis | r = 0.999, bias +9.8 mm |
| TL | tibial plateau centre → plafond centre | r = 0.998, bias +6.3 mm |

The FL/TL offsets are constant (different landmarks than the clinical tool) and are shown as they are, without correction.

## Debugging

- `index.html?panel=sliders|select|display` shows the HoloLens panel in the desktop view.
- To see the HoloLens console, open the Windows Device Portal and use Edge DevTools remote debugging (`edge://inspect`).

Libraries: [Babylon.js](https://www.babylonjs.com/) 9.33.1 (Apache-2.0, vendored in `docs/lib`).
