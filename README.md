# Knee Shape Viewer

A small WebXR viewer for the HoloLens 2 (it also works in a desktop browser). You set clinical measurements with sliders and
see the femur–tibia shape a statistical shape model (SSM) predicts. You can grab and move the bones, compare them with the
population-mean shape, and show measurement overlays (HKA axes, flexion axes, femur/tibia lengths).

**The shape model is not part of this repository.** The app only contains code. The model file
`knee_shape_model.ksv` is generated locally and loaded by the user, and it is listed in `.gitignore`.

## Using it

1. Generate the model file (from the folder that holds `SSM_femtib.mat` etc.):
   ```
   python HoloLensViewer/export_model.py
   ```
   This writes `docs/data/knee_shape_model.ksv` (about 1.7 MB).
2. Open the app:
   - **Hosted (GitHub Pages):** open the Pages URL, click *Choose model file…* and select `knee_shape_model.ksv`.
     The file is then cached in that browser (IndexedDB), so you only do this once per device.
     *About the model → Remove stored model* deletes the cached copy.
   - **Local:** `python -m http.server 8000` inside `docs/`, then open http://localhost:8000. The model loads
     automatically from `docs/data/`. WebXR needs HTTPS, so this route is for the desktop only.
3. **On the HoloLens 2:** copy `knee_shape_model.ksv` to the HoloLens first, for example over USB in File Explorer
   (*HoloLens → Internal storage → Documents*) or through OneDrive. Then open the Pages URL in Edge, load the file,
   and press the goggles button (bottom right) to enter mixed reality.

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
