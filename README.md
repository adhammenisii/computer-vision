# Computer Vision

Coursework project applying classical computer vision techniques with OpenCV and NumPy —
image enhancement, segmentation, alignment, panorama stitching, object recognition,
stereo depth estimation, and HDR imaging.

## Contents

```
notebooks/   Jupyter notebooks (main deliverables)
scripts/     Standalone scripts for Tasks 7 & 8
images/      Input images and generated results
docs/        Project brief
```

## Notebooks

### `notebooks/CV_Project.ipynb`
The full task set (Tasks 1–8), written for Google Colab:

| Task | Topic |
|------|-------|
| 1 | Selective image enhancement (the rose) — HSV color segmentation + unsharp masking |
| 2 | Image alignment & bone subtraction (X-ray) |
| 3 | Auto enhancement |
| 4 | Document enhancement & noise removal |
| 5 | Manual panorama stitching |
| 6 | Invariant object recognition |
| 7 | Depth approximation from two images |
| 8 | High dynamic range (HDR) imaging |

Also covers the building blocks: RGB→grayscale conversion, Gaussian blur, unsharp-mask
sharpening, red-mask color segmentation, and image fusion.

### `notebooks/CV_Tasks_7_8_FINAL.ipynb`
A cleaned-up, self-contained write-up of Tasks 7 and 8.

**Task 7 — Depth approximation from a stereo pair**
1. Load left & right images
2. Grayscale conversion + CLAHE contrast equalization
3. Disparity map via `cv2.StereoBM`
4. Post-processing and normalization
5. Color-mapped depth visualization

**Task 8 — HDR imaging**
1. Load a bracketed exposure series
2. Estimate the camera response function (Debevec method)
3. Merge into a radiance map
4. Tone-map to a displayable image, compared against a manual implementation

## Scripts

- `scripts/task7_depth_map.py` — `compute_depth_map(left_path, right_path)`, StereoBM disparity pipeline
- `scripts/task8_hdr.py` — `create_hdr(image_paths, exposure_times)`, Debevec CRF estimation and tone mapping

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

Task 8 additionally uses `opencv-contrib-python` for some tone-mapping operators.

## Results

Generated outputs are in `images/outputs/`:

| File | Description |
|------|-------------|
| `task7_depth_gray.jpg` | Raw disparity/depth map (grayscale) |
| `task7_depth_color.jpg` | Depth map with a color map applied |
| `task7_final.jpg` | Task 7 summary figure |
| `task8_radiance_map.jpg` | Merged HDR radiance map |
| `task8_hdr_cv.jpg` | Tone-mapped result (OpenCV) |
| `task8_hdr_manual.jpg` | Tone-mapped result (manual implementation) |
| `task8_comparison.jpg` | Side-by-side tone-mapping comparison |
| `task8_final.jpg` | Task 8 summary figure |

## Note on input images

`images/inputs/` holds the source images still available locally: the stereo pair
(`left_image.jpg`, `right_image.jpg`) and the Task 1–4 sources (`Rose_Image`, `X_Ray`,
`foggy_landscape`).

Some inputs were uploaded directly into Colab sessions (`files.upload()`) and were never
saved to disk — the HDR exposure brackets (`bright.jpg`, `normal.jpg`, `dark.jpg`) and
several Task 4–6 sources. To re-run those cells you'll need to supply your own images.
All results remain embedded in the notebook outputs, so the notebooks render fully as-is
on GitHub.
