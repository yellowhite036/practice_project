# Fixed ROI HOG + SVM

This project uses a fixed rectangular ROI across the road lanes, extracts HOG features from that ROI, and trains or applies a linear SVM classifier.

## Files

- `123.mp4`: input traffic video.
- `roi_hog_svm.py`: ROI preview, ROI sample collection, HOG + SVM training, and video classification.
- `roi_preview.jpg`: generated preview image with the default ROI rectangle.

## Default ROI

The default ROI is:

```text
x=250, y=520, w=1350, h=310
```

It is selected for the 1920x1080 video and spans the main two-lane road area. You can override it with `--roi x,y,w,h`.

## 1. Preview The Fixed ROI

```powershell
py roi_hog_svm.py preview --video 123.mp4 --frame 120 --output roi_preview.jpg
```

## 2. Collect ROI Samples

This exports cropped ROI images every 15 frames. Manually sort the exported images into `samples/positive` and `samples/negative`.

```powershell
py roi_hog_svm.py collect --video 123.mp4 --output-dir samples/unlabeled --step 15
```

Suggested labels:

- `samples/positive`: ROI contains the target traffic condition or vehicle presence.
- `samples/negative`: ROI does not contain the target condition.

## 3. Train SVM

```powershell
py roi_hog_svm.py train --positive-dir samples/positive --negative-dir samples/negative --model models/roi_hog_svm.yml
```

The script resizes each ROI crop to `128x64`, extracts HOG features, and trains a linear `C_SVC` SVM.

## 4. Classify Video Frames

```powershell
py roi_hog_svm.py detect --video 123.mp4 --model models/roi_hog_svm.yml --output output_roi_hog_svm.mp4
```

The output video draws the fixed ROI and displays the SVM classification result for each frame.

py roi_hog_svm.py preview --frame 120 --output roi_preview.jpg
py roi_hog_svm.py collect --output-dir samples/unlabeled --step 15
py roi_hog_svm.py train --positive-dir samples/positive --negative-dir samples/negative --model models/roi_hog_svm.yml
py roi_hog_svm.py detect --model models/roi_hog_svm.yml --output output_roi_hog_svm.mp4