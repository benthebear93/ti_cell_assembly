# RealSense UR5e Calibration

Files copied from calibration_gui-master refined calibration run.

## Files

- Calibration.xml: refined hand-eye/extrinsic result. Use MovingTransform frame=Camera as the transform from UR TCP/tool frame to RealSense color camera frame.
- camera_intrinsics.yaml: RealSense color camera intrinsics used for pose estimation. Captured at 1920x1080.
- refinement_notes.txt: original sample indices kept/excluded during outlier filtering.

## Refined Calibration Quality

- Average residual: 3.044 mm / 0.474 deg
- Maximum residual: 5.348 mm / 0.973 deg

## Test Marker

- Dictionary: DICT_ARUCO_ORIGINAL
- Marker ID: 364
- PDF nominal black-square side length: 25.4 mm

Measure the printed marker black-square side length and pass the measured value to --marker-length-mm for best accuracy.
