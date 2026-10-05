These four camera-model files are from the Oxford RobotCar Dataset SDK:
https://github.com/ori-mrg/robotcar-dataset-sdk/tree/master/models

The SDK repository states a CC BY-NC-SA 4.0 license:
https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/LICENSE

`stereo_wide_left.txt` and `stereo_wide_right.txt` provide the rectified
wide-camera intrinsics and camera poses. Their matching binary distortion LUTs
map each rectified output pixel back into the raw Bayer images. Do not substitute
LUTs from another camera or image resolution. These files are used here only
with the Oxford RobotCar sample, not as generic calibration for other cameras.
