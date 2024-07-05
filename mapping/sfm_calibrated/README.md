Recover poses and camera intrinsic for ordered images taken from a monocular video.

How it works
 - It reconstructs the scene (and estimates intrinsics) incrementally (frames are added one by one)
 - Feature extraction and matching: extract features from adjacent frames, match using optical flow
 - Pose recovery: recovers pose based on epipolar geometry and PnP with RANSAC outlier rejection (using OpenCV)
 - Bundle adjustment: optimize poses, points, and camera intrinsics as error accumulates
 - Loop closure: close loops to correct drift over time

Speed / Robustness
 - On my laptop (16 CPU cores), recent versions (see Progress Breakdown) extracts 150-250 frames from a 30-60s 30fps video in 2-4min
 - Works well for most scenes with clear structures and good lighting, failure rate is higher on scenes with rich details, strong moving shadow/highlights, strong noise / blur, repetitive patterns, or large rotation with little translation

Accuracy
 - Works for training NeRF / Gaussian Splatting, better with camera poses further optimized with gradient descent
 - Successful loop closure can significantly increase accuracy

Failure cases
 - PnP failure; signs: "in-the-wild" cameras, small clusters of cameras
 - Camera intrinsics converge to a poor local minima

C++ Dependencies
 - Ceres Solver, for bundle adjustment
 - FBOW, for loop closure
   - Set `vocab_path` in the script to the path to vocabulary
 - At this time, you need to manually build bindings for these libraries in `ba_solver` and `lc_solver` directories

## Progress Breakdown

Each script is independent. Roughly in chronological order listing my progress building the tool.

`triangulation`: give 2 frames, estimate relative pose and feature point positions
 - `triangulation_01`: recover pose from essential matrix + triangulation, looks good

`ba_3`: give 3 frames, estimate relative poses and feature point positions
 - `ba_3_01`: estimate pairwise relative pose, minimize relative translation constraint; fail, translation becomes negative
 - `ba_3_02`: estimate relative pose for two frames, find pose for third frame using PnP; working

`ba_n`: give $n\ge3$ frames, estimate relative poses and feature point positions
 - `ba_n_01`: fail, optimization becomes unstable
 - `ba_n_02`: $O(N^2)$ pairwise feature matching, choose two frames for initial reconstruction, incrementally add more frames; largely working but fails on some scenes
 - `ba_n_03`: try autodiff BA Jacobian using Jax, doesn't make faster
 - `ba_n_04`: incrementally add frames in order after initial matches, run BA after adding each PnP point cloud; incorporate optical flow in feature matching; works on much longer sequences, fails on scenes with large rotation and coplanar geometry
 - `ba_n_05`: multi-frame optical flow feature tracking, skipped bundle adjustment; able to track long sequences of frames under good conditions, fails at large drift
 - `ba_n_06`: bundle adjustment with Ceres solver; outlier rejection
 - `ba_n_07`: better outlier rejection, skip adjacent frames with little motion

`feature`: explore better management of features
 - `feature_01`: using OpenCV RANSAC outlier rejection for matched features; performance isn't promising

`lc`: explore loop closure
 - `lc_01`: get basic loop closure working, doesn't work well for noisy scenes
 - `lc_02`: directly read from video (instead of image sequence), automatically select frames
 - `lc_03`: basic relocalization after tracking lost
 - `lc_04`: don't change camera intrinsics after pre-calibration
