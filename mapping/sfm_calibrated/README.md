SfM with images taken from the same calibrated camera, on ordered video frames.

`triangulation`: give 2 frames, estimate relative pose and feature point positions
 - `triangulation_01`: recover pose from essential matrix + triangulation, looks good

`ba_3`: give 3 frames, estimate relative poses and feature point positions
 - `ba_3_01`: estimate pairwise relative pose, minimize relative translation constraint; fail, translation becomes negative
 - `ba_3_02`: estimate relative pose for two frames, find pose for third frame using PnP; working

`ba_n`: give $n\ge3$ frames, estimate relative poses and feature point positions
 - `ba_n_01`: fail, optimization becomes unstable
 - `ba_n_02`: $O(N^2)$ pairwise feature matching, choose two frames for initial reconstruction, incrementally add more frames; largely working but fails on some scenes
