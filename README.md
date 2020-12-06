# Argoverse-Monocular_depth-dataset-creation
Follow the instructions in the notebook to generate depth maps from lidar ground truth points corresponding to the ring camera frames and use it for monocular depth estimation model training. 

NOTE: This notebook will be merged with Argoverse through [this](https://github.com/argoai/argoverse-api/pull/146) PR.

![Alt text](depth_map.png?raw=true "Sample")
*Depth map dilated for better visualization



# Monocular-Depth-Estimation
Check out [Monocular-Depth-Estimation-Argoverse](https://github.com/TilakD/Monocular-Depth-Estimation-Argoverse) for model training results from the above created dataset.
Below are few sample depth estimation from the Argoverse trained Resnext101 BTS model. Click to watch the video on youtube.

Depth estimation on Front Center Camera
[![Screenshot](https://github.com/TilakD/Monocular-Depth-Estimation-Argoverse/blob/master/images/vlcsnap-2020-07-20-15h53m17s829.png)](https://youtu.be/Fu7XHyHw1Gc)
Depth estimation on Other Ring Cameras
[![Screenshot](https://github.com/TilakD/Monocular-Depth-Estimation-Argoverse/blob/master/images/vlcsnap-2020-07-21-14h43m47s958.png)](https://youtu.be/mjnpUREeBcM)


