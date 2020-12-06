#!/usr/bin/env python
# coding: utf-8
# Tilak 

import fnmatch
import glob
import os
from typing import Optional, Tuple

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST


class Lidar2Depth:
    """
    Convert 360 degree LiDAR point cloud to depth map corresponding to each ring camera
    for monocular depth estimation.

    To use:
    >>> input_log_dir = "path/to/3d20ae25-5b29-320d-8bae-f03e9dc177b9/"
    >>> output_save_path = "path/to/depth_dataset/"
    >>> Lidar2Depth(input_log_dir, output_save_path)  
    """

    def __init__(self, input_log_dir: str, output_save_path: str) -> None:

        self.input_log_dir = input_log_dir
        self.output_save_path = output_save_path
        self.log_id = os.path.basename(input_log_dir)
        print("Log ID ", self.log_id)

        # Load Argo data
        dataset = os.path.dirname(self.input_log_dir)
        self.argoverse_loader = ArgoverseTrackingLoader(dataset)
        self.argoverse_data = self.argoverse_loader.get(self.log_id)

        # Count the number of LiDAR ply files in the log dir
        self.lidar_frame_counter = len(
            glob.glob1(os.path.join(self.input_log_dir, "lidar"), "*.ply")
        )

        # Setup depth dataset dir
        self.depth_data_dir_setup()

        # Extract depth data and ring camera frames
        self.depth_extraction()

    def depth_data_dir_setup(self) -> None:
        """
        Depth dataset structure
        +-- train/val/test
        |   +-- depth
        |   |   +-- 00c561b9-2057-358d-82c6-5b06d76cebcf
        |   |   |   +-- ring_front_center
        |   |   |   |   +-- 1.png
        |   |   |   |   +-- 2.png
        |   |   |   |   +--   .
        |   |   |   |   +--   .
        |   |   |   |   +-- n.png
        |   |   |   +-- ring_front_left
        |   |   |   +--        .
        |   |   |   +--        .
        |   |   |   +-- ring_side_right
        |   |   +-- 0ef28d5c-ae34-370b-99e7-6709e1c4b929
        |   |   |   +-- ring_front_center
        |   |   |   +--        .
        |   |   |   +--        .
        |   |   |   +-- ring_side_right
        |   |   +--            .
        |   |   +--            .
        |   +-- rgb
        |   |   +-- 00c561b9-2057-358d-82c6-5b06d76cebcf
        |   |   |   +-- ring_front_center
        |   |   |   |   +-- 1.png
        |   |   |   |   +--   .
        |   |   |   |   +-- n.png
        |   |   |   +-- ring_front_left
        |   |   |   +--        .
        |   |   |   +-- ring_side_right
        |   |   +-- 0ef28d5c-ae34-370b-99e7-6709e1c4b929
        |   |   |   +-- ring_front_center
        |   |   |   +-- ring_front_left
        |   |   |   +--        .
        |   |   |   +-- ring_side_right
        |   |   +--            .
        |   |   +--            .
        """
        if fnmatch.fnmatchcase(self.input_log_dir, "*" + "train" + "*"):
            self.save_name = os.path.join(self.output_save_path, "train")
            self.logid_type = "train"

        elif fnmatch.fnmatchcase(self.input_log_dir, "*" + "val" + "*"):
            self.save_name = os.path.join(self.output_save_path, "val")
            self.logid_type = "val"

        elif fnmatch.fnmatchcase(self.input_log_dir, "*" + "test" + "*"):
            self.save_name = os.path.join(self.output_save_path, "test")
            self.logid_type = "test"

        for camera_name in RING_CAMERA_LIST:
            paths = [
                os.path.join(self.save_name, "depth", self.log_id, camera_name),
                os.path.join(self.save_name, "rgb", self.log_id, camera_name),
            ]
            for sub_path in paths:
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

    def extract_lidar_image_pair(
        self, camera_ID: int, lidar_frame_idx: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        For the provided camera_ID and LiDAR ply file,
        extract rgb image and corresponding LiDAR points in the fov.
        """
        img = self.argoverse_data.get_image_sync(lidar_frame_idx, camera=camera_ID)
        self.calib = self.argoverse_data.get_calibration(camera_ID)
        pc = self.argoverse_data.get_lidar(lidar_frame_idx)
        uv = self.calib.project_ego_to_image(pc).T
        lidar_frame_idx_ = np.where(
            np.logical_and.reduce(
                (
                    uv[0, :] >= 0.0,
                    uv[0, :] < np.shape(img)[1] - 1.0,
                    uv[1, :] >= 0.0,
                    uv[1, :] < np.shape(img)[0] - 1.0,
                    uv[2, :] > 0,
                )
            )
        )
        lidar_image_projection_points = uv[:, lidar_frame_idx_]
        if lidar_image_projection_points is None:
            print("No point image projection")
            return np.array(img), None
        else:
            return np.array(img), lidar_image_projection_points

    def save_image_pair(
        self,
        camera_ID: int,
        img: np.ndarray,
        lidar_frame_idx: str,
        lidar_image_projection_points: np.ndarray,
    ) -> None:
        """
        Save the depth images and camera frame to the created dataset dir.
        """
        x_values = np.round(lidar_image_projection_points[0], 0).astype(int)
        y_values = np.round(lidar_image_projection_points[1], 0).astype(int)
        lidar_depth_val = lidar_image_projection_points[2]

        # Create a blank image to place lidar points as pixels with depth information
        sparse_depth_img = np.zeros(
            [img.shape[0], img.shape[1]]
        )  # keeping it float to maintain precision
        sparse_depth_img[y_values, x_values] = lidar_depth_val

        # Multiple to maintain precision, while model training, remember to divide by 256
        # NOTE: 0 denotes a null value, rather than actually zero depth in the saved depth map
        depth_rescaled = sparse_depth_img * 256.0
        depth_scaled = depth_rescaled.astype(np.uint16)
        depth_scaled = Image.fromarray(depth_scaled)
        raw_depth_path = os.path.join(
            self.save_name,
            "depth",
            self.log_id,
            str(camera_ID),
            str(lidar_frame_idx) + ".png",
        )
        depth_scaled.save(raw_depth_path)  # Save Depth image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_img_path = os.path.join(
            self.save_name,
            "rgb",
            self.log_id,
            str(camera_ID),
            str(lidar_frame_idx) + ".png",
        )
        cv2.imwrite(
            raw_img_path, img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )  # Save RGB image

    def frame2depth_mapping(self, camera_ID: int, lidar_frame_idx: str) -> None:
        """
        For your training dataloader, you will likely find it helpful to read image paths
        from a .txt file. We explicitly write to a .txt file all rgb image paths that have
        a corresponding sparse ground truth depth file along with focal length.
        """
        mapping_file = open(
            os.path.join(
                self.output_save_path, "argo_" + self.logid_type + "_files_with_gt.txt"
            ),
            "a",
        )
        file_path = os.path.join(
            str(self.log_id), camera_ID, str(lidar_frame_idx) + ".png"
        )
        gt_string = file_path + " " + file_path + " " + str(np.round(self.calib.fv, 4))
        mapping_file.write(gt_string + "\n")

    def depth_extraction(self) -> None:
        """
        For every lidar file, extract ring camera frames and store it in the save dir
        along with depth map
        """
        for lidar_frame_idx in tqdm(range(self.lidar_frame_counter)):
            for camera_ID in RING_CAMERA_LIST:
                # Extract camera frames and associated lidar points
                img, lidar_image_projection_points = self.extract_lidar_image_pair(
                    camera_ID, lidar_frame_idx
                )
                # Save image and depth map if LiDAR projection points exist
                if lidar_image_projection_points is not None:
                    # Save the above extracted images
                    self.save_image_pair(
                        camera_ID, img, lidar_frame_idx, lidar_image_projection_points
                    )
                    # Write path of rgb image, depth image along with focal length
                    # in a txt file for data loader
                    self.frame2depth_mapping(camera_ID, lidar_frame_idx)
                else:
                    continue


# Modify paths here,
local_path_to_argoverse_splits = (
    "./Argoverse/full_data/extracted/argoverse-tracking"
)
output_save_path = (
    "./Argoverse/monocular_depth_dataset/"
)



folders = [
    f"{local_path_to_argoverse_splits}/train1/",
    f"{local_path_to_argoverse_splits}/train2/",
    f"{local_path_to_argoverse_splits}/train3/",
    f"{local_path_to_argoverse_splits}/train4/",
    f"{local_path_to_argoverse_splits}/val/",
    f"{local_path_to_argoverse_splits}/test/",
]

log_list = []
for folder in folders:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    log_list.extend(subfolders)
    
log_list.sort()
for input_log_dir in log_list:
    Lidar2Depth(input_log_dir, output_save_path)
    break  # Remove break to run on all logids
