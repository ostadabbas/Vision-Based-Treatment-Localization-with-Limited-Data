import math
import torch
import cv2
import copy
import numpy as np

from models.with_mobilenet import PoseEstimationWithMobileNet
from utils.pose.keypoints import extract_keypoints, group_keypoints
from utils.pose.load_state import load_state
from utils.pose.pose import Pose, track_poses
from utils.pose.augmentations.geometric import GeometricPrediction
from utils.evaluate import *
from utils.interventions.interventions import *
from utils.interventions.body import *

class HPE():

    def __init__(self, device=0, min_pose_precent=0, use_geometric_pose=True) -> None:
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load('weights/pose.pth', map_location='cpu')
        load_state(self.net, checkpoint)
        self.net = self.net.eval() # Set model to evaluation mode
        if str(device).isdigit():
            self.net = self.net.cuda() # Set model to GPU
        elif device == 'mps':
            self.net = self.net.to(torch.device('mps')) # Set model to Metal Performance Shaders
        self.device = device

        self.previous_poses = []
        self.patientPose=None
        
        self.min_pose_percent = min_pose_precent
        
        self.use_geometric_pose = use_geometric_pose
        if self.use_geometric_pose:
            self.gp = GeometricPrediction(18)

    def infer_fast(self, img, net_input_height_size=256, stride=9, upsample_ratio=4,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        '''
        Develops inference for heatmaps for pose detection.

        PARAMETERS:
            img: frame to be evalutated (Mat)
            net_input_height_size: frame shape (n x m)
            stride: video stride
            upsample_ratio: upsample ratio

        OUTPUT:
            heatmaps: Heatmaps for joint locations
            pafs: part affinity fields
            scale: net_input_height_size / height
            pad: padding for rescaled image
        '''
        #Resize images to make uniform
        height, width, _ = img.shape
        scale = net_input_height_size / height
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = np.array(scaled_img, dtype=np.float32)
        scaled_img = (scaled_img - img_mean) * img_scale
        #Force 256 x n, s.t. n >= 256
        min_dims = [256, max(scaled_img.shape[1], 256)]

        # Pad image to make it divisible by 32
        h, w, _ = scaled_img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(scaled_img, pad[0], pad[2], pad[1], pad[3],
                                        cv2.BORDER_CONSTANT, value=pad_value)

        # Prepare image for network
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

        if str(self.device).isdigit():
            tensor_img = tensor_img.cuda() # Set tensor to GPU
        elif self.device == 'mps':
            tensor_img = tensor_img.to(torch.device("mps")) # Set tensor to Metal Performance Shaders

        # Run network
        stages_output = self.net(tensor_img)

        # Extract heatmaps
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        # Extract pafs
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def calc(self, input, output):

        patientPoseUpdated = False
        torso_bbox = None

        # Get pose estimation
        heatmaps, pafs, scale, pad = self.infer_fast(input)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = Pose.num_kpts
        
        # Get keypoints
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        # Get poses
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        
        # Scale keypoints
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * 2 - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * 2 - pad[0]) / scale
        
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        
        track_poses(self.previous_poses, current_poses, smooth=False)

        # if there are no previous poses, set the current poses as the previous poses
        if not len(self.previous_poses):
            self.previous_poses = current_poses

        biggest_id = -1
        biggest_value = 0
        for current_pose in current_poses:
            area = current_pose.bbox[2] * current_pose.bbox[3]
            if area > biggest_value:
                biggest_id = current_pose.id
                biggest_value = area
            loopBroken = False
            for previous_pose in self.previous_poses:
                if current_pose.id == previous_pose.id:
                    previous_pose.keypoints = current_pose.keypoints
                    previous_pose.bbox = current_pose.bbox
                    loopBroken = True
                    break
            if not loopBroken:
                self.previous_poses.append(current_pose)
                
        for pose in current_poses:
            if current_pose.id == biggest_id:
                any_missing = np.where(pose.keypoints == [-1,-1])[0].any()
                pose.draw(output)
                if self.use_geometric_pose:
                    if not any_missing:
                        # if we have a full set, update our procrustes estimate
                        self.gp.updateProcrustes(pose.keypoints)
                        self.patientPose = pose
                    else:
                        # geometric prediction
                        p_pose = copy.copy(pose)
                        new_pts_procrustes = self.gp.correctProcrustes(pose.keypoints)
                        p_pose.keypoints = new_pts_procrustes
                        p_pose.draw(output)
                        self.patientPose = p_pose
                    self.patientPose.draw(output, [0, 255, 0])
                    pose.draw(output, [255, 255, 0])
                
                patientPoseUpdated = True

                # BBOX Torso Test
                torso_kpts = []
                if (self.patientPose.keypoints[1] != [-1,-1]).all():
                   torso_kpts.append(self.patientPose.keypoints[1])
                if (self.patientPose.keypoints[2] != [-1,-1]).all():
                   torso_kpts.append(self.patientPose.keypoints[2])
                if (self.patientPose.keypoints[5] != [-1,-1]).all():
                   torso_kpts.append(self.patientPose.keypoints[5])
                if (self.patientPose.keypoints[8] != [-1,-1]).all():
                   torso_kpts.append(self.patientPose.keypoints[8])
                if (self.patientPose.keypoints[11] != [-1,-1]).all():
                   torso_kpts.append(self.patientPose.keypoints[11])
                torso_kpts = np.array(torso_kpts)
                torso_bbox = torch.tensor(cv2.boundingRect(torso_kpts), device=self.device)

        return output, self.patientPose, torso_bbox, patientPoseUpdated