import os
import re
import cv2
import copy
import math
import torch
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from utils.general import (non_max_suppression, scale_boxes,print_args, check_file)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from models.with_mobilenet import PoseEstimationWithMobileNet
from utils.pose.keypoints import extract_keypoints, group_keypoints
from utils.pose.load_state import load_state
from utils.pose.pose import Pose, track_poses
from utils.pose.augmentations.geometric import GeometricPrediction
from utils.pose.augmentations.sparseopticalflow import SparseOpticalFlow
from utils.pose.augmentations.feature_transformation import FeatureTransformation
from utils.gui import GUI
from utils.evaluate import *
from utils.interventions.body import *
from utils.interventions.interventions import *

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, device='cpu',
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    '''
    Develops inference for heatmaps for pose detection.

    PARAMETERS:
        net: pose estimation model (Torch.nn)
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

    if str(device).isdigit():
        tensor_img = tensor_img.cuda() # Set tensor to GPU
    elif device == 'mps':
        tensor_img = tensor_img.to(torch.device("mps")) # Set tensor to Metal Performance Shaders

    # Run network
    stages_output = net(tensor_img)

    # Extract heatmaps
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    # Extract pafs
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def get_tourn_summary(body, labels):
    #print(f"RES:Video: {new_dir}")
    tccc_summary = body.getSummary() #This provides list of 4 tuples [(),(),(),]
    summary_metrics = get_summary_metrics(tccc_summary, labels)

    # filter results to throw away assignments that have fewer than the
    # largest num of frames

    max_count = 0
    for _,_,count,_ in tccc_summary:
        if count > max_count:
            max_count = count
    tccc_summary_filtered = [hit for hit in tccc_summary if hit[2] >= .5 * max_count] #attempting to push out FP by dropping pairs that have total frames < X, s.t. X is the highest pair frame count
    summary_metrics_filtered = get_summary_metrics(tccc_summary_filtered, labels)
    return summary_metrics, summary_metrics_filtered, tccc_summary_filtered

@torch.no_grad()
def run(opt, min_jt_count, max_allowed_dist_pct, min__jt_area_pct, labels=None, proms = [1,5,10,15]):

    '''
    Main method to run pipeline
    '''

    # Check source type
    add_random_chest_seal = False
    source = str(opt.source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load YOLO object detection model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    print('Classes: ' + str(names)+'\n')

    if opt.hpe or opt.gpe:
        # Load Mobilenetv1 pose model
        net = PoseEstimationWithMobileNet()
        #checkpoint = torch.load('weights/pose.pth', map_location='cpu')
        #checkpoint = torch.load('weights/fine_tune_no_val.pth', map_location='cpu')
        checkpoint = torch.load(opt.hpe_weights, map_location='cpu')
        load_state(net, checkpoint)
        net = net.eval() # Set model to evaluation mode
        if str(opt.device).isdigit():
            net = net.cuda() # Set model to GPU
        elif opt.device == 'mps':
            net = net.to(torch.device('mps')) # Set model to Metal Performance Shaders

    if opt.bpe:
        # Load Yolov5 body part detection model
        model_bpe = DetectMultiBackend('weights/Yolov5_rotcvid_limbs_weights_LR.pt', device=device, dnn=False, data=None, fp16=False)
        stride_bpe, names_bpe, pt_bpe = model_bpe.stride, model_bpe.names, model_bpe.pt
        print('Classes: ' + str(names_bpe)+'\n')

     # GUI Initialization
    gui = GUI()

    # optical flow
    of = SparseOpticalFlow()

    # feature transforms
    ft = FeatureTransformation()

    # assignment (Will - evaluation)
    body = Body()
    intervention = Intervention(body, names)

    #assignment frame counter and hitter
    frame_count = 0
    frame_hits = 0

    if opt.hpe or opt.gpe:
        #Track scores for joints
        kpt_names = ['nose', 'neck',
                     'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                     'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                     'r_eye', 'l_eye',
                     'r_ear', 'l_ear']

        joint_hits = np.zeros((len(names), len(kpt_names)))
        patientPose = None

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=(640, 640), stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=(640, 640), stride=stride, auto=pt)
        nr_sources = 1

    # Run YOLO tracking
    model.warmup()

    if opt.gpe:
        # Geometric prediction
        gp = GeometricPrediction(18)

    if opt.hpe or opt.gpe:
        previous_poses = []

    if opt.bpe:
        # Run body part detection
        model_bpe.warmup()

    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    print('\n')
    print('Starting Inference...')

    # Progress bar
    progressBar = tqdm(total = dataset.frames) if is_file else None

    #Make dir for storing pose/det overlap imgs
    new_dir = opt.output_dir + source.rsplit('/', 1)[-1].split('.')[0]
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)

    # Start vid writer
    if opt.write_vid:
        vid_writer = cv2.VideoWriter(new_dir+'/recording.mp4',
                            cv2.VideoWriter_fourcc(*'MP4V'),
                            30, # TODO: get fps from video source
                            (1920 ,1080)) # get resolution from video source
        
    # Carry Pose counter
    frames_without__pose_update = 0

    final_img = None
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        #frame count
        frame_count = frame_idx

        # Get original image
        orig_img = im0s
        if webcam:
            orig_img = im0s[0]
        output_img = orig_img

        gui.clearTCCC() # Clear TCCC

        torso_bbox = None
        if opt.hpe or opt.gpe:
            # Get pose estimation
            heatmaps, pafs, scale, pad = infer_fast(net, orig_img, 256, 9, 4, device=opt.device)
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

            # Filter poses based on score
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
            track_poses(previous_poses, current_poses, smooth=False)
            previous_poses = current_poses

            # compare current poses with previous poses,
            # if id is the same, update the previous pose with the current pose
            # else add the current pose to the previous poses

            biggest_id = -1
            biggest_value = 0
            patientPoseUpdated = False # Variable for whether HPE model has updated
            frames_without__pose_update += 1

            for current_pose in current_poses:
                area = current_pose.bbox[2] * current_pose.bbox[3]
                if area > biggest_value:
                    biggest_id = current_pose.id
                    biggest_value = area
                loopBroken = False
                for previous_pose in previous_poses:
                    if current_pose.id == previous_pose.id:
                        previous_pose.keypoints = current_pose.keypoints
                        previous_pose.bbox = current_pose.bbox
                        loopBroken = True
                        break
                if not loopBroken:
                    previous_poses.append(current_pose)

            for pose in current_poses:

                if current_pose.id == biggest_id:

                    if True:
                    #if pose.bbox[2]*pose.bbox[3]/(1920*1080) > float(opt.min_pose_percent): # if pose is greater than 15% of input frame size

                        # geometric prediction
                        if opt.gpe:
                            any_missing = np.where(pose.keypoints == [-1,-1])[0].any()
                            if not any_missing:     # if we have a full set, update our procrustes estimate
                                gp.updateProcrustes(pose.keypoints)
                                patientPose = pose

                            else:   # if we don't have a full set, use the procrustes estimate

                                # geometric prediction with procrustes
                                p_pose = copy.copy(pose)
                                #new_pts_procrustes, _ = gp.predictProcrustes(pose.keypoints)
                                new_pts_procrustes = gp.correctProcrustes(pose.keypoints)
                                p_pose.keypoints = new_pts_procrustes
                                p_pose.draw(output_img, [0, 255, 0])
                                patientPose = p_pose # TODO : Add docs and in a better location
                        else:

                            patientPose = pose
                        pose.draw(output_img, [255, 255, 0])
                        patientPoseUpdated = True
                        frames_without__pose_update = 0

                        # BBOX Torso Test
                        torso_kpts = []
                        if (patientPose.keypoints[1] != [-1,-1]).all():
                           torso_kpts.append(patientPose.keypoints[1])
                        if (patientPose.keypoints[2] != [-1,-1]).all():
                           torso_kpts.append(patientPose.keypoints[2])
                        if (patientPose.keypoints[5] != [-1,-1]).all():
                           torso_kpts.append(patientPose.keypoints[5])
                        if (patientPose.keypoints[8] != [-1,-1]).all():
                           torso_kpts.append(patientPose.keypoints[8])
                        if (patientPose.keypoints[11] != [-1,-1]).all():
                           torso_kpts.append(patientPose.keypoints[11])
                        if torso_kpts:
                            torso_kpts = np.array(torso_kpts)
                            torso_bbox = torch.tensor(cv2.boundingRect(torso_kpts), device=device)

        # Optical flow
        if opt.optical_flow and (opt.hpe or opt.gpe):
            if patientPose is not None:
                output_img, shifts = of.calc(orig_img, frame_idx, patientPose.keypoints) # TODO
                of_pose = None
                if not patientPoseUpdated:
                    of_pose = patientPose
                    for i in range(len(of_pose.keypoints)):
                        of_pose.keypoints[i]+=shifts[i]
                    of_pose.draw(output_img, (255,150,255)) if (frames_without__pose_update <= opt.carry_pose) else None
                if of_pose is not None:
                    patientPose = of_pose

        # Feature transformations
        if opt.feature_transform and (opt.hpe or opt.gpe):
            if patientPose is not None:
                ft_matrix = ft.calc(orig_img, frame_idx, patientPose, patientPoseUpdated, new_dir)
                ft_pose = patientPose
                ft_pose.keypoints = cv2.perspectiveTransform(patientPose.keypoints.astype(ft_matrix.dtype).reshape(-1, 1, 2), ft_matrix).astype(np.int32).reshape(-1, 2)
                ft_pose.keypoints[np.where(patientPose == -1)] = -1 
                ft_pose.draw(output_img, (0,255,0)) if (frames_without__pose_update <= opt.carry_pose) else None

        # Yolo Preprocessing + Inference + NMS
        im = torch.from_numpy(im).to(device).float()
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im) # Inference
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45) # NMS

        annotator_bpe = None
        pred_bpe = None
        if opt.bpe:
            pred_bpe = model_bpe(im) # Inference
            pred_bpe = non_max_suppression(pred_bpe, conf_thres=0.25, iou_thres=0.45) # NMS

            for i, det in enumerate(pred_bpe):
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                curr_frames[i] = im0

                s += '%gx%g ' % im.shape[2:]
                annotator_bpe = Annotator(output_img, line_width=3, example=str(names_bpe))

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names_bpe[int(c)]}{'s' * (n > 1)}, "  # add to string

                    bbox = det[0][0:4]
                    conf = det[0][4]
                    cls = det[0][5]
                    c = int(cls)  # integer class

                    # Draw bbox and annotate
                    label = f'{names_bpe[c]} {conf:.2f}'
                    color = colors(c, True)
                    annotator_bpe.box_label(bbox, label, color=color)

        # Process Yolo detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            curr_frames[i] = im0

            s += '%gx%g ' % im.shape[2:]  # print string

            # Annotator
            annotator = Annotator(output_img, line_width=3, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                bbox = det[0][0:4]
                conf = det[0][4]
                cls = det[0][5]
                c = int(cls)  # integer class
                if opt.hpe or opt.gpe:

                    if (frames_without__pose_update > opt.carry_pose):
                        patientPose = None

                    #check for which pure overlap (add forgiveness paramter for f number of frames back to check)
                    if patientPose is not None:
                        pose_bbox = pose.get_bbox(patientPose.keypoints)
                        x0, y0, w, h = pose_bbox
                        #print(patientPose.keypoints)
                        mask = patientPose.keypoints > -1
                        #print(mask)
                        #print(pose_bbox)
                        pose_area = w * h
                        #print("Area:", area)

                        num_jts_detected = np.count_nonzero(mask)/2
                        #pose_cross_distance = math.sqrt((xf - x0)**2 + (yf - y0)**2)
                        #mid_pose_x = int((x0+xf)/2)
                        #mid_pose_y = int((y0+yf)/2)
                        #x_jt_len = abs(x0-xf)
                        #y_jt_len = abs(y0-yf)
                        y_frame_len, x_frame_len, _ = orig_img.shape
                        frame_area = y_frame_len * x_frame_len
                        #print(orig_img.shape)
                        pose_area_pct = pose_area/frame_area
                        #print(pose_area_pct)
                        #if (x_jt_len*y_jt_len)/(y_frame_len*x_frame_len) > min__jt_area_pct: #and num_jts_detected >= min_jt_count:
                        #if pose_area_pct > opt.min_pose_percent:
                        if True:
                            min_idx = map_tr_to_jts(bbox, patientPose.keypoints, new_dir, joint_hits, frame_idx, c, orig_img, max_allowed_dist_pct, num_jts_detected, min_jt_count, pose_area_pct, float(opt.min_pose_percent), save_img=opt.write_imgs)

                            # Only for hemostatic and pressure dressings TODO this was moved under if statement
                            if min_idx is not None:
                                frame_hits += 1
                                #if torso_bbox is not None and (names[c] == 'hd' or names[c] == 'pd'):
                                    #torsoIOU = bbox_iou(bbox, torso_bbox)
                                    #print(torsoIOU.item())
                                    #if torsoIOU.item() > opt.min_bpe_iou:
                                    #    intervention.add_intervention(Limbs.torso, frame_idx, names[c])
                                    #else:
                                    #intervention.add_interventions_kpts(min_idx,frame_idx, names[c])
                                if names[c] == 'cs':
                                    #if torsoIOU.item() > opt.min_bpe_iou:
                                    intervention.add_intervention(Limbs.torso, frame_idx, names[c])
                                else:
                                    #if names[c] == 'tour': # Tourniquet
                                    intervention.add_interventions_kpts(min_idx, frame_idx, names[c])
                                    if add_random_chest_seal and frame_idx%10 == 0:
                                        joint_hits[2,1] += 1
                                        intervention.add_intervention(Limbs.torso, frame_idx, names[c])


                                
                if opt.bpe:
                    limbDist = (None, None) # (Dist, Limb)
                    for _, limb_det in enumerate(pred_bpe):
                        if limb_det is not None and len(limb_det):
                            limb_center = (int((limb_det[0][0]+limb_det[0][2])/2),int((limb_det[0][1]+limb_det[0][3])/2))
                            treatment_center = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
                            dist = math.sqrt(math.pow(limb_center[0]-treatment_center[0],2)+math.pow(limb_center[1]-treatment_center[1],2))
                            limb_name = names_bpe[int(limb_det[0][5])]
                            if limbDist[0] == None:
                                limbDist = (dist, limb_name)
                            else:
                                if dist < limbDist[0]:
                                    limbDist = (dist, limb_name)
                    if limbDist[0] is not None:
                        # Classes: {0: 'head', 1: 'left arm', 2: 'left leg', 3: 'right arm', 4: 'right leg', 5: 'torso'}
                        match limbDist[1]:
                            case 'head':
                                intervention.add_intervention(Limbs.head, frame_idx, names[c])
                            case 'left arm':
                                intervention.add_intervention(Limbs.left_arm, frame_idx, names[c])
                            case 'left leg':
                                intervention.add_intervention(Limbs.left_leg, frame_idx, names[c])
                            case 'right arm':
                                intervention.add_intervention(Limbs.right_arm, frame_idx, names[c])
                            case 'right leg':
                                intervention.add_intervention(Limbs.right_leg, frame_idx, names[c])
                            case 'torso':
                                intervention.add_intervention(Limbs.torso, frame_idx, names[c])
                        cv2.line(output_img, limb_center, treatment_center, (255,115,140), 5)

                # Draw bbox and annotate
                label = f'{names[c]} {conf:.2f}'
                color = colors(c, True)
                annotator.box_label(bbox, label, color=color)

            # Mark on gui
            tccc_summary_int = body.getSummary()
            tccc_summary_int.sort(key=lambda x: x[2], reverse=True)
            pair_count = 0
            for max_bp, max_treatment, max_frames, _ in tccc_summary_int:
#                if pair_count == 0:
#                    print(f'top: {max_bp}, {max_treatment}, {max_frames}')
                color = [0, 0, 255] if pair_count == 0 else [0, 255, 255]
                gui.markTCCCLimb(
                    max_bp, max_treatment,
                    15 if max_frames > 350 else 15 * max_frames / 350,
                    color)
                pair_count = pair_count + 1

            # bpe annotations
            if opt.bpe:
                output_img = annotator_bpe.result()

            # Stream results
            output_img = annotator.result() # Draw annotations

            if torso_bbox is not None: # Having issues with the annotator so... this is the way
                torso_bbox = torso_bbox.tolist()
                output_img = cv2.rectangle(output_img, (torso_bbox[0], torso_bbox[1]), (torso_bbox[0]+torso_bbox[2], torso_bbox[1]+torso_bbox[3]),(0,165,255),2)

            # Update progress bar
            progressBar.update(1) if is_file else None

            # Update previous frame
            prev_frames[i] = curr_frames[i]

            # Assign GUI output frame
            gui.outputFrame = output_img

            # Write frame to video
            if opt.write_vid:
                if opt.verbose_gui:
                    vid_writer.write(gui.drawGUI())
                else:
                    vid_writer.write(output_img)

            final_img = orig_img

    #print(joint_hits)
    joint_hits = nullify_impossible_pts(joint_hits)
    #print(joint_hits_dropped_impossible)
    #print(body.body_parts[Limbs.torso.value])

    # Pre zscore summary
    tccc_summary_no_z = copy.deepcopy(body.getSummary())
    
    tccc_summary = body.getSummary()
    #print(f'pre-zscore tccc_summary: {tccc_summary}')

    zscore_drops = intervention.process(new_dir, opt.zscore_window_size, opt.zscore_threshold) #TODO Verify
    #tour_zscore_drops = zscore_drops
    # zscore_drops = intervention.processTQ(new_dir) #TODO get stuff here
    #zscore_drops = [sum(zscore_right_arm), sum(zscore_left_arm), sum(zscore_right_leg), sum(zscore_left_leg)]    
    #print(joint_hits)
    joitnt_hits_dropped_tour_jts_not_preserved = drop_zscore_tour(zscore_drops, joint_hits) # TODO: Broken -> evaluate.py 267 (IndexError: index 2 is out of bounds for axis 0 with size 2)
    #print(joitnt_hits_dropped_tour_jts_not_preserved)
    tccc_summary = body.getSummary()
    #print(f'tccc_summary: {tccc_summary}')
    raw_metrics_limb = None
    prom_metrics = None

    #print(joint_hits[2])
    #print(tccc_summary)
    #print(tccc_summary_no_z)


    if opt.labels != '0':

        prom_metrics = output_metrics(source, kpt_names, names, joint_hits, labels, proms, opt, new_dir)
        zscore_sorted_new_dir = new_dir + '/zscore_sorted'
        os.mkdir(zscore_sorted_new_dir)
        _ = output_metrics(source, kpt_names, names, joitnt_hits_dropped_tour_jts_not_preserved, labels, proms, opt, zscore_sorted_new_dir)
        #raw_metrics_jt = get_joint_hits_metrics(joint_hits, labels)
        #print(raw_metrics_jt)
        raw_metrics_limb = get_limb_hits_metrics(joint_hits, labels)
        raw_metrics_limb_filtered = get_limb_hits_metrics(joitnt_hits_dropped_tour_jts_not_preserved, labels)
        write_raw_metrics(raw_metrics_limb, new_dir)

        print(f"RES:Video: {new_dir}")
        tccc_summary = body.getSummary() #This provides list of 4 tuples [(),(),(),]

        summary_metrics = get_summary_metrics(tccc_summary, labels)

        summary_metrics_no_z = get_summary_metrics(tccc_summary_no_z, labels)

        # filter results to throw away assignments that have fewer than half the
        # largest num of frames

        max_count = 0
        for _,_,count,_ in tccc_summary:
            if count > max_count:
                max_count = count
        tccc_summary_filtered = [hit for hit in tccc_summary if hit[2] >= opt.marcs_threshold * max_count] #attempting to push out FP by dropping pairs that have total frames < X, s.t. X is the highest pair frame count
        summary_metrics_filtered = get_summary_metrics(tccc_summary_filtered, labels)

        max_count = 0
        for _,_,count,_ in tccc_summary_no_z:
            if count > max_count:
                max_count = count
        tccc_summary_filtered_no_z = [hit for hit in tccc_summary_no_z if hit[2] >= opt.marcs_threshold * max_count] #attempting to push out FP by dropping pairs that have total frames < X, s.t. X is the highest pair frame count
        summary_metrics_filtered_no_z = get_summary_metrics(tccc_summary_filtered_no_z, labels)

        '''
        #FOR DOT GETTING BIGGER
        gui.clearTCCC() # Clear TCCC
        gui.outputFrame = final_img
        pair_count = 0
        tccc_summary_filtered.sort(key=lambda x: x[2], reverse=True)
        for max_bp, max_treatment, max_frames, _ in tccc_summary_filtered:
            color = [0, 0, 255] if pair_count == 0 else [0, 255, 255]
            gui.markTCCCLimb(
                max_bp, max_treatment,
                15 if max_frames > 350 else 15 * max_frames / 350,
                color)
            pair_count = pair_count + 1
        if opt.verbose_gui and opt.write_vid:
            for qq in range(200):
                vid_writer.write(gui.drawGUI())
        '''
        
        # Release video writer
        if opt.write_vid:
            vid_writer.release()
    else:
        output_results(source, kpt_names, names, joint_hits, proms, opt, new_dir)

    progressBar.close() if is_file else None

    #print(summary_metrics_filtered)
    #print(summary_metrics_filtered_no_z)

    return raw_metrics_limb, raw_metrics_limb_filtered, frame_hits/frame_count, summary_metrics, summary_metrics_filtered, prom_metrics, summary_metrics_no_z, summary_metrics_filtered_no_z

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_vids', type=str, default='0', help='directory of videos to run on')
    parser.add_argument('--source', type=str, default='0', help='file, 0 for webcam')
    parser.add_argument('--device', default='0', help='cuda, i.e. 0 or 0,1,2,3 or metal (mps) or cpu')
    parser.add_argument('--weights', default='./weights/heavy_4tr.pt', help='Give a path to a weights file')
    parser.add_argument('--hpe_weights', default='./weights/fine_tune_no_val.pth', help='Give a path to an hpe weights file')
    parser.add_argument('--output_dir', default='./output/', help='Give a path to output your results')
    parser.add_argument('--labels', default='0', help='Give a path to you excel file labels')
    parser.add_argument('--check_label', default='0', help='1 if you want to check loading labels')

    # Treatment Assignment Parameters
    parser.add_argument('--min_jts', default=1, help='Set a minimum number of joints to be detected for a pose to be considered valid')
    parser.add_argument('--max_allowed_dist_pct', default=1, help='Set a maximum distance as a percentage of the frame size from the treatment to the pose for the frame to be considered valid')
    #parser.add_argument('--min_jt_area', default=0, help='Set a minimum area as a percentage of frame area that a pose bbox must occupy for a frame to be considered valid')
    parser.add_argument('--min_pose_percent', default=0.25, help='Set a minimum precentage of screen that pose must be')
    parser.add_argument('--min_bpe_iou', default = 0.1, type=float, help='minimum iou (jaccard index) of bpe bounding box and treatment bounding box')
    parser.add_argument('--zscore_window_size', default=60, type=int, help='Zscore moving window size')
    parser.add_argument('--zscore_threshold', default=1.5, type=float, help='Zscore threshold to drop frame')
    parser.add_argument('--marcs_threshold', default=.5, type=float, help='Majorityt Vote Threshold')
    
    # Boolean Flags
    parser.add_argument('--verbose_gui', action=argparse.BooleanOptionalAction, default=False, help='Saves recording with TCCC output drawn in video')
    parser.add_argument('--write_vid', action=argparse.BooleanOptionalAction, default=False, help='Write out a video file of the run')
    parser.add_argument('--write_imgs', action=argparse.BooleanOptionalAction, default=False, help='Write out a image folder for hits of the run')

    # Pose Estimation Model Selection
    assign = parser.add_mutually_exclusive_group(required=True)
    assign.add_argument('--bpe', action=argparse.BooleanOptionalAction, default=False, help='body part estimator')
    assign.add_argument('--hpe', action=argparse.BooleanOptionalAction, default=False, help='human pose estimator')
    assign.add_argument('--gpe', action=argparse.BooleanOptionalAction, default=False, help='HPE + Geometric')

    # Pose Estimation Augementations
    parser.add_argument('--optical_flow', action=argparse.BooleanOptionalAction, default=False, help='Draws motion from sparse optical flow using Lucas-Kanade method')
    parser.add_argument('--feature_transform', action=argparse.BooleanOptionalAction, default=False, help='Uses SIFT feature matching to transform poses between frames')
    parser.add_argument('--carry_pose', default=0, type=int, help='Carry old pose through how many frames')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

natsort = lambda s: [int(t) if t.isdigit()
                     else t.lower() for t in re.split('vid(\d+)', s)]
def main(opt):
    vid_path = opt.path_to_vids
    if opt.output_dir != './output/':
        opt.output_dir = './' + opt.output_dir + '/'
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)
    if opt.check_label == '1' and opt.labels != 0:
        label_dict = get_data(opt.labels)
        vid_name = opt.source.rsplit('/', 1)[-1].split('.')[0]
        print(label_dict[vid_name])

    elif vid_path != '0' and opt.labels != '0':
        vids = os.listdir(vid_path)
        sorted_vids = sorted(vids, key=natsort)
        sorted_vids_new = filtered_list = [video for video in sorted_vids if not video.startswith('._')]
        run_videos(sorted_vids_new, vid_path, opt)

    elif opt.labels != '0':
        label_dict = get_data(opt.labels)
        vid_name = opt.source.rsplit('/', 1)[-1].split('.')[0]
        raw_metrics, _, frame_hit_rate, summary_metrics, summary_metrics_filtered, prom_metrics, _, _ = run(opt, int(opt.min_jts),float(opt.max_allowed_dist_pct),float(opt.min_pose_percent), labels=label_dict[vid_name])
        #print(raw_metrics)
        #print(frame_hit_rate)
        #print(summary_metrics)
        #print(summary_metrics_filtered)
        #print(prom_metrics)
    else:
        run(opt, int(opt.min_jts),float(opt.max_allowed_dist_pct),float(opt.min_jt_area))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)





'''
print(f'videos: {sorted_vids}')
label_dict = get_data(opt.labels)
output_dict = {}
for v in sorted_vids:
    print('-------------------------------------------STARTING NEW VIDEO---------------------------------------------------------')
    path_v = vid_path + v
    opt.source = path_v
    vid_name = opt.source.rsplit('/', 1)[-1].split('.')[0]
    raw_metrics, frame_hit_rate, summary_metrics, summary_metrics_filtered, prom_metrics = run(opt, int(opt.min_jts),float(opt.max_allowed_dist_pct),float(opt.min_jt_area), labels=label_dict[vid_name])
    if raw_metrics is not None:
        (true_pos_pair_count, false_pos_pair, pair_precision), (true_pos_tr, false_pos_tr, tr_precision), (true_pos_limb, false_pos_limb, limb_precision), missed_labels, metric_per_label, drop_impossible_tour_points, bad_jt_counts_per_tr = raw_metrics
        per_label_metric_dict = {}
        #for output in metric_per_label:
        #    this_label = output[0][0] + " " + output[0][1]
        #    per_label_metric_dict[this_label] = {
        #        "Correct Hits" : output[1],
        #        "Percent Hits Treatment" : output[2]
        #    }
        output_dict[v] = {
            "Overall_TP" : true_pos_pair_count,
            "Overall_FP" : false_pos_pair,
            "Overall_Prec" : pair_precision,
            "Overall_False_Discovery": 1-pair_precision,
            "Treatment_TP" : true_pos_tr,
            "Treatment_FP" : false_pos_tr,
            "Treatment_Prec" : tr_precision,
            "Limb_TP" : true_pos_limb,
            "Limb_FP" : false_pos_limb,
            "Limb_Prec" : limb_precision,
            "Frame Hit Rate": frame_hit_rate,
            "TCCC TP": summary_metrics[0],
            "TCCC FP": summary_metrics[1],
            "TCCC FN": summary_metrics[2],
            "TCCC Sorted TP": summary_metrics_filtered[0],
            "TCCC Sorted FP": summary_metrics_filtered[1],
            "TCCC Sorted FN": summary_metrics_filtered[2],
            " ": " ",
                
        }
        for output in metric_per_label:
            this_label = output[0][0] + " " + output[0][1]
            output_dict[v].update({
                "Correct Hits for " + this_label : output[1],
                "Percent Hits for " + this_label : output[2]
            })
    print(output_dict)
    df = pd.DataFrame(output_dict)
    excel_name = opt.output_dir + 'out.xlsx'
    df.to_excel(excel_name, index=True)
    overall_metrics(excel_name)


    #print(f"RES:Video: {new_dir}")
    tccc_summary = body.getSummary() #This provides list of 4 tuples [(),(),(),]
    summary_metrics = get_summary_metrics(tccc_summary, labels)

    # filter results to throw away assignments that have fewer than the
    # largest num of frames

    max_count = 0
    for _,_,count,_ in tccc_summary:
        if count > max_count:
            max_count = count
    tccc_summary_filtered = [hit for hit in tccc_summary if hit[2] >= .5 * max_count] #attempting to push out FP by dropping pairs that have total frames < X, s.t. X is the highest pair frame count
    summary_metrics_filtered = get_summary_metrics(tccc_summary_filtered, labels)

    #print(f"RES:GT: {list(zip(labels['Limbs'], labels['Treatments']))}")
    #print(f'RES:tccc_summary: {tccc_summary}')
    #print(f'RES:TP: {summary_metrics[0]} FP: {summary_metrics[1]} '
    #      f'FN: {summary_metrics[2]}')

    #print(f'RES:tccc_summary_filtered: {tccc_summary_filtered}')
    #print(f'RES:TP: {summary_metrics_filtered[0]} FP: {summary_metrics_filtered[1]} '
    #      f'FN: {summary_metrics_filtered[2]}')

'''