
import numpy as np
import json
import datetime
from utils.evaluate import map_tr_to_jts
from utils.metrics import bbox_iou
from utils.pose.pose import Pose
from utils.interventions.interventions import Intervention, Limbs
from utils.interventions.body import Body

class Mapper:

    def __init__(self, treatment_model) -> None:

        #Track scores for joints
        self.kpt_names = ['nose', 'neck',
                     'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                     'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                     'r_eye', 'l_eye',
                     'r_ear', 'l_ear']

        # Track scores for limbs
        self.limb_names = ['head', 'r_arm', 'l_arm', 'r_leg', 'l_leg', 'chest']

        # Treatment Names
        self.names = treatment_model.names

        # Body Map
        self.body = Body()
        self.intervention = Intervention(self.body, self.names)

        self.joint_hits = np.zeros((len(self.names), len(self.kpt_names)))

    def map(self, img, frame_idx, pred, patientPose, torso_bbox,
                min_jt_area_pct=0, min_jt_count=1, max_allowed_dist_pct=1, min_bpe_iou=0.75):

        # Process Yolo detections
        for i, det in enumerate(pred):

            min_idxs = []
            if det is not None and len(det):

                bbox = det[0][0:4]
                cls = det[0][5]
                c = int(cls)  # integer class

                #check for which pure overlap (add forgiveness paramter for f number of frames back to check)
                if patientPose is not None:
                    pose_bbox = Pose.get_bbox(patientPose.keypoints)
                    mask = patientPose.keypoints != -1
                    num_jts_detected = np.count_nonzero(mask)/2
                    x0, y0, xf, yf = pose_bbox
                    mid_pose_x = int((x0+xf)/2)
                    mid_pose_y = int((y0+yf)/2)
                    x_jt_len = abs(x0-xf)
                    y_jt_len = abs(y0-yf)
                    y_frame_len, x_frame_len, _ = img.shape
                    if (x_jt_len*y_jt_len)/(y_frame_len*x_frame_len) > min_jt_area_pct and num_jts_detected >= min_jt_count:
                        min_idx = map_tr_to_jts(bbox, patientPose.keypoints, 'tmp', self.joint_hits, frame_idx, c, img, max_allowed_dist_pct,)
                        min_idxs.append((min_idx, self.names[c]))

                        # Only for hemostatic and pressure dressings TODO this was moved under if statement
                        if torso_bbox is not None and (self.names[c] == 'hd' or self.names[c] == 'pd'):
                            torsoIOU = bbox_iou(bbox, torso_bbox)
                            if torsoIOU.item() > min_bpe_iou:
                                self.intervention.add_intervention(Limbs.torso, frame_idx, self.names[c])
                            else:
                                self.intervention.add_interventions_kpts(min_idx,frame_idx, self.names[c])

                if len(min_idxs) > 0:
                    for min_idx, treatment in min_idxs:
                        #print(f'treatment is : {treatment}')
                        if treatment == 'tour': # Tourniquet
                            self.intervention.add_interventions_kpts(min_idx, frame_idx, treatment)
                        elif treatment == 'cs': # Chest Seal
                            self.intervention.add_intervention(Limbs.torso, frame_idx, treatment)
                else:
                    self.intervention.add_interventions_kpts(-1, frame_idx, None)

    def summarized_filter(self):
        tccc_summary = self.body.getSummary()

        # filter results to throw away assignments that have fewer than half the
        # largest num of frames

        max_count = 0
        for _,_,count,_ in tccc_summary:
            if count > max_count:
                max_count = count
        tccc_summary_filtered = [hit for hit in tccc_summary if hit[2] >= .5 * max_count] #attempting to push out FP by dropping pairs that have total frames < X, s.t. X is the highest pair frame count
        print(f'RES:tccc_summary_filtered: {tccc_summary_filtered}')

        # JSON read template
        json_template = open('utils/template.json')
        json_dict = json.load(json_template)
        json_template.close()

        # update values
        for treatment in tccc_summary_filtered:
            json_dict.get("patient").get(treatment[1]).update({str(datetime.datetime.now()):treatment[0]})

        # write new json
        with open("tmp/incomplete.json", "w") as json_output:
            json.dump(json_dict, json_output, indent=4, default=str)