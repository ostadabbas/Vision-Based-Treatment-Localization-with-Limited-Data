import cv2
import numpy as np
import math

from utils.pose.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from utils.pose.one_euro_filter import OneEuroFilter

class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):

        '''
        Pose

        PARAMETERS:
            keypoints: Keypoints (numpy array)
            confidence: Confidence (float)

        OUTPUT:
            None
        '''

        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):

        '''
        Get bounding box

        PARAMETERS:
            keypoints: Keypoints (numpy array)

        OUTPUT:
            bbox: Bounding box (numpy array)
        '''

        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):

        '''
        Update id

        PARAMETERS:
            id: Id (int)

        OUTPUT:
            None
        '''

        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, color=[0, 224, 255]):

        '''
        Draw pose

        PARAMETERS:
            img: Image (numpy array)
            color: color (array)

        OUTPUT:
            None
        '''

        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                img = cv2.circle(img, (int(x_a), int(y_a)), 10, color, -1)
                
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                img = cv2.circle(img, (int(x_b), int(y_b)), 10, color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                img = cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 5)
        return img

    def getSegment(self, xyxy):

        '''
        Get segment

        PARAMETERS:
            xyxy: Bounding box (numpy array)

        OUTPUT:
            minSeg: Segment (tuple)
        '''

        assert self.keypoints.shape == (Pose.num_kpts, 2)
        distances = {}
        minSeg = None
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            if self.keypoints[kpt_a_id, 0] != -1 and self.keypoints[kpt_b_id, 0] != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                x_b, y_b = self.keypoints[kpt_b_id]
                pt = (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2))
                distances.update({(kpt_a_id,kpt_b_id): pDistance(pt[0], pt[1], x_a, y_a, x_b, y_b)})
                minSeg = (min(distances, key=distances.get))

        return minSeg

def get_similarity(a, b, threshold=0.5):

    '''
        Get similarity between two poses

        PARAMETERS:
            a: Pose (Pose)
            b: Pose (Pose)
            threshold: Threshold (float)

        OUTPUT:
            num_similar_kpt: Number of similar keypoints (int)
    '''

    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt

def track_poses(previous_poses, current_poses, threshold=3, smooth=False):

    '''
        Propagate poses ids from previous frame results. Id is propagated,
        if there are at least `threshold` similar keypoints between pose from previous frame and current.
        If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.
        

        PARAMETERS:
            previous_poses: poses from previous frame with ids (list)
            current_poses: poses from current frame to assign ids (list)
            threshold: minimal number of similar keypoints between poses (int)
            smooth: smooth pose keypoints between frames (bool)

        OUTPUT:
            None
    '''

    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)

# Get Distance from point to segment
def pDistance(x, y, x1, y1, x2, y2):

    '''
        Get Distance from point to segment

        PARAMETERS:
            x: x coordinate of point (float)
            y: y coordinate of point (float)
            x1: x coordinate of segment start (float)
            y1: y coordinate of segment start (float)
            x2: x coordinate of segment end (float)
            y2: y coordinate of segment end (float)

        OUTPUT:
            distance: distance from point to segment (float)
    '''
    

    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if (len_sq != 0):
        param = dot / len_sq

    xx = None
    yy = None

    if (param < 0):
        xx = x1
        yy = y1
    elif (param > 1):
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx * dx + dy * dy)

