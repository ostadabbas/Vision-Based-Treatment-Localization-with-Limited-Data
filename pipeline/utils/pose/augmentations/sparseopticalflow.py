import numpy as np
import cv2

class SparseOpticalFlow:

    def __init__(self) -> None:
        
        self.lk_params = dict(winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners = 20,
                        qualityLevel = 0.3,
                        minDistance = 10,
                        blockSize = 7 )


        self.trajectory_len = 10
        self.detect_interval = 5
        self.trajectories = []
        self.prevgray = None

    def centroid(self, points):
        
        n = len(points)
        if n == 0:
            return None
        x_sum = sum(pt[0] for pt in points)
        y_sum = sum(pt[1] for pt in points)
        x_centroid = x_sum / n
        y_centroid = y_sum / n
        return (x_centroid, y_centroid)


    def calc(self, frame, frame_idx, pts, radius=100, draw=True):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()

        if len(self.trajectories) > 0:
            if self.prevgray is None:
                self.prevgray = frame_gray
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []

            # Get all the trajectories
            for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                if len(trajectory) > self.trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                # Newest detected point
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            self.trajectories = new_trajectories

            # Draw all the trajectories
            cv2.polylines(img, [np.int32(trajectory) for trajectory in self.trajectories], False, (0, 255, 0))

        # Update interval - When to update and detect new features
        if frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Lastest point in latest trajectory
            for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect the good features to track
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
            if p is not None:
                # If good features can be tracked - add that to the trajectories
                for x, y in np.float32(p).reshape(-1, 2):
                    self.trajectories.append([(x, y)])

        shifts = np.zeros((18,2), dtype=np.int32)
        for i in range(len(pts)):
            pt = pts[i]
            cv2.rectangle(img, (pt[0]-radius, pt[1]-radius), (pt[0]+radius, pt[1]+radius), (255,255,150))

            features_in_pts = []
            for j in range(len(self.trajectories)):
                trajectory = self.trajectories[j]
                x, y  = np.int32(trajectory[-1]) 
                if len(trajectory)>1:
                    xPast, y_past = np.int32(trajectory[-2])
                else:
                    xPast, y_past = x, y
                if x > pt[0]-radius and x < pt[0]+radius and y > pt[1]-radius and y < pt[1]+radius:
                    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)
                    features_in_pts.append((x-xPast,y-y_past))
                if len(features_in_pts):
                    shifts[i] = np.median(features_in_pts,0)
            
            cv2.putText(img, f'Kpt: {i}', 
                        (pt[0]-radius+15, pt[1]-radius+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
                
            cv2.putText(img, f'Features in Kpt: {len(features_in_pts)}', 
                        (pt[0]-radius+15, pt[1]-radius+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)

            cv2.putText(img, f'Estimated Shift: {shifts[i]}', 
                        (pt[0]-radius+15, pt[1]-radius+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
            
        self.prev_gray = frame_gray

        return img, shifts