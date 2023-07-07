import numpy as np
import cv2
import os

class FeatureTransformation():

    def __init__(self):
        self.MIN_MATCH_COUNT = 10
        self.sift = cv2.SIFT_create()
        self.lastFrame = None

    
    def calc(self, frame, frame_idx, save_dir = None):
        '''
        Calculates transformation matrix between two frames

        PARAMETERS:
            frame: Current frame (cv2.Mat)

        OUTPUT:
            M: 3x3 transformation matrix (np.ndarray)
        '''

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (self.lastFrame is None):
            self.lastFrame = frame
            return np.identity(3) # Return transformation matrix as the identity matrix

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(self.lastFrame,None)
        kp2, des2 = self.sift.detectAndCompute(frame,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []

        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>self.MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            # Save interframe mappings
            if save_dir is not None:
                os.mkdir('./'+save_dir+'/feature_mappings') if not os.path.exists('./'+save_dir+'/feature_mappings') else None
                filename = './'+ save_dir +'/feature_mappings/' + str(frame_idx) + '.jpg'

                matchesMask = mask.ravel().tolist()
                h,w = self.lastFrame.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
                out_img = cv2.drawMatches(self.lastFrame,kp1,frame,kp2,good,None,**draw_params)
                cv2.imwrite(filename, out_img)

            self.lastFrame = frame
            return M
        else:
            return np.identity(3) # Return transformation matrix as the identity matrix