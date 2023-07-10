import numpy as np
import cv2

# TODO: Better docs needed
class GeometricPrediction():

    '''
    Geomtric predictions
    '''

    # 2d template points
#    template = np.asarray([[0, 0], [0, 2], [1,0], [1,2]], dtype=np.float32)
    # should have this as a constructor initialization

    template = np.asarray([[1020, 177],
                           [1054, 227],
                           [961, 244],
                           [852, 312],
                           [717, 421],
                           [1155, 219],
                           [1274, 320],
                           [1206, 286],
                           [1080, 472],
                           [1105, 708],
                           [1130, 877],
                           [1215, 421],
                           [1324, 641],
                           [1434, 877],
                           [1004, 168],
                           [1029, 151],
                           [978, 185],
                           [1063, 160]], dtype=np.float32)

    def __init__(self, n_points):
        self.n_points = n_points
        self.n_measures = n_points*2
        self.procrustes_template = self.template.copy()
        self.pts_list = []

    def resetProcrustes(self):
        self.pts_list = []
        self.procrustes_template = self.template.copy()

    def updateProcrustes(self, pts_or_pts_list): # , img, pose):
        pts_list = []
        if not isinstance(pts_or_pts_list, list):
            pts_list.append(pts_or_pts_list)
        else:
            pts_list = pts_or_pts_list

        for points in pts_list:
            assert len(points) == self.n_points, 'Incorrect # of points'
            assert not np.where(points == [-1, -1])[0].any(),\
                'Must not have missing key points'

        self.pts_list.extend(pts_list.copy())
        # print(f'pts_list: {pts_list}\n self.pts_list: {self.pts_list}')
        self.computeProcrustes()#img, pose)

    def computeProcrustes(self): #, img, pose):
        list_len = len(self.pts_list)
        # print(f'list length: {list_len}')
        if list_len == 0:
            return
#        for points in self.pts_list:
#            pose.keypoints = points.copy()
#            pose.draw(img, [0, 0, 255])

#        if list_len == 2:
#            for i in range(18):
#                for j in range(0, list_len, 2):
#                    x_a, y_a = self.pts_list[j][i]
#                    x_b, y_b = self.pts_list[j+1][i]
#                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)),\
#                             [0, 255, 255], 5)

        num_iter = 0
        while num_iter < 1000:
            num_iter = num_iter + 1
            new_mean = np.zeros(self.template.shape, dtype=np.float32)
            for points in self.pts_list:
#                for i in range(18):
#                    x_a, y_a = points[i]
#                    x_b, y_b = self.procrustes_template[i]
#                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)),\
#                             [0, 255, 255], 5)
                # compute transformation from points to current mean
#                print(f'points: {points}\nproc_template: {self.procrustes_template}')
                transform, _ = cv2.estimateAffinePartial2D(
                    from_=points, to=self.procrustes_template,
                    method=cv2.LMEDS)
                transformed_pts = np.dot(points, transform[:, :2].T)\
                    + transform[np.newaxis, :, 2]
#                print(f'transformed points: {transformed_pts}')

                # add to new_mean
                new_mean = new_mean + transformed_pts * 1./list_len
#            pose.keypoints = new_mean.copy()
#            pose.draw(img, [255, 0, 0])

            # compute distance between new_mean and previous template
            error = np.linalg.norm(self.procrustes_template - new_mean)
            # print(f'error: {error}')
            self.procrustes_template = new_mean.copy()

#            height, width, _ = img.shape
#            cv2.namedWindow('template', cv2.WINDOW_NORMAL)
#            cv2.resizeWindow('template', int(width / 2), int(height / 2))
#            cv2.imshow('template', img)
            # stop if we have converged
            if error < 1E-2:
                break
    def correctProcrustes(self, pts):
        return self.correct(pts, self.procrustes_template)

    def correctStatic(self, pts):
        return self.correct(pts, self.template)

    def correct(self, pts, the_template):
        points = np.copy(pts)
        # print(f'self.template = {self.template}')
        assert len(points) == self.n_points, 'incorrect # of points'

        indices_for_missing = np.where(points == [-1, -1])[0]
        # print(f'indices_for_missing: {indices_for_missing"')

        #print(the_template)
        if indices_for_missing.any():
            incomplete_template = np.delete(
                the_template, indices_for_missing, axis=0)
            incomplete_pts = np.delete(pts, indices_for_missing, axis=0)

            transform, _ = cv2.estimateAffinePartial2D(
                from_=incomplete_template, to=incomplete_pts, method=cv2.LMEDS)
            transformed_template = np.dot(the_template, transform[:, :2].T)\
                + transform[np.newaxis, :, 2]

            points[indices_for_missing] =\
                transformed_template[indices_for_missing]

        return points
#
#    def predict(self, pts):
#        points = np.copy(pts)
#        # print(f'self.template = {self.template}')
#        assert len(points) == self.n_points, 'incorrect # of points'
#
#        indices_for_missing = np.where(points == [-1, -1])[0]
#        if indices_for_missing.any():
#            incomplete_template = np.delete(
#                self.template, indices_for_missing, axis=0)
#            pts = np.delete(pts, indices_for_missing, axis=0)
#        else:
#            incomplete_template = self.template
#
#        transform, _ = cv2.estimateAffinePartial2D(
#            incomplete_template, pts, False)
#        transformed_template = np.dot(self.template, transform[:, :2].T)\
#            + transform[np.newaxis, :, 2]
#
#        points[indices_for_missing] = transformed_template[indices_for_missing]
#
#        return points, transform
#
#    procrustes_template = template.copy()
#    procrustes_template -= np.mean(procrustes_template, 0)
#    def predictProcrustes(self, pts):
#        points = np.copy(pts)
#        # print(f'self.template = {self.template}')
#        assert len(points) == self.n_points, 'incorrect # of points'
#
#        indices_for_missing = np.where(points == [-1, -1])[0]
#        if indices_for_missing.any():
#            incomplete_template = np.delete(
#                self.procrustes_template, indices_for_missing, axis=0)
#            pts = np.delete(pts, indices_for_missing, axis=0)
#        else:
#            incomplete_template = self.procrustes_template
#
#        transform, _ = cv2.estimateAffinePartial2D(
#            incomplete_template, pts, False)
#        transformed_template = np.dot(self.procrustes_template, transform[:, :2].T)\
#            + transform[np.newaxis, :, 2]
#
#        points[indices_for_missing] = transformed_template[indices_for_missing]
#
#        transformed_template -= np.mean(transformed_template, 0)
#        norm_template = np.linalg.norm(self.procrustes_template)
#        norm_transformed = np.linalg.norm(transformed_template)
#
#        if norm_template == 0 or norm_transformed == 0:
#            raise ValueError("Input matrices must contain >1 unique points")
#
#        self.procrustes_template /= norm_template
#        transformed_template /= norm_transformed
#
#        R, s = orthogonal_procrustes(transformed_template, self.procrustes_template)
#        self.procrustes_template = np.dot(self.procrustes_template, R.T) * s
#
#        return points, transform
#                self.procrustes_template, indices_for_missing, axis=0)
#            pts = np.delete(pts, indices_for_missing, axis=0)
#        else:
#            incomplete_template = self.procrustes_template
#
#        transform, _ = cv2.estimateAffinePartial2D(
#            from_ = incomplete_template,
#            to = pts,
#            method = cv2.LMEDS
#        )
#
#
#        transformed_template = np.dot(self.procrustes_template, transform[:, :2].T)\
#            + transform[np.newaxis, :, 2]
#
#        points[indices_for_missing] = transformed_template[indices_for_missing]
#
#        transformed_template -= np.mean(transformed_template, 0)
#        norm_template = np.linalg.norm(self.procrustes_template)
#        norm_transformed = np.linalg.norm(transformed_template)
#
#        if norm_template == 0 or norm_transformed == 0:
#            raise ValueError("Input matrices must contain >1 unique points")
#
#        self.procrustes_template /= norm_template
#        transformed_template /= norm_transformed
#
#        R, s = orthogonal_procrustes(self.procrustes_template, transformed_template)
#        transformed_template = np.dot(transformed_template, R.T) * s
#        self.procrustes = np.multiply(np.add(self.procrustes_template, transformed_template),2)
#
#        return points, transform
