import torch
import numpy as np

from utils.dataloaders import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend

class Yolo():

    def __init__(self, weights, device=0) -> None:
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        print('Classes: ' + str(self.names)+'\n')
        self.model.warmup()

    def calc(self, image):
        im = letterbox(image, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2,0,1))[::-1] # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im) # contiguous

        im = torch.from_numpy(im).to(self.device).float()
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im) # Inference
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45) # NMS

        for i, det in enumerate(pred):  # detections per image

            # Annotator
            annotator = Annotator(image, line_width=3, example=str(self.names))

            if det is not None and len(det):
                pred[i][:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class

                bbox = det[0][0:4]
                conf = det[0][4]
                cls = det[0][5]
                c = int(cls)  # integer class

                # Draw bbox and annotate
                label = f'{self.names[c]} {conf:.2f}'
                color = colors(c, True)
                annotator.box_label(bbox, label, color=color)

        return image, pred