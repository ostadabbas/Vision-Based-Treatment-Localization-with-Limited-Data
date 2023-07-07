import os
import cv2
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='''Splits videos into frames to be used for labeling''')
parser.add_argument("--source",required=True, type=str, help="Path to video source")
parser.add_argument("--output_dir", required=True, type=str, help="Path for output directory")
parser.add_argument("--fps", required=False, default=2, type=int, help="How many frames to be used in a second of footage")
args = parser.parse_args()

new_dir = args.output_dir + '/' +args.source.rsplit('/', 1)[-1].split('.')[0]
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.makedirs(new_dir)  

cap = cv2.VideoCapture(args.source)

mod = int(cap.get(cv2.CAP_PROP_FPS)/args.fps)

currentframe = 0
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while(True):
    
    ret,frame = cap.read()
  
    if ret:
        if currentframe % mod == 0:
            cv2.imwrite(new_dir +'/' +args.source.rsplit('/', 1)[-1].split('.')[0] +'-frame'+ str(currentframe) + '.jpg', frame)
        currentframe += 1
        pbar.update(1)
    else:
        break

pbar.close
cap.release()
cv2.destroyAllWindows()
