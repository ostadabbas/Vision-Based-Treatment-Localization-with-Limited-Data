from enum import Enum
from utils.interventions.body import *

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

class Intervention():

    '''
    Class for Combat Application Tourniquet (CAT) interventions
    '''

    class ValidLocations(Enum):

        '''
        Valid joint locations for tourniqet application if using keypoints
        '''

        # Right arm locations
        r_sho = 2
        r_elb = 3
        r_wri = 4

        # Left arm locations
        l_sho = 5
        l_elb = 6
        l_wri = 7

        # Right leg locations
        r_hip  = 8
        r_knee = 9
        r_ank  = 10

        # Left leg locations
        l_hip  = 11
        l_knee = 12
        l_ank  = 13

        @classmethod
        def has_value(cls, value):
            return value in cls._value2member_map_

    def __init__(self, body:Body, names) -> None:
        self.body = body
        self.last_frame = 0

        for treatment in names.values():
            for limb in Limbs:
                self.body.body_parts[limb.value].update({treatment: []})

    def add_intervention(self, limb: Limbs, frame_idx: int, treatment):

        '''
        Add a tourniquet intervention to the list of interventions assuming IoU is already calculated
        '''
        if limb.value != -1:
            self.body.body_parts[limb.value].update(
                {treatment:\
                 self.body.body_parts[limb.value].get(treatment)+[frame_idx]})

        self.last_frame = frame_idx

    def add_interventions_kpts(self, min_idx, frame_idx, treatment):

        '''
        Add a intervention (only tourniquet, pd, hd) to the list of interventions for the given frame
        
        Requires pose estimation model
        '''

        body_part = None

        # Right arm
        if min_idx == self.ValidLocations.r_sho.value:
            body_part = Limbs.right_arm
        elif min_idx == self.ValidLocations.r_elb.value:
            body_part = Limbs.right_arm
        #elif min_idx == self.ValidLocations.r_wri.value and (treatment == 'hd' or treatment == 'pd'):
        #    body_part = Limbs.right_arm

        # Left arm
        elif min_idx == self.ValidLocations.l_sho.value:
            body_part = Limbs.left_arm
        elif min_idx == self.ValidLocations.l_elb.value:
            body_part = Limbs.left_arm
        #elif min_idx == self.ValidLocations.l_wri.value and (treatment == 'hd' or treatment == 'pd'):
        #    body_part = Limbs.left_arm

        # Right leg
        elif min_idx == self.ValidLocations.r_hip.value:
            body_part = Limbs.right_leg
        elif min_idx == self.ValidLocations.r_knee.value:
            body_part = Limbs.right_leg
        elif min_idx == self.ValidLocations.r_ank.value:
            body_part = Limbs.right_leg

        # Left leg
        elif min_idx == self.ValidLocations.l_hip.value:
            body_part = Limbs.left_leg
        elif min_idx == self.ValidLocations.l_knee.value:
            body_part = Limbs.left_leg
        elif min_idx == self.ValidLocations.l_ank.value:
            body_part = Limbs.left_leg

        if body_part is not None:
            self.add_intervention(body_part, frame_idx, treatment)

    def zscore(self, frame_idx, limb_arr, window_size, threshold):
        if frame_idx > window_size:
            window = limb_arr[frame_idx-window_size:frame_idx]
            zscore = (limb_arr[frame_idx]-window.mean())/window.std()
            return abs(zscore) < threshold
        else:
            return True
    
    def process(self, save_path, zscore_window_size, zscore_threshold):


        rows = ['tour', 'hd','pd'] # Treatment 
        columns = [Limbs.right_arm, Limbs.left_arm, Limbs.right_leg, Limbs.left_leg, Limbs.torso] # Location
        
        #--> sum(right_arm) number of hits for tour-right_arm
        limb = np.zeros((len(rows), len(columns), self.last_frame))
        zscored_limb = np.zeros((len(rows), len(columns), self.last_frame))
        
        if zscore_window_size == 0: # If zscore window is 0 then zscore should not be used therefore return zero drops
            return np.zeros((len(rows), len(columns)))
        
        for frame_idx in range(self.last_frame):
            det = 0
            zscore_det = 0

            for i, treatment in enumerate(rows):
                for j, body_part in enumerate(columns):
                    if self.body.body_parts[body_part.value].get(treatment) is not None:
                        if frame_idx in self.body.body_parts[body_part.value].get(treatment):
                            limb[i,j,frame_idx] = 1
                            det+=1

                            if self.zscore(frame_idx, limb[i,j,:], zscore_window_size, zscore_threshold):
                                zscored_limb[i,j,frame_idx] = 1
                                zscore_det += 1
                            else:
                                zscored_limb[i,j,frame_idx] = 0
                                self.body.body_parts[body_part.value].get(treatment).remove(frame_idx)
                        else:
                            limb[i,j,frame_idx] = 0
                            zscored_limb[i,j,frame_idx] = 0
        for i, treatment in enumerate(rows):
            # Plotting
            fig, ax = plt.subplots(5, 2, figsize=(16,9))
            xnew = np.linspace(0, self.last_frame, self.last_frame)

            ax[0,0].step(xnew, limb[i][0][:])
            ax[0,0].set_title('Right Arm')

            ax[0,1].step(xnew, zscored_limb[i][0][:])
            ax[0,1].set_title('Zscore Right Arm')

            ax[1,0].step(xnew, limb[i][1][:])
            ax[1,0].set_title('Left Arm')

            ax[1,1].step(xnew, zscored_limb[i][1][:])
            ax[1,1].set_title('Zscore Left Arm')

            ax[2,0].step(xnew, limb[i][2][:])
            ax[2,0].set_title('Right Leg')

            ax[2,1].step(xnew, zscored_limb[i][2][:])
            ax[2,1].set_title('Zscore Right Leg')

            ax[3,0].step(xnew, limb[i][3][:])
            ax[3,0].set_title('Left Leg')

            ax[3,1].step(xnew, zscored_limb[i][3][:])
            ax[3,1].set_title('Zscore Left Leg')

            ax[4,0].step(xnew, limb[i][4][:])
            ax[4,0].set_title('Torso')

            ax[4,1].step(xnew, zscored_limb[i][4][:])
            ax[4,1].set_title('Zscore Torso')

            for k in range(5):
                for l in range(2):
                    ax[k,l].set_xlim(left = 0)
                    ax[k,l].set_ylim(bottom = 0)
                    ax[k,l].set_xlabel('Frame Number')
                    ax[k,l].set_ylabel('Hits')

            plt.suptitle(str(rows[i]) + ' Detection Graphs')
            plt.tight_layout()
            #plt.show()
            plt.savefig(save_path+'/'+str(rows[i])+'_step.png')

        return np.sum(limb, axis=2)-np.sum(zscored_limb, axis=2) # Dropped frames from Zscore

