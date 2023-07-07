from enum import Enum
from collections import Counter

import numpy as np

class Limbs(Enum):
    head = 0
    right_arm = 1
    left_arm = 2
    right_leg = 3
    left_leg = 4
    torso = 5

class Body:

    def __init__(self) -> None:
        self.body_parts = []
        for limb in Limbs:
            self.body_parts.append({})

    def getSummary(self):
        '''
        This function maps a treatment to each limb. #TODO Please help comment this Marc

        Returns lsit of pairs from majority vote

        Majority vote logic
        '''
        summary = []
        for limb in Limbs:
            max_frames = 0
            all_frames = []
            for treatment, frames in self.body_parts[limb.value].items():
                all_frames.extend(frames)
                if(len(frames) > max_frames):
                    max_frames = len(frames)
                    max_treatment = treatment
                    max_bp = limb.name
            # calculate occurences of same frameid
            counts = Counter(all_frames)
            counts_of_counts = Counter(counts.values())
#            print(f'RES: limb: {limb.name} frame_counts: {counts_of_counts}')

            if max_frames > 0:
                hist, _ = np.histogram(
                    self.body_parts[limb.value][max_treatment],
                    bins=max(int(max_frames/8), 10))
                summary.append((max_bp, max_treatment, max_frames, hist)) #hist is of frame numbers

        return summary


    def convert(self, names, kpt_names):
        joint_hits = np.zeros((len(names), len(kpt_names)))
        for limb in Limbs:
            for treatment, frames in self.body_parts[limb.value].items():
                joint_hits[list(names.keys())[list(names.values()).index(treatment)], limb.value] = len(frames)
        return joint_hits
    
    def clear_all(self):
        self.body_parts = []
        for limb in Limbs:
            self.body_parts.append({})

