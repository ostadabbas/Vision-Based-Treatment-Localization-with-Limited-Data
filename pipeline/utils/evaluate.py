import os
import matplotlib.pyplot as plt
import math
import cv2
import copy
from pandas import *
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import json
import seaborn as sns
from matplotlib.patches import Rectangle

def output_results(source, kpt_names, names, joint_hits, prominences, opt, new_dir):

    #def output_results(source, kpt_names, names, joint_hits, prominences):

    #new_dir = source.rsplit('/', 1)[-1].split('.')[0]
    filename = './'+ new_dir +'/results.txt'
    filename_params = './'+ new_dir +'/params.txt'
    os.mkdir(new_dir+'/figs')
    #print(joint_hits)
    with open(filename, 'w') as f:
        num_tr = len(joint_hits)
        for tr_idx, tr_jt_hits in enumerate(joint_hits):
            f.write('--------------------------------'+names[tr_idx].upper()+'--------------------------------------')
            f.write('\n')
            for p in prominences:
                peaks, _ = find_peaks(tr_jt_hits, prominence=p)
                f.write('Prominence: ')
                f.write(str(p))
                f.write('\n')
                choosen_kpts = [kpt_names[i] for i in peaks]
                for kpt in choosen_kpts:
                    f.write(kpt)
                    f.write(', ')
                f.write('\n')
            f.write('\n')
            for idx_jt, jt_score in enumerate(tr_jt_hits):
                f.write(kpt_names[idx_jt])
                f.write('\t')
                f.write(str(jt_score))
                f.write('\n')
            plt.figure(tr_idx)
            plt.bar(kpt_names, tr_jt_hits)
            title = names[tr_idx] + 'Distribution'
            plt.suptitle(title)
            plt.xticks(rotation='vertical')
            img_name = './'+ new_dir + '/figs/' + names[tr_idx] + '.png'
            plt.savefig(img_name,dpi=400)
            plt.tight_layout()
            plt.clf()

            plt.figure(tr_idx+num_tr)
            plt.bar(limb_labels[0:-1], jt_to_limb(tr_jt_hits))
            title = names[tr_idx] + 'Distribution'
            plt.suptitle(title)
            plt.xticks(rotation='vertical')
            img_name = './'+ new_dir + '/figs/limb_' + names[tr_idx] + '.png'
            plt.savefig(img_name,dpi=400)
            plt.tight_layout()
            plt.clf()
    with open(filename_params, 'w') as f:
        f.write("Minimum Joints: ")
        f.write(str(opt.min_jts))
        f.write('\n')
        f.write("Max Allowed Distance: ")
        f.write(str(opt.max_allowed_dist_pct))
        f.write('\n')
        f.write("Minimum Joint Area: ")
        f.write(str(opt.min_jt_area))
        f.write('\n')



def map_tr_to_jts(bbox, patient_kpts, new_dir, joint_hits, frame_idx, class_idx, orig_img, max_allowed_dist_pct, num_jts_detected, min_jt_count, save_img=True):
#def map_tr_to_jts(bbox, patient_kpts, new_dir, joint_hits, frame_idx, class_idx, orig_img, max_allowed_dist_pct, save_img=True):
    '''
    This function is where the magic happens. It is where we decide what will count as a hit. That is, when do we map a treatment to a joint and consider 
    it in our final calculation. Additionally, this function also writes all frames which are considered as hits to an image, which is saved to the images folder.
    
    Parameters:
    bbox: This is the bbox for the treatment
    patient_kpts: The detected key points of the patient
    new_dir: The directory to which everything from the run will be saved
    joint_hits: This is a numpy array of shape (num_of_tr, num_of_jts) it is built up such 
                that an index [x,y] refers to the number of hits for the xth treatment at the yth joint 
    frame_idx: refers to the frame number of the video
    class_idx: refers to which treatment class is currently being investigated, by that class' index
    orig_img: this is the plain cv2 image we are currently investigating
    max_allowed_dist_pct: The distance as a percentage of the frame to which a treatment may be from the nearest joint/limb to count as a hit

    Returns: 
    min_idx: Will please fill this out, I'm not sure what you are using min_idx for
    '''
    y_frame_len, x_frame_len, _ = orig_img.shape
    frame_cross_dist = math.dist([0,0], [x_frame_len, y_frame_len])
    max_allowed_dist = int(frame_cross_dist * max_allowed_dist_pct)
    center_x = (bbox[0]+bbox[2])/2
    center_y = (bbox[1]+bbox[3])/2
    min_dist = 100000 # TODO: make relative to bbox size
    min_idx = None
    
    #cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255) # white
    thickness = 2
    
    for idx, kpt in enumerate(patient_kpts):
        if kpt[0] > 0 and kpt[1] > 0:
            d = math.dist([center_x, center_y], kpt)
            if d < min_dist:
                min_dist = d
                min_idx = idx
    if min_dist < max_allowed_dist and num_jts_detected >= min_jt_count:
        joint_hits[class_idx,min_idx] += 1 #TODO This is the modification needed for a non-binary mapping function
        if save_img:
            os.mkdir('./'+new_dir+'/images') if not os.path.exists('./'+new_dir+'/images') else None
            filename = './'+ new_dir +'/images/' + str(frame_idx) + '.jpg'
            if min_idx is not None:
                orig_img = cv2.line(orig_img, (int(center_x), int(center_y)), (patient_kpts[min_idx][0], patient_kpts[min_idx][1]), (0, 255, 0), 10)
            orig_img = cv2.rectangle(orig_img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255, 255, 255), 20)
            cv2.imwrite(filename, orig_img)
    elif save_img:
        os.mkdir('./'+new_dir+'/failed_imgs') if not os.path.exists('./'+new_dir+'/failed_imgs') else None
        filename = './'+ new_dir +'/failed_imgs/' + str(frame_idx) + '.jpg'
        if min_idx is not None:
            orig_img = cv2.line(orig_img, (int(center_x), int(center_y)), (patient_kpts[min_idx][0], patient_kpts[min_idx][1]), (0, 255, 0), 10)
        orig_img = cv2.rectangle(orig_img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255, 255, 255), 20)
        max_dist_str = "Max distance " + str(max_allowed_dist) + " of total distance " + str(int(frame_cross_dist))
        curr_dist_str = "Current distance: " + str(int(min_dist))
        min_jts_str = "Min joints required:  " + str(min_jt_count)
        curr_jts_str = "Current joints: " + str(int(num_jts_detected))
        
        orig_img = cv2.putText(orig_img, max_dist_str, (50,50),font, font_scale, color, thickness, cv2.LINE_AA)
        orig_img = cv2.putText(orig_img, curr_dist_str, (50,80),font, font_scale, color, thickness, cv2.LINE_AA)
        orig_img = cv2.putText(orig_img, min_jts_str, (50,110),font, font_scale, color, thickness, cv2.LINE_AA)
        orig_img = cv2.putText(orig_img, curr_jts_str, (50,140),font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(filename, orig_img)
    if min_dist < max_allowed_dist and num_jts_detected >= min_jt_count:
        return min_idx
    else: 
        return None

def get_individual_metrics(p, truth):
    '''
    This function is currently not used
    '''
    true_pos = 0
    false_neg = 0
    false_pos = 0
    pred = copy.deepcopy(p)
    for t in truth:
        if t in pred:
            pred.remove(t)
            true_pos += 1
        else:
            false_neg += 1
    false_pos = len(pred)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos+false_neg)
    return (true_pos, false_pos, false_neg), (precision, recall)

def get_data(file_path, possible_treatments = 4):
    '''

    This function gets labels from the excel file that has the truth labels. We convert these labels to a dictionary of the format:
    labels = 'video_name': {'Mapping': [(Treatment, Limb, Joint)...for all triplets], 'Treatments': List of treatments, 
                            'Joints': List of Joints, 'Limbs': List of limbs, 'Uniform': Uniform label (string), 
                            'Location': 'Indoor' or 'Outdoor', 'Occlusion': 'Yes' or 'No'} 
            for all videos
    We return this list.

    Params:
    file_path: The path to the labels' excel file
    possible_treatments: The number of treatments are YOLO model is finding for

    '''
    xls = ExcelFile(file_path)
    df = xls.parse(xls.sheet_names[0])
    df = df.to_dict()

    labels = {}
    for idx in range(len(df['Video name'])):
        vid_name = df["Video name"][idx].split('.')[0]
        labels[vid_name] = {"Mapping":[],"Treatments":[],"Joints":[],"Limbs":[],"Uniform":df["Uniform"][idx],"Location":df["Location"][idx],"Occlusion":df["Occlusion"][idx]}
        notNone = True
        count = 1
        while notNone and count <= possible_treatments:
            treatment = df["Treatment "+str(count)][idx]
            limb = df["Limb "+str(count)][idx]
            joint = df["Joint "+str(count)][idx]
            if treatment == "None" or limb == "None" or joint == "None":
                notNone = False
            else:
                labels[vid_name]["Treatments"].append(treatment)
                labels[vid_name]["Limbs"].append(limb)
                labels[vid_name]["Joints"].append(joint)
                mapped = (treatment,limb,joint)
                labels[vid_name]["Mapping"].append(mapped)
                count += 1
    #print('label dict')
    #print(labels)
    return labels
def get_truth_for_vid(vid_name, labels):
    return labels[vid_name]


#TODO Change this to not be hard coded
#tr_labels = ['ES','HD','IGEL','ND','ND_PKG','NPA','PD','SS','TRN']
#tr_labels = ['TRN', 'PD', 'CS']


#THESE MAPPINGS MAP THE EXCEL LABEL FORMAT TO THE YOLO AND OPENPOSE FORMAT
tr_labels = ['TRN', 'PD', 'CS','HD']
#tr_label_mapping = {'TRN': 'tourniquet',
#                    'PD': 'pressure_dressing',
#                    'CS': 'chest_seal',
#                    'HD': 'hemostatic_dressing'}
tr_label_mapping = {'TRN': 'tour',
                    'PD': 'pd',
                    'CS': 'cs',
                    'HD': 'hd'}

#limb_labels = ['Head', 'R_arm', 'L_arm', 'R_leg', 'L_leg', 'Torso', 'Chest']
limb_labels = ['Head', 'R_arm', 'L_arm', 'R_leg', 'L_leg', 'Torso']
def jt_to_limb(arr):
    limb_indices = [[0, 1, 14, 15, 16], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
    sums = np.array([arr[idx].sum() for idx in limb_indices])
    return sums

limb_label_mapping = {'Head': 'head',
                      'R_arm': 'right_arm',
                      'L_arm': 'left_arm',
                      'R_leg': 'right_leg',
                      'L_leg': 'left_leg',
                      'Torso': 'torso',
                      'Chest': 'torso',
                      'Back': 'torso'}
limb_indicies_map = {
                      'Head': [0,1,14,15,16,17],
                      'R_arm': [2,3], #wrist removed (kpt == 4)
                      'L_arm': [5,6], #wrist removed (kpt == 7)
                      'R_leg': [8,9,10],
                      'L_leg': [11,12,13],
                      'Torso': None,
                      'Chest': None,
                      'Back': None}

kpt_labels = ['nose', 'neck',
             'R_shoulder', 'R_elbow', 'R_wrist', 'L_shoulder', 'L_elbow', 'L_wrist',
             'R_hip', 'R_knee', 'R_ankle', 'L_hip', 'L_knee', 'L_ankle',
             'R_eye', 'L_eye',
             'R_ear', 'L_ear']

def drop_zscore_tour(zscore_drops, joint_hits, tr_idxs=[0,3,1]): #TRN,HD,PD
    '''
    This function takes as input the joint_hits array, and the counts of tour-limb mappings that should be dropped from each limb
    It returns a limb_hits_sorted --> NOTE THIS RETURNED ARRAY DOES NOT PRESERVE LIMB JOINT LEVEL INFORMATION, ONLY LIMB LEVEL INFORMATION
    It also assumes the input list is in the format right arm, left arm, right, leg, left leg
    '''
    limb_hits = copy.deepcopy(joint_hits)
    zscore_order = ['R_arm', 'L_arm', 'R_leg', 'L_leg']
    for zscore_idx, tr_idx in enumerate(tr_idxs):
        zscore_drop_copy = copy.deepcopy(zscore_drops)
        zscore_current = zscore_drop_copy[zscore_idx, 0:4] #unique to tour and pd/hd, don't consider chest
        for idx, total_count in enumerate(zscore_current):
            count = total_count
            indices_for_limb = limb_indicies_map[zscore_order[idx]]
            while count > 0:
                if limb_hits[tr_idx, indices_for_limb[0]] > 0:
                    limb_hits[tr_idx, indices_for_limb[0]] -= 1
                elif limb_hits[tr_idx, indices_for_limb[1]] > 0:
                    limb_hits[tr_idx, indices_for_limb[1]] -= 1
                elif len(indices_for_limb) == 3 and limb_hits[tr_idx, indices_for_limb[2]] > 0:
                    limb_hits[tr_idx, indices_for_limb[2]] -= 1
                else:
                    print("FAILED - ERROR")
                    return
                count -= 1
            #print(limb_hits)
    return limb_hits

def nullify_impossible_pts(joint_hits, tr_idx=[0,1,3]): #does this for TRN, PD/HD
    out_hits = copy.deepcopy(joint_hits)
    dropped_pts = [0,1,14,15,16,17,4,7] #removes head and wrists
    for idx in tr_idx:
        np.put(out_hits[idx], dropped_pts, 0)
    return out_hits
        



def parse_peaks(p_out):
    trt_pred_per_jt = []
    for idx, tr_jts in enumerate(p_out):
        result = [kpt_labels[i] for i in tr_jts if i != -1]
        trt_pred_per_jt.append(result)
    return trt_pred_per_jt

def peaks_to_np(data):
    max_len = max(len(row) for row in data)
    array_2d = np.full((len(data), max_len), -1, dtype=int)
    for i, row in enumerate(data):
        array_2d[i, :len(row)] = row
    return array_2d

def get_pred_in_label_format(jt_hits, prominences):
    '''
    Parameters:
    joint_hits: This is a numpy array of shape (num_of_tr, num_of_jts) it is built up such 
                that an index [x,y] refers to the number of hits for the xth treatment at the yth joint 
    prominences: This is a list of integers corresponding to the selected prominences 
    Returns a list of lists of length treatments where trt_per_jt[x] is a 
            list of the proposed places for the xth treatment to be mapped
            to such as ['R_elbow', 'R_knee', 'L_knee']
    '''
    all_peaks = []
    trt_per_jt = []
    for p in prominences:
        for row in jt_hits:
            peaks, _ = find_peaks(row, prominence=p)
            all_peaks.append(peaks)
        p_out = peaks_to_np(all_peaks)
        trt_per_jt.append(parse_peaks(p_out))
        all_peaks = []
    return trt_per_jt

def get_summary_metrics(p_out, labels_in):
    '''
    This function is similar to get_metrics, except that it maps p_out to limbs
    TODO: THIS FUNCTION TAKES MARC'S OUTPUT, AND TURNS IT INTO METRICS. WE NEED TO COMBINE HD AND PD OUTPUT, THAT IS BOTH ARE NOW PREDICTORS FOR THE DRESSING CLASS
    '''
    pred = copy.deepcopy(p_out)
    labels = copy.deepcopy(labels_in)
    #print(labels)
    #remove HD
    for idx, mapping in enumerate(labels["Mapping"]):
        if mapping[0] == 'HD':
            del labels['Mapping'][idx]
            del labels['Joints'][idx]
            del labels['Limbs'][idx]
            del labels['Treatments'][idx]
    # maybe add in limb metrics (i.e. separate metric for getting the limb
    # correct regardless of whether or not you get the treatment correct)
    full_metrics = np.zeros(3)
    #print(pred)
    #print(labels)
    for idx, limb in enumerate(labels["Limbs"]):
        #print(idx)
        #print(limb)
        #hits = [hit for hit in pred if hit[0] == limb_label_mapping[limb]
        #        and hit[1] == tr_label_mapping[labels["Treatments"][idx]]]
        hits = []
        for hit in pred:
            if hit[0] == limb_label_mapping[limb]:
                print(hit[1])
                if (hit[1] == 'hd' or hit[1] == 'pd') and (tr_label_mapping[labels["Treatments"][idx]] == 'hd' or tr_label_mapping[labels["Treatments"][idx]] == 'pd'):
                    hits.append(hit)
                elif hit[1] == tr_label_mapping[labels["Treatments"][idx]]:
                    hits.append(hit)
        #print(hits)
        if len(hits) > 0:
            full_metrics[0] += 1
            pred.remove(hits[0]) #I think this should be changed to remove everything that is in predictions??
        else:
            full_metrics[2] += 1

    full_metrics[1] += len(pred)
    #print(full_metrics)
    #print('----------------------')
    return full_metrics

# TODO issues with p_out
def get_metrics(p_out, truths, tr_labels=tr_labels, limbs=False):
    '''
    This function provides metrics for true pos, false pos, false neg for given prominence detection

    p_out: This is the list of predictions we have made for joints/limbs, it is such that p_out[x] is a 
            list corresponding to all the locations we predict the xth treatment at
    truths: Truths is a list of tuples s.t. each tuple is (Treatment, limb, joint)
    tr_labels: This is simply a list of the treatment labels corresponding 
                to the current YOLO treatment detection model
    limbs: Whether or not we are mapping treatments to limbs or joints. 
            False if the treatments are mapped to joints.
    Returns:
    metrics: An np array such that (# true pos, # false pos, # false neg)
    '''
    #print('--------------------------------------')
    #print(p_out)
    #print(truths)
    #print(tr_labels)
    #print('-------------------------------------')

    label_dict = {label: i for i, label in enumerate(tr_labels)}
    pred = copy.deepcopy(p_out)
    #print("preds")
    #print(pred)
    metrics = np.zeros((len(tr_labels),3)) # true pos, false pos, false neg
    for label in truths:
        index_of_tr = label_dict[label[0]]
        jt = label[2] if limbs else label[1]
        #print(jt)
        if jt in pred[index_of_tr]:
            #Add one to true pos and drop that particular joint
            metrics[index_of_tr,0] += 1
            pred[index_of_tr].remove(jt)
        else:
            #Add one to false neg
            metrics[index_of_tr,2]+= 1
    for index_of_tr, tr_list in enumerate(pred):
        metrics[index_of_tr,1] += len(tr_list)
    return metrics
''' 
def get_metrics_raw(joint_hits, truths, joint_kpts, tr_labels=tr_labels, limbs=False):
    This function provides metrics for true positive overall, true positive joints, true positive treatments, 
        false positive joints, and false positive treatments given joint_hits

    joint_hits: This is a numpy array of shape (num_of_tr, num_of_jts) it is built up such 
                that an index [x,y] refers to the number of hits for the xth treatment at the yth joint 
    truths: Truths is a list of tuples s.t. each tuple is (Treatment, limb, joint)
    tr_labels: This is simply a list of the treatment labels corresponding 
                to the current YOLO treatment detection model
    limbs: Whether or not we are mapping treatments to limbs or joints. 
            False if the treatments are mapped to joints.
    Returns:
    metrics: An np array such that (# true pos, # false pos, # false neg)


    label_dict = {label: i for i, label in enumerate(tr_labels)}
    pred = copy.deepcopy(joint_hits)
    metrics = np.zeros((len(tr_labels),5)) # true pos (TR,JT), true pos TR, true pos JT, false pos TR, false pos JT, false pose (TR, JT)
    print('GET METRICS RAW==================================================================================')
    for label in truths:
        print(label)
    for row in joint_hits:
        normalized_row = row / np.sum(row)
    
    for tr_idx, jts_for_tr in enumerate(joint_hits):
        for jt_idx, jt_count_for_tr in enumerate(jts_for_tr):
            pred_pair = (tr_labels[tr_idx], joint_kpts[jt_idx])
            pred_tr, pred_jt = pred_pair
            count = jt_count_for_tr
            pair_match_found = False
            tr_match_found = False
            jt_match_found = False
            for label in truths:
                true_pair = (label[0],label[2])
                true_tr, true_jt = true_pair
                if pred_pair == true_pair:





        index_of_tr = label_dict[label[0]]
        jt = label[2] if limbs else label[1]
        if jt in pred[index_of_tr]:
            #Add one to true pos and drop that particular joint
            metrics[index_of_tr,0] += 1
            pred[index_of_tr].remove(jt)
        else:
            #Add one to false neg
            metrics[index_of_tr,2]+= 1
    for index_of_tr, tr_list in enumerate(pred):
        metrics[index_of_tr,1] += len(tr_list)
    return metrics
'''

def get_joint_tr_indicies(label, tr_labels=tr_labels, kpt_labels=kpt_labels):
    tr, limb, jt = label
    tr_idx = tr_labels.index(tr)
    jt_idx = kpt_labels.index(jt)
    limb_idxs = limb_indicies_map[limb]
    return tr_idx, jt_idx, limb_idxs


def get_joint_hits_metrics(joint_hits, labels_full, drop_impossible_tour_points=True):
    #if drop_impossible_tour_points:
    #    impossible_indicies = limb_indicies_map['Head']
    #    tour_idx = tr_labels.index('TRN')
    #    for i in impossible_indicies:
    #        joint_hits[tour_idx, i] = 0
    labels = labels_full['Mapping']
    #print(labels)
    hits_pairs = copy.deepcopy(joint_hits)
    hits_trs = copy.deepcopy(joint_hits)
    hits_jts = copy.deepcopy(joint_hits)
    true_pos_pair_count = 0
    true_pos_tr = 0
    true_pos_jt = 0
    missed_labels = []
    metric_per_label = []

    for label in labels: #label is of the format (TR, Limb, Joint)
        true_pair, true_jt, true_label = (label[0],label[2]), label[2], label[0]
        print(label)
        tr_idx, jt_idx, _ = get_joint_tr_indicies(label) # int, int
        true_pos_pair_count += joint_hits[tr_idx, jt_idx]
        if joint_hits[tr_idx, jt_idx] == 0: missed_labels.append(label)
        hits_pairs[tr_idx, jt_idx] = 0
		
        true_pos_tr += sum(hits_trs[tr_idx, :])
        hits_trs[tr_idx, :] = 0

        true_pos_jt += sum(hits_jts[:, jt_idx])
        hits_jts[:, jt_idx] = 0

        metric_per_label.append((label, joint_hits[tr_idx, jt_idx], joint_hits[tr_idx, jt_idx]/joint_hits[tr_idx,:].sum()))

    false_pos_pair = hits_pairs.sum()
    false_pos_tr = hits_trs.sum()
    false_pos_jt = hits_jts.sum()

    total_hits = joint_hits.sum()
    pair_precision = true_pos_pair_count/total_hits
    tr_precision = true_pos_tr/total_hits
    jt_precision = true_pos_jt/total_hits

    return (true_pos_pair_count, false_pos_pair, pair_precision), (true_pos_tr, false_pos_tr, tr_precision), (true_pos_jt, false_pos_jt, jt_precision), missed_labels, metric_per_label



def get_limb_hits_metrics(joint_hits, labels_full, drop_impossible_tour_points=True):
    #Dont need drop_impossible_tour_points with nullify_impossible_pts function used in demo.py
    #if drop_impossible_tour_points:
    #    impossible_indicies = limb_indicies_map['Head']
    #    tour_idx = tr_labels.index('TRN')
    #    for i in impossible_indicies:
    #        joint_hits[tour_idx, i] = 0
    labels = labels_full['Mapping']
    hits_pairs = copy.deepcopy(joint_hits)
    hits_trs = copy.deepcopy(joint_hits)
    hits_limbs = copy.deepcopy(joint_hits)
    true_pos_pair_count = 0
    true_pos_tr = 0
    true_pos_limb = 0
    missed_labels = []
    bad_jt_counts_per_tr = []

    metric_per_label = []

    for label in labels: #label is of the format (TR, Limb, Joint)
        if label[1] == 'Chest':
            chest_cnt = hits_pairs[2, :].sum()
            true_pos_pair_count += chest_cnt
            true_pos_tr += chest_cnt
            true_pos_limb += chest_cnt
            hits_pairs[2, :] = 0
            hits_trs[2, :] = 0
            hits_limbs[2, :] = 0
            metric_per_label.append((label, chest_cnt, 1))

        #true_pair, true_jt, true_label = (label[0],label[2]), label[2], label[0]
        else:
            tr_idx, _, limb_idxs = get_joint_tr_indicies(label) # int, [ints]
            limb_cnt = 0
            for l_idx in limb_idxs:
                limb_cnt += joint_hits[tr_idx, l_idx]
                hits_pairs[tr_idx, l_idx] = 0

                true_pos_limb += hits_limbs[:, l_idx].sum()
                hits_limbs[:, l_idx] = 0

            true_pos_pair_count += limb_cnt
            if limb_cnt == 0: missed_labels.append(label)
            
            true_pos_tr += sum(hits_trs[tr_idx, :])
            hits_trs[tr_idx, :] = 0

            metric_per_label.append((label, limb_cnt, limb_cnt/joint_hits[tr_idx,:].sum())) #label, true_pos for that label, percentage of hits for the treatment that were assigned as the (tr,limb) pair specified in the label


    false_pos_pair = hits_pairs.sum()
    false_pos_tr = hits_trs.sum()
    false_pos_limb = hits_limbs.sum()

    total_hits = joint_hits.sum()

    pair_precision = true_pos_pair_count/total_hits
    tr_precision = true_pos_tr/total_hits
    limb_precision = true_pos_limb/total_hits

    for i in range(len(tr_labels)):
        if joint_hits[i,:].sum() != 0:
            bad_jt_counts_per_tr.append((tr_labels[i],hits_pairs[i,:].sum(), hits_pairs[i,:].sum()/joint_hits[i,:].sum()))

    #bad_jt_counts_per_tr --> TR Label, # incorrect hits for that TR, incorrect hits/all hits for TR
    #metric_per_label -->  Label, # correct hits for that TR, percentage of hits for the treatment that were assigned as the (tr,limb) pair specified in the label

    final_overall_metrics = (true_pos_pair_count, false_pos_pair, pair_precision), (true_pos_tr, false_pos_tr, tr_precision), (true_pos_limb, false_pos_limb, limb_precision)


    return (true_pos_pair_count, false_pos_pair, pair_precision), (true_pos_tr, false_pos_tr, tr_precision), (true_pos_limb, false_pos_limb, limb_precision), missed_labels, metric_per_label, drop_impossible_tour_points, bad_jt_counts_per_tr


def write_raw_metrics(raw_metrics, new_dir):
    pair_metrics, tr_metrics, limb_metrics, missed_labels, metric_per_label, drop_impossible_tour_points, bad_jt_counts_per_tr = raw_metrics
    filename = './'+ new_dir +'/raw_metrics_limb.txt'
    with open(filename, 'w') as f:
        if drop_impossible_tour_points:
            f.write("Note: These metrics drop facial keypoints when considering tourneqiet assignment.")
            f.write('\n')
        f.write('--------------------------------Overall Metrics--------------------------------------')
        f.write('\n')
        f.write('----------Metrics for (treatment,limb) pairs---------')
        f.write('\n')
        f.write('-- True Positive: '+ str(pair_metrics[0]))
        f.write('\n')
        f.write('-- False Positive: '+ str(pair_metrics[1]))
        f.write('\n')
        f.write('-- Precision: '+ str(pair_metrics[2]))
        f.write('\n')
        f.write('----------Metrics for treatments by video labels---------')
        f.write('\n')
        f.write('-- True Positive: '+ str(tr_metrics[0]))
        f.write('\n')
        f.write('-- False Positive: '+ str(tr_metrics[1]))
        f.write('\n')
        f.write('-- Precision: '+ str(tr_metrics[2]))
        f.write('\n')
        f.write('----------Metrics for limbs by video labels---------')
        f.write('\n')
        f.write('-- True Positive: '+ str(limb_metrics[0]))
        f.write('\n')
        f.write('-- False Positive: '+ str(limb_metrics[1]))
        f.write('\n')
        f.write('-- Precision: '+ str(limb_metrics[2]))
        f.write('\n')
        f.write('--------------------------------Per Treatment Metrics--------------------------------------')
        f.write('\n')
        for output in metric_per_label:
            f.write("The truth label for " + output[0][0] + " has " + str(int(output[1])) + " correct hits on " + output[0][1] + ", accounting for " + "{:.2%}".format(output[2]) + " of all " + output[0][0]  +" detections in this video.")
            f.write('\n')
        for output in bad_jt_counts_per_tr:
            f.write("Overall, there were " + str(int(output[1])) + " incorrect predictions for " + output[0]+ " accounting for "+ "{:.2%}".format(output[2]) + " of all " + output[0]  +" detections in this video.")
            f.write('\n')

#limb_indicies_map = {
#                      'Head': [0,1,14,15,16,17],
#                      'R_arm': [2,3,4],
#                      'L_arm': [5,6,7],
#                      'R_leg': [8,9,10],
#                      'L_leg': [11,12,13],
#                      'Torso': None,
#                      'Chest': None,
#                     'Back': None}


def output_hm_jts(joint_hits,new_dir):

    sns.heatmap(joint_hits, cmap='YlOrBr', annot=True)

    # add a rectangle patch around the (1, 1) box
    ax = plt.gca()
    cell_width = ax.get_xlim()[1] / len(joint_hits[0])
    cell_height = ax.get_ylim()[1] / len(joint_hits)
    rect = Rectangle((1 * cell_width, 1 * cell_height), cell_width, cell_height, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    
    f = './'+ new_dir + '/heatmap_joints.png'
    plt.savefig(f, dpi=300, bbox_inches='tight')


def output_metrics(source, kpt_names, names, joint_hits, labels, proms, opt, new_dir):
    output_results(source, kpt_names, names, joint_hits, proms, opt, new_dir)
    list_pred_by_prom = get_pred_in_label_format(joint_hits, proms)
    #get_metrics_raw(joint_hits, labels['Mapping'])
    #new_dir = source.rsplit('/', 1)[-1].split('.')[0]
    m = labels['Mapping']
    metrics_list = []
    if labels is not None:
        filename = './'+ new_dir +'/prominence_metrics.txt'
        with open(filename, 'w') as f:
            for idx, pred_list in enumerate(list_pred_by_prom):
                prom = proms[idx]
                f.write('--------------------------------'+'Prominence Value '+str(prom)+'--------------------------------------')
                f.write('\n')
                metrics = get_metrics(pred_list, labels['Mapping'])
                for tr_idx, m in enumerate(metrics):
                    f.write('----------'+names[tr_idx].upper()+'---------')
                    f.write('\n')
                    f.write('-- True Positive: '+ str(m[0]))
                    f.write('\n')
                    f.write('-- False Positive: '+ str(m[1]))
                    f.write('\n')
                    f.write('-- False Negative: '+ str(m[2]))
                    f.write('\n')
                f.write('TOTAL METRICS')
                total_metrics = np.sum(metrics, axis=0)
                metrics_list.append(total_metrics)
                f.write('\n')
                f.write('- True Positive: '+ str(total_metrics[0]))
                f.write('\n')
                f.write('- False Positive: '+ str(total_metrics[1]))
                f.write('\n')
                f.write('- False Negative: '+ str(total_metrics[2]))
                f.write('\n')
                f.write('\n')
    return metrics_list


def overall_metrics(excel_file):
    df = pd.read_excel(excel_file)

    # set all empty cells to zero
    df.fillna(0, inplace=True)
    cols_before_avg = df.shape[1]
    # calculate row averages and insert them at the end of each row
    df["row_average"] = ''
    df["row_sum"] = ''

    for index, row in df.iterrows():
        name = row[0]
        values = row[1:cols_before_avg]
        #print(name)
        if name != ' ':
            v_sum = values.sum()
            v_avg = values.mean()
            df["row_average"][index] = v_avg
            df["row_sum"][index] = v_sum
        else:
            break
    df.to_excel(excel_file, index=False)

from demo import run

def get_parser_args(opt):
    fileName = opt.output_dir + 'args.txt'
    with open(fileName, 'w') as f:
        for arg in vars(opt):
            f.write(f"{arg}={getattr(opt, arg)}\n")

def run_videos(sorted_vids, vid_path, opt):
    #print(f'videos: {sorted_vids}')
    label_dict = get_data(opt.labels)
    output_dict = {}
    get_parser_args(opt)
    for v in sorted_vids:
        print('-------------------------------------------STARTING NEW VIDEO---------------------------------------------------------')
        path_v = vid_path + v
        opt.source = path_v
        vid_name = opt.source.rsplit('/', 1)[-1].split('.')[0]
        raw_metrics, raw_metrics_filtered, frame_hit_rate, summary_metrics, summary_metrics_filtered, prom_metrics = run(opt, int(opt.min_jts),float(opt.max_allowed_dist_pct),float(opt.min_jt_area), labels=label_dict[vid_name])
        if raw_metrics is not None:
            (true_pos_pair_count, false_pos_pair, pair_precision), (true_pos_tr, false_pos_tr, tr_precision), (true_pos_limb, false_pos_limb, limb_precision), missed_labels, metric_per_label, drop_impossible_tour_points, bad_jt_counts_per_tr = raw_metrics
            (true_pos_pair_count_filtered, false_pos_pair_filtered, pair_precision_filtered), _, _, _, _, _, _ = raw_metrics_filtered

            output_dict[v] = {
                "Overall_TP" : true_pos_pair_count,
                "Overall_FP" : false_pos_pair,
                "Overall_Prec" : pair_precision,
                "Overall_TP_Filtered" : true_pos_pair_count_filtered,
                "Overall_FP_Filtered" : false_pos_pair_filtered,
                "Overall_Prec_Filtered" : pair_precision_filtered,
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
        #print(output_dict)
        df = pd.DataFrame(output_dict)
        excel_name = opt.output_dir + 'out.xlsx'
        df.to_excel(excel_name, index=True)
        overall_metrics(excel_name)

