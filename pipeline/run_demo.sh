#!/bin/bash

# Define constant variables
#vids_paths=("../../data/Full_Pipeline_Testing/mini_man_batch1/" "../../data/Full_Pipeline_Testing/mini_batch1/")
#labels_paths=("../../data/Full_Pipeline_Testing/man_batch1.xlsx" "../../data/Full_Pipeline_Testing/batch1.xlsx")

#../../data/Full_Pipeline_Testing/hum_batches/hum_batch2/hum_batch2_ID1.xlsx
#../../data/Full_Pipeline_Testing/hum_batches/hum_batch2/hum_batch2_ID1

vids_paths=("../../data/Full_Pipeline_Testing/full_mansts_set/")
labels_paths=("../../data/Full_Pipeline_Testing/full_mansts_set.xlsx")

hpe_weights=("./weights/mann_1800.pth") #"./weights/fine_tune_no_val.pth") "./weights/huMann2_3100.pth"
hpe_weight_names=("mann_full_1800") #"fine_tune_no_val") "huMann2_full_3100" 
#vid_type=("man" "hum")
algo_types=("hpe")

# Define arrays for max_allowed_dist_pct and min_jts
max_allowed_dist_pct_arr=(.1)
min_jts_arr=(1)
min_pose_pct_arr=(.1)
current_date=$(date +'%d-%m-%Y')

# Iterate over all combinations of max_allowed_dist_pct and min_jts
for w_index in "${!hpe_weights[@]}"; do
    overall_dir="results_${hpe_weight_names[w_index]}_${current_date}"
    mkdir "$overall_dir"
    for index in "${!vids_paths[@]}"; do
        for algo in "${algo_types[@]}"; do
            for min_pose_pct in "${min_pose_pct_arr[@]}"; do
                for max_allowed_dist_pct in "${max_allowed_dist_pct_arr[@]}"; do
                        for min_jts in "${min_jts_arr[@]}"; do
                            # Define the output directory
                            output_dir="$overall_dir/${algo}_maxDIST${max_allowed_dist_pct}_minJTS${min_jts}_minPOSEpct${min_pose_pct}" #${vid_type[index]}"
                            if [ "$algo" = "of" ]; then
                                python demo.py --hpe_weights "${hpe_weights[w_index]}" --gpe --optical_flow --path_to_vids "${vids_paths[index]}" --labels "${labels_paths[index]}" --max_allowed_dist_pct "$max_allowed_dist_pct" --output_dir "$output_dir" --min_jts "$min_jts"
                                #echo --hpe_weights "${hpe_weights[w_index]}" --min_pose_percent $min_pose_pct  --gpe --optical_flow --path_to_vids "${vids_paths[index]}" --labels "${labels_paths[index]}" --max_allowed_dist_pct "$max_allowed_dist_pct" --output_dir "$output_dir" --min_jts "$min_jts"
                            else
                                python demo.py --hpe_weights "${hpe_weights[w_index]}" --"$algo" --path_to_vids "${vids_paths[index]}" --labels "${labels_paths[index]}" --max_allowed_dist_pct "$max_allowed_dist_pct" --output_dir "$output_dir" --min_jts "$min_jts" --zscore_window_size 60 --min_pose_percent "$min_pose_pct" #--write_imgs
                                #echo --hpe_weights "${hpe_weights[w_index]}" --min_pose_percent $min_pose_pct --"$algo" --path_to_vids "${vids_paths[index]}" --labels "${labels_paths[index]}" --max_allowed_dist_pct "$max_allowed_dist_pct" --output_dir "$output_dir" --min_jts "$min_jts"
                            fi
                    done
                done
            done
        done
    done
done