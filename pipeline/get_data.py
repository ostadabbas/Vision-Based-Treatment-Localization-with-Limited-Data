import os
import pandas as pd

# Path to the top-level directory containing the subdirectories with the excel files
directory_path = './results_pose_20-04-2023/'
filename = 'results_pose_20-04-2023.txt'
output_dict = {}
with open(filename, 'w') as f:
    # Loop through each subdirectory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        dir_count = len(os.listdir(subdir_path)) - 2
        # Loop through each file in the subdirectory
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
         
            # Check if the file is an excel file
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                
                # Load the excel file as a pandas dataframe
                df = pd.read_excel(file_path)
                
                # Extract the desired information from the dataframe
                
                #index for average column
                col_index_avg = df.columns.get_loc('row_average')
                #index for sum column
                col_index_sum = df.columns.get_loc('row_sum')
                #Overall precision
                row_index_prec = df.index[df.iloc[:, 0] == 'Overall_Prec'][0]
                overall_prec = df.iloc[row_index_prec, col_index_avg]

                #Overall precision
                row_index_prec = df.index[df.iloc[:, 0] == 'Overall_Prec'][0]
                overall_prec = df.iloc[row_index_prec, col_index_avg]

                #Overall precision filtered by Z score
                row_index_prec_filtered = df.index[df.iloc[:, 0] == 'Overall_Prec_Filtered'][0]
                overall_prec_filtered = df.iloc[row_index_prec_filtered, col_index_avg]
                
                #Overall precision filtered by Z score
                tccc_tp = int(df.iloc[df.index[df.iloc[:, 0] == 'TCCC TP'][0], col_index_sum])
                tccc_fp = int(df.iloc[df.index[df.iloc[:, 0] == 'TCCC FP'][0], col_index_sum])
                tccc_fn = int(df.iloc[df.index[df.iloc[:, 0] == 'TCCC FN'][0], col_index_sum])
                tccc_prec = str(tccc_tp/(tccc_tp+tccc_fp))
                tccc_recall = str(tccc_tp/(tccc_tp+tccc_fn))
                
                #Save Result to Excel
                output_dict[subdir]={
                    "Precision": round(float(overall_prec),4),
                    "Precision Filtered by Zscore": round(float(overall_prec_filtered),4),
                    "TCCC Precision": round(float(tccc_prec),4),
                    "TCCC Recall": round(float(tccc_recall),4),
                    "Number of Videos Used": str(dir_count)
                }

                # Write the result
                f.write(f"=============================={subdir}==============================")
                f.write('\n')
                f.write(f"Precision: {overall_prec}")
                f.write('\n')
                f.write(f"Precision Filtered by Zscore: {overall_prec_filtered}")
                f.write('\n')
                f.write(f"TCCC Precision: {tccc_prec}")
                f.write('\n')
                f.write(f"TCCC Recall: {tccc_recall}")
                f.write('\n')
                f.write(f"Number of Videos Used: {str(dir_count)}")
                f.write('\n')
df = pd.DataFrame(output_dict)
df = df.transpose()
excel_name = 'results_pose_20-04-2023.xlsx'
df.to_excel(excel_name, index=True)