import pandas as pd
import pickle
import os
import json
from tqdm import tqdm
import shutil


# prepared_data_dir = '/home/zhouhongyu/MIMIC/MD/data'
# with open(f'{prepared_data_dir}/train_idx3.json', 'r') as f:
#     data_identifiers = json.load(f)

# with open(f'{prepared_data_dir}/val_idx3.json', 'r') as f:
#     val_indexes = json.load(f)
#     data_identifiers.update(val_indexes)

# with open(f'{prepared_data_dir}/test_idx3.json', 'r') as f:
#     test_indexes = json.load(f)
#     data_identifiers.update(test_indexes)


# with open(f'data_indexes.json', 'w') as f:
#     json.dump(data_identifiers, f)
# raise ValueError

# data_dir = "/data1/zhouhongyu/MIMIC-CXR/physionet.org/files/mimicv/data"

# unique_image_paths = []
# for idx, values in tqdm(data_indexes.items()):
#     subject_id, hamd_id, stay_id = values
#
#     current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
#     hosp_ed_cxr_df = pd.read_csv(f'{current_data_path}/hosp_ed_cxr_data.csv')
#
#     for idx, row in hosp_ed_cxr_df.iterrows():
#         img_path = 'p' + str(row['subject_id'])[:2] + '/p' + \
#                    str(row['subject_id']) + '/s' + str(row['study_id']) + '/' + row['dicom_id'] + '.jpg'
#         unique_image_paths.append(img_path)



USER="YOUR_USERNAME"
PASSWORD="YOUR_PASSWORD"
CXR_SAVE_DIR="data/MMCaD_CXR"
os.makedirs(CXR_SAVE_DIR, exist_ok=True)

with open('unique_image_paths.json','r') as f:
    unique_image_paths = json.load(f)

with open('unique_image_paths.txt', 'w') as f:
    for line in unique_image_paths:
        f.write(f"https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/{line}\n")
print(f"total number of images {len(unique_image_paths)}")
os.system(f"parallel -a unique_image_paths.txt --jobs 300 wget --no-check-certificate -r -N -c -np --user {USER} --password {PASSWORD}")

# with open('unique_image_paths.json','r') as f:
#     unique_image_paths = json.load(f)
# for img_path in unique_image_paths:
#     os.makedirs(os.path.join(CXR_SAVE_DIR,'/'.join(img_path.split('/')[:-1])), exist_ok=True)
#     # os.system(f"wget --no-check-certificate -r -N -c -np --user {USER} --password {PASSWORD} https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/{img_path} -O {CXR_SAVE_DIR}/{img_path}")
#     os.system(f"wget --no-check-certificate -r -N -c -np --user {USER} --password {PASSWORD} https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/{img_path} -O {CXR_SAVE_DIR}/{img_path}")

