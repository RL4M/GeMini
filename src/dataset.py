import os
from posixpath import join
from sys import path
import time
import torch
import json
import math
import numpy as np
import collections

import pickle

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor

from PIL import Image
import torchvision.transforms as transforms

from src.numeric_features import get_bin_feature


import pandas as pd

@dataclass
class MultimodalInput:

    image_feature : Union[torch.tensor, List[torch.tensor]]
    image_attention_mask : Union[torch.tensor, List[torch.tensor]]
    labevents : Union[torch.tensor, List[torch.tensor]]
    microbiology_input : Union[float, List[float]]
    microbiology_comment_embeddings: Union[torch.tensor, List[torch.tensor]]
    microbiology_comment_attention_mask: Union[torch.tensor, List[torch.tensor]]
    medical_history_embeddings: Union[torch.tensor, List[torch.tensor]]
    medical_history_attention_mask: Union[torch.tensor, List[torch.tensor]]
    family_history_embeddings: Union[torch.tensor, List[torch.tensor]]
    family_history_attention_mask: Union[torch.tensor, List[torch.tensor]]
    patient_input: Union[float, List[float]]
    triage_input: Union[float, List[float]]
    chiefcomplaint_embedding: Union[torch.tensor, List[torch.tensor]]
    chiefcomplaint_attention_mask: Union[torch.tensor, List[torch.tensor]]
    labels: Union[float, List[float]]
    diagnosis_text_embeddings: Dict = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def category2id(category_set):
    category_labels = {}
    category_count = 0.
    for element in category_set:
        category_labels[element] = category_count
        category_count +=1

    return category_labels


class MMCaD(Dataset):
    """
    Dataset for Multimodal Input.

    Args:
        data_dir: data directory
        idx_path: path to train/val/test index file
    """

    def __init__(
            self,
            data_dir: str,
            cxr_dir: str,
            idx_path: str,
            prepared_data_path: str,
            icd_diagnosis_threshold: int,
            model_path: str,
            tokenizer,
            vit_processor16,
            vit_processor32
    ):
        self.data_dir = data_dir
        self.cxr_dir = cxr_dir
        self.data_idx_path = idx_path
        self.prepared_data_path = prepared_data_path
        self.icd_diagnosis_threshold = icd_diagnosis_threshold
        self.model_path = model_path

        self.tokenizer = tokenizer
        self.vit_processor16 = vit_processor16
        self.vit_processor32 = vit_processor32

        self._load_data()

    def _load_data(self):
        '''
        Each input is stored as a dictionary:

         {'image_feature':image_features,
          'image_attention_mask':img_attention_mask,
          'labevents': labevents_input,
          'labevents_attention_mask': labevents_attention_mask,
          'microbiologyevents_df':microbiology_df,
          'microbiology_comment_embeddings':comment_embeddings,
          'medical_history_embeddings': medical_history_embeddings,
          'medical_history_attention_mask': medical_history_attention_mask,
          'family_history_embeddings': family_history_embeddings,
          'family_history_attention_mask': family_history_attention_mask,
          'patient_data': patient_data,
          'triage_data': triage_data,
          'chiefcomplaint_embedding': chiefcomplaint_embedding}

        '''
        with open(self.data_idx_path, 'r') as f:
            data_indexes = json.load(f)

        start_t = time.time()
        print("Loading data from ", self.data_dir, "Index path is", self.data_idx_path)

        if 'train' in self.data_idx_path:
            split = 'train'
        elif 'val' in self.data_idx_path:
            split = 'val'
        elif 'test' in self.data_idx_path:
            split = 'test'
        self.split = split


        idx_list = []
        id_list = []
        input_list = []

        debug_counter = 0
        for idx, values in tqdm(data_indexes.items()):

            idx_list.append(idx)
            id_list.append(values)
            input_list.append({})

            debug_counter+=1

        self.load_category_id_dict()

        self.idx_list = idx_list
        self.id_list = id_list
        self.input_list = input_list
        print("Number of samples", len(self.input_list))

        self.prepare_inputs()
        print('Time taken to load data: ', time.time() - start_t)

      
    def load_category_id_dict(self):

        with open(os.path.join(self.prepared_data_path, 'all_icd_code9_list.json'), 'r') as f:
            all_icd_code_dict = json.load(f)
        all_icd_code_dict = dict(collections.Counter(all_icd_code_dict))
        self.diagnosis_label_ids = {k: v for k, v in all_icd_code_dict.items() if
                                    v > self.icd_diagnosis_threshold}

        # Extracting diagnosis embeddings
        print('Extracting diagnosis text embeddings')
        icd_diagnoses_definition_dict = pd.read_csv(os.path.join(self.prepared_data_path, 'mimiciv/hosp/d_icd_diagnoses.csv.gz'),
                                                    compression='gzip')

        # only select diagnosis defined in mimic-iv hosp icd_diagnoses_definition_dict
        self.diagnosis_label_ids = {k: v for k, v in self.diagnosis_label_ids.items() if
                                    len(icd_diagnoses_definition_dict[
                                            icd_diagnoses_definition_dict['icd_code'] == k]) > 0}

        self.diagnosis_appearence_counts = list(self.diagnosis_label_ids.values())

        del self.diagnosis_label_ids['4149']

        # reset index for diagnosis labels
        counter = 0
        for key in self.diagnosis_label_ids.keys():
            self.diagnosis_label_ids[key] = counter
            counter += 1

        diagnosis_text = [icd_diagnoses_definition_dict[icd_diagnoses_definition_dict['icd_code'] == i].long_title.values[0] for i in
                          list(self.diagnosis_label_ids.keys())]

        self.diagnosis_text = diagnosis_text
        diagnosis_text_embeddings=[]
        pretrained_model = AutoModel.from_pretrained(self.model_path)
        pretrained_model.eval()
        pretrained_model.cuda()
        for i, text in tqdm(enumerate(diagnosis_text)):
            diagnosis_text_tokens = self.tokenizer(text,return_tensors ='pt')
            diagnosis_text_tokens = {k: v.cuda() for k,v in  diagnosis_text_tokens.items()}
            outputs = pretrained_model(**diagnosis_text_tokens)
            diagnosis_text_embeddings.append((outputs.last_hidden_state[:,0,:]).detach().cpu())
        self.diagnosis_text_embeddings = torch.concat(diagnosis_text_embeddings,dim=0)


        self.num_labels = len(self.diagnosis_label_ids)
        with open('diagnosis_dict_icd9.pkl', 'wb') as f:
            pickle.dump(self.diagnosis_label_ids, f)
        with open('diagnosis_counts_icd9.pkl', 'wb') as f:
            pickle.dump(self.diagnosis_appearence_counts, f)
        print('Number of labels: ', self.num_labels)

        with open(os.path.join(self.prepared_data_path, 'unique_labevent_item_id.json'), 'r') as f:
            unique_labevent_item_id = json.load(f)
        self.labevent_category_ids = category2id(['0'] + unique_labevent_item_id)
        self.num_labevent_category = len(self.labevent_category_ids)

        with open(os.path.join(self.prepared_data_path, 'micro_spec_itemid_category_ids.json'), 'r') as f:
            self.micro_spec_itemid_category_ids = json.load(f)
        self.num_micro_spec_itemid_category = len(self.micro_spec_itemid_category_ids)

        with open(os.path.join(self.prepared_data_path, 'micro_test_itemid_category_ids.json'), 'r') as f:
            self.micro_test_itemid_category_ids = json.load(f)
        self.num_micro_test_itemid_category = len(self.micro_test_itemid_category_ids)

        with open(os.path.join(self.prepared_data_path, 'micro_org_itemid_category_ids.json'), 'r') as f:
            self.micro_org_itemid_category_ids = json.load(f)
        self.num_micro_org_itemid_category = len(self.micro_org_itemid_category_ids)

        with open(os.path.join(self.prepared_data_path, 'micro_ab_itemid_category_ids.json'), 'r') as f:
            self.micro_ab_itemid_category_ids = json.load(f)
        self.num_micro_ab_itemid_category = len(self.micro_ab_itemid_category_ids)

        with open(os.path.join(self.prepared_data_path, 'micro_dilution_comparison_category_ids.json'), 'r') as f:
            self.micro_dilution_comparison_category_ids = json.load(f)
        self.num_micro_dilution_comparison_category = len(self.micro_dilution_comparison_category_ids)

        with open(os.path.join(self.prepared_data_path, 'patient_category_ids.json'), 'r') as f:
            self.patient_category_ids = json.load(f)
        self.num_patient_category = len(self.patient_category_ids)

        with open(os.path.join(self.prepared_data_path, 'triage_category_ids.json'), 'r') as f:
            self.triage_category_ids = json.load(f)
        self.triage_category_ids['-100.0'] = 0.
        self.num_triage_category = len(self.triage_category_ids)

    def prepare_inputs(self):

        # get the min and max value for numerical inputs, and prepare labevents
        min_labevent_value = -1000.0
        max_labevent_value = 1000.0

        min_microbiology_value = -100.0
        max_microbiology_value = 1000.0

        min_age_value = -100.0
        max_age_value = 100.0

        min_triage_value = -100.0
        max_triage_value = 1000.0

        self.image_names = []
        self.image=[]
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])


        with open(f'{self.prepared_data_path}/gem/2015_I10gem.txt', 'r') as f:
            lines = [line.rstrip() for line in f]
            lines = [line.split() for line in lines]
            icd_10 = []
            icd_9 = []

            for line in lines:
                icd_10.append(line[0])
                icd_9.append(line[1])
            assert len(icd_10) == len(icd_9)
            icd_gem_10to9 = pd.DataFrame({'icd_10': icd_10, 'icd_9': icd_9})
        self.icd_gem_10to9 = icd_gem_10to9

        diagnosis_label_list = []
        for i, input in tqdm(enumerate(self.idx_list)):
            id = self.idx_list[i]

            subject_id, hamd_id, stay_id = self.id_list[i]
            current_data_path = os.path.join(self.data_dir, subject_id, hamd_id, stay_id)

            hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))

            current_label = pd.read_csv(os.path.join(current_data_path, 'icd_diagnosis.csv'))
            icd_9_diagnosis_list=[]
            for k in range(len(current_label.icd_code.values)):
                if current_label.icd_version.values[k] == 9:
                    icd_9_diagnosis_list.append(str(current_label.icd_code.values[k]))
                else:
                    try:
                        icd_9_diagnosis = \
                            self.icd_gem_10to9[self.icd_gem_10to9['icd_10'] == current_label.icd_code.values[k]].icd_9.values[0]
                        icd_9_diagnosis_list.append(str(icd_9_diagnosis))
                    except:
                        # rarely, there is no conversion from icd10 to icd9, disgard these icd_codes
                        pass
            diagnosis_label_list.append(icd_9_diagnosis_list)


            # text
            discharge_summary = hosp_ed_cxr_df.loc[0, 'discharge_note_text']
            start1 = discharge_summary.lower().find('past medical history:')
            end1 = discharge_summary.lower().find('social history:', start1)

            medical_history = discharge_summary[start1:end1]
            start2 = discharge_summary.lower().find('family history:')
            end2 = discharge_summary.lower().find('physical exam:', start2)

            family_history = discharge_summary[start2:end2]

            medical_history_tokens = self.tokenizer(medical_history, return_tensors='pt', padding='max_length',
                                                    truncation=True, max_length=277)
            family_history_tokens = self.tokenizer(family_history, return_tensors='pt', padding='max_length',
                                                   truncation=True, max_length=61)
            self.input_list[i]['medical_history_embeddings'] = medical_history_tokens['input_ids']
            self.input_list[i]['medical_history_attention_mask'] = medical_history_tokens['attention_mask']
            self.input_list[i]['family_history_embeddings'] = family_history_tokens['input_ids']
            self.input_list[i]['family_history_attention_mask'] = family_history_tokens['attention_mask']


            chiefcomplaint = hosp_ed_cxr_df.loc[0, 'ed_chiefcomplaint']
            chiefcomplaint = str(chiefcomplaint)
            chiefcomplaint_tokens = self.tokenizer(chiefcomplaint, return_tensors='pt', padding='max_length',
                                                   truncation=True, max_length=22)

            self.input_list[i]['chiefcomplaint_embedding'] = chiefcomplaint_tokens['input_ids']
            self.input_list[i]['chiefcomplaint_attention_mask'] = chiefcomplaint_tokens['attention_mask']
            

            # image
            hosp_ed_cxr_df = hosp_ed_cxr_df.sort_values(by='img_charttime', ascending=True).reset_index(
                drop=True)

            current_sample_images_names =[]
            for _, row in hosp_ed_cxr_df.loc[:4,].iterrows():
                img_path = self.cxr_dir + 'p' + str(row['subject_id'])[:2] + '/p' + \
                           str(row['subject_id']) + '/s' + str(row['study_id']) + '/' + row['dicom_id'] + '.jpg'
                current_sample_images_names.append(img_path)
            self.image_names.append(current_sample_images_names)
           
           
            labevents_df = pd.read_csv(os.path.join(current_data_path, 'labevents.csv'))
            labevents_df = labevents_df[
                ['valuenum', 'ref_range_lower', 'ref_range_upper', 'itemid']].dropna().reset_index(drop=True)

            # labevent_input = np.zeros(self.num_labevent_category)
            labevent_input = []
            if len(labevents_df) > 0:
                test_itemid = list(labevents_df['itemid'].values)
                test_itemid = [self.labevent_category_ids[str(i)] for i in test_itemid]

                for idx in range(len(labevents_df)):
                    valuenum = labevents_df.loc[idx, 'valuenum']
                    valuenum = max_labevent_value if valuenum > max_labevent_value else valuenum
                    valuenum = min_labevent_value if valuenum < min_labevent_value else valuenum

                    ref_lower = labevents_df.loc[idx, 'ref_range_lower']
                    ref_upper = labevents_df.loc[idx, 'ref_range_upper']

                    if ref_lower == ref_upper:
                        calibrated_value = valuenum - ref_lower
                    else:
                        calibrated_value = (valuenum - ref_lower) / (ref_upper - ref_lower)

                    if calibrated_value > max_labevent_value:
                        max_labevent_value = calibrated_value
                    if calibrated_value < min_labevent_value:
                        min_labevent_value = calibrated_value

                    labevent_input.append([get_bin_feature(float(calibrated_value)), test_itemid[idx]])


                assert len(test_itemid) == len(labevents_df)


            self.input_list[i]['labevent_input'] = labevent_input


            microbiology_category_input = []
            microbiology_num_input = []
            microbiology_comment_embeddings = []
            microbiology_comment_attention_mask = []
            try:
                microbiologyevents_df = pd.read_csv(os.path.join(current_data_path, 'microbiologyevents.csv'))
            except:
                microbiologyevents_df = pd.DataFrame()
            if len(microbiologyevents_df) > 0:
                microbiologyevents_df = microbiologyevents_df.fillna(-100.0)
                spec_itemid = list(microbiologyevents_df['spec_itemid'].values)
                spec_itemid = [self.micro_spec_itemid_category_ids[str(i)] for i in spec_itemid]

                test_itemid = list(microbiologyevents_df['test_itemid'].values)
                test_itemid = [self.micro_test_itemid_category_ids[str(i)] for i in test_itemid]

                org_itemid = list(microbiologyevents_df['org_itemid'].values)
                org_itemid = [self.micro_org_itemid_category_ids[str(i)] for i in org_itemid]

                ab_itemid = list(microbiologyevents_df['ab_itemid'].values)
                ab_itemid = [self.micro_ab_itemid_category_ids[str(i)] for i in ab_itemid]

                dilution_comparison = list(microbiologyevents_df['dilution_comparison'].values)
                dilution_comparison = [self.micro_dilution_comparison_category_ids[str(i)] for i in dilution_comparison]

                for idx in range(len(microbiologyevents_df)):
                    dilution_value = microbiologyevents_df.loc[idx, 'dilution_value']
                    dilution_value = max_microbiology_value if dilution_value > max_microbiology_value else dilution_value
                    dilution_value = min_microbiology_value if dilution_value < min_microbiology_value else dilution_value

                    normalized_dilution_value = float(
                        (dilution_value - min_microbiology_value) / (max_microbiology_value - min_microbiology_value))

                    microbiology_category_input.append([spec_itemid[idx],
                                                        test_itemid[idx],
                                                        org_itemid[idx],
                                                        ab_itemid[idx],
                                                        dilution_comparison[idx],
                                                        ]) 
                    microbiology_num_input.append(get_bin_feature(dilution_value))

                    micro_biology_comment = str(microbiologyevents_df.loc[idx, 'comments'])
                    microbiology_comment_tokens = self.tokenizer(micro_biology_comment, return_tensors='pt', padding='max_length',
                                                    truncation=True, max_length=512)
                    microbiology_comment_embeddings.append(microbiology_comment_tokens['input_ids'])
                    microbiology_comment_attention_mask.append(microbiology_comment_tokens['attention_mask'])

            self.input_list[i]['microbiology_category_input'] = microbiology_category_input
            self.input_list[i]['microbiology_num_input'] = microbiology_num_input
            self.input_list[i]['microbiology_comment_embeddings'] = microbiology_comment_embeddings
            self.input_list[i]['microbiology_comment_attention_mask'] = microbiology_comment_attention_mask

            patient_data = hosp_ed_cxr_df.loc[0, ['gender', 'race', 'arrival_transport', 'anchor_age']].fillna(-100.0)
            patient_input = []
            patient_input.append(self.patient_category_ids[patient_data['gender']])
            patient_input.append(self.patient_category_ids[patient_data['race']])
            patient_input.append(self.patient_category_ids[patient_data['arrival_transport']])

            age = patient_data['anchor_age']
            age = max_age_value if age > max_age_value else age
            age = min_age_value if age < min_age_value else age
            patient_input.append(get_bin_feature(age))
            self.input_list[i]['patient_input'] = patient_input

            triage_data = hosp_ed_cxr_df.loc[
                0, ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp', 'ed_acuity',
                    'ed_pain']].fillna(-100.0)
            triage_input = list(
                triage_data[
                    ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp']])


            triage_input = [max_triage_value if i > max_triage_value else i for i in triage_input]
            triage_input = [min_triage_value if i < min_triage_value else i for i in triage_input]
            triage_input = [get_bin_feature(i) for i in triage_input]

            triage_input.append(self.triage_category_ids[str(triage_data['ed_pain'])])
            triage_input.append(self.triage_category_ids[str(triage_data['ed_acuity'])])
            self.input_list[i]['triage_input'] = triage_input

        converted_diagnosis_label_list = []
        for list_of_diagnosis in diagnosis_label_list:
            list_of_diagnosis = [i for i in list_of_diagnosis if i in self.diagnosis_label_ids.keys()]
            converted_diagnosis_label_list.append([self.diagnosis_label_ids[i] for i in list_of_diagnosis])
        self.target_diagnosis = converted_diagnosis_label_list


        all_diagnosis_list = []
        for diagnosis_label in converted_diagnosis_label_list:
            all_diagnosis_list+=diagnosis_label
        with open(f'all_diagnosis_list_{self.split}.json', 'w') as f:
            json.dump(list(set(all_diagnosis_list)),f)
        

    def __getitem__(self, index):
        diagnosis_text_embeddings = self.diagnosis_text_embeddings

        # label
        labels = self.target_diagnosis[index]
        target_vector = torch.zeros(self.num_labels)
        for idx in labels:
            target_vector[idx] = 1

        multimodal_data = self.input_list[index]


        image_names = self.image_names[index]
        img_count=0
        image_features = torch.zeros([4, 3, 224, 224])
        image_attention_mask = torch.ones(347)
        num_img_tokens = 0
        num_img_token_dict={1: 197,
                            2: 50,
                            3: 50,
                            4: 50}
        for img in image_names[:4]:
            img_count+=1
            try:
                image = Image.open(img).convert('RGB')
                # raise ValueError
            except:
                image_features[img_count-1] = torch.zeros([1, 3, 224, 224])
                continue
            
            if img_count ==1:
                image_feature = self.vit_processor16(images=image, size=224,return_tensors="pt")['pixel_values']
            else:
                image_feature = self.vit_processor32(images=image, size=224,return_tensors="pt")['pixel_values']

            image_features[img_count-1] = image_feature
            num_img_tokens+=num_img_token_dict[img_count]


        image_attention_mask[num_img_tokens:] = 0


        labevents = multimodal_data['labevent_input']
        microbiologyevents_input = [multimodal_data['microbiology_category_input'],
                                    multimodal_data['microbiology_num_input']]
        microbiology_comment_embeddings = multimodal_data['microbiology_comment_embeddings']
        microbiology_comment_attention_mask = multimodal_data['microbiology_comment_attention_mask']

        medical_history_embeddings = multimodal_data['medical_history_embeddings']
        medical_history_attention_mask = multimodal_data['medical_history_attention_mask']
        family_history_embeddings = multimodal_data['family_history_embeddings']
        family_history_attention_mask = multimodal_data['family_history_attention_mask']

        patient_input = multimodal_data['patient_input']

        # for triage, only last element is categorical
        triage_input = multimodal_data['triage_input']

        chiefcomplaint_embedding = multimodal_data['chiefcomplaint_embedding']
        chiefcomplaint_attention_mask = multimodal_data['chiefcomplaint_attention_mask']

        return MultimodalInput(image_feature=image_features,
                               image_attention_mask=image_attention_mask,
                               labevents=labevents,
                               microbiology_input=microbiologyevents_input,
                               microbiology_comment_embeddings=microbiology_comment_embeddings,
                               microbiology_comment_attention_mask=microbiology_comment_attention_mask,
                               medical_history_embeddings=medical_history_embeddings,
                               medical_history_attention_mask=medical_history_attention_mask,
                               family_history_embeddings=family_history_embeddings,
                               family_history_attention_mask=family_history_attention_mask,
                               patient_input=patient_input,
                               triage_input=triage_input,
                               chiefcomplaint_embedding=chiefcomplaint_embedding,
                               chiefcomplaint_attention_mask=chiefcomplaint_attention_mask,
                               labels=target_vector,
                               diagnosis_text_embeddings=diagnosis_text_embeddings)

    def __len__(self):
        return len(self.input_list)