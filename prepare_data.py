import json
import pickle
from tqdm import tqdm
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"



from src.dataset import MultimodalDataset
from transformers import BertTokenizer, AutoModel
import torch

data_dir = 'data/MMCaD'

with open('data/train_idx.json', 'r') as f:
    train_indexes = json.load(f)

with open('data/val_idx.json', 'r') as f:
    val_indexes = json.load(f)

with open('data/test_idx.json', 'r') as f:
    test_indexes = json.load(f)

with open('data/data_identifiers.json', 'r') as f:
    data_indexes = json.load(f)

def category2id(category_set):
    category_labels = {}
    category_count = 0.
    for element in category_set:
        category_labels[element] = category_count
        category_count += 1

    return category_labels


def get_parent_icd(code):
    code = str(code)
    try:
        # icd code start with number
        code0 = int(code[0])

        return code[:3]
    except:
        # return code
        if str(code[0]) == 'E':
            return code[:3]
        elif str(code[0]) == 'V':
            return code[:3]
    print('ICD code {0} does not start with number, "E" or "V"'.format(code))
    return code

def get_truncate_icd_diagnosis_cxr():
    with open('data/gem/2015_I10gem.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
        lines = [line.split() for line in lines]
        icd_10 = []
        icd_9 = []
    
        for line in lines:
            icd_10.append(line[0])
            icd_9.append(line[1])
        assert len(icd_10) == len(icd_9)
        icd_gem_10to9 = pd.DataFrame({'icd_10': icd_10, 'icd_9': icd_9})

    all_icd_code_9_list=[]
    indirect_conversions=[]
    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values
    
        # load 'hosp_ed_cxr_data.csv'
        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
        hosp_ed_cxr_data = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
    
        # load diagnosis label
        current_label =pd.read_csv(os.path.join(current_data_path, 'icd_diagnosis.csv'))
    
        # convert icd10 to icd9
        icd_9_diagnosis_list = []
        for i in range(len(current_label.icd_code.values)):
            if current_label.icd_version.values[i] == 9:
                icd_9_diagnosis_list.append(str(current_label.icd_code.values[i]))
            else:
                try:
                    icd_9_diagnosis = \
                    icd_gem_10to9[icd_gem_10to9['icd_10'] == current_label.icd_code.values[i]].icd_9.values[0]
                    icd_9_diagnosis_list.append(str(icd_9_diagnosis))
                except:
                    # rarely, there is no conversion from icd10 to icd9, disgard these icd_codes
                    indirect_conversions.append(str(current_label.icd_code.values[i]))
    
        all_icd_code_9_list += icd_9_diagnosis_list

    print('number of indirect_conversions:', len(set(indirect_conversions)))
    print(len(set(all_icd_code_9_list)))

    with open('data/indirect_conversions.json', 'w') as f:
        json.dump(indirect_conversions, f)
    with open('data/all_icd_code9_list.json', 'w') as f:
        json.dump(all_icd_code_9_list, f)
# get_truncate_icd_diagnosis_cxr()


def get_unique_labevent_test_id():
    d_labevent_dict = pd.read_csv('data/mimiciv/hosp/d_labitems.csv.gz', compression='gzip')

    with open('data/unique_labevent_item_id.json', 'w') as f:
        json.dump(list(d_labevent_dict['itemid'].values.astype(str)), f)

    print(type(d_labevent_dict.loc[0, 'itemid']))
    print(d_labevent_dict.loc[0, 'itemid'])
    print(d_labevent_dict.itemid.min())
    print(d_labevent_dict[d_labevent_dict['itemid'] == 50801])


def get_unique_category_ids():
    print('prepraring unique_category_ids')
    micro_spec_itemid_category_ids = set()
    micro_test_itemid_category_ids = set()
    micro_org_itemid_category_ids = set()
    micro_ab_itemid_category_ids = set()
    micro_dilution_comparison_category_ids = set()

    patient_category_ids = set()
    triage_category_ids = set()

    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values

        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)

        # with open(os.path.join(current_data_path, 'input_embeddings.pkl'), 'rb') as f:
        #     current_input = pickle.load(f)
        hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
        try:
            microbiologyevents_df = pd.read_csv(os.path.join(current_data_path, 'microbiologyevents.csv'))
        except:
            microbiologyevents_df = pd.DataFrame()
        if len(microbiologyevents_df) > 0:
            microbiologyevents_df = microbiologyevents_df.fillna(-100)
            micro_spec_itemid_category_ids = micro_spec_itemid_category_ids | set(
                list(microbiologyevents_df['spec_itemid']))
            micro_test_itemid_category_ids = micro_test_itemid_category_ids | set(
                list(microbiologyevents_df['test_itemid']))
            micro_org_itemid_category_ids = micro_org_itemid_category_ids | set(list(microbiologyevents_df['org_itemid']))
            micro_ab_itemid_category_ids = micro_ab_itemid_category_ids | set(list(microbiologyevents_df['ab_itemid']))
            micro_dilution_comparison_category_ids = micro_dilution_comparison_category_ids | set(
                list(microbiologyevents_df['dilution_comparison']))

        patient_data = hosp_ed_cxr_df.loc[0,['gender','race','arrival_transport','anchor_age']].fillna(-100.0)

        patient_category_ids = patient_category_ids | set([patient_data['gender']])
        patient_category_ids = patient_category_ids | set([patient_data['race']])
        patient_category_ids = patient_category_ids | set([patient_data['arrival_transport']])

        triage_data = hosp_ed_cxr_df.loc[0,['ed_temperature','ed_heartrate','ed_resprate','ed_o2sat','ed_sbp','ed_dbp','ed_acuity','ed_pain']].fillna(-100.0)
        triage_category_ids = triage_category_ids | set([triage_data['ed_pain']])
        triage_category_ids = triage_category_ids | set([triage_data['ed_acuity']])

    micro_spec_itemid_category_ids = category2id(micro_spec_itemid_category_ids)
    micro_test_itemid_category_ids = category2id(micro_test_itemid_category_ids)
    micro_org_itemid_category_ids = category2id(micro_org_itemid_category_ids)
    micro_ab_itemid_category_ids = category2id(micro_ab_itemid_category_ids)
    micro_dilution_comparison_category_ids = category2id(micro_dilution_comparison_category_ids)

    patient_category_ids = category2id(patient_category_ids)
    triage_category_ids = category2id(triage_category_ids)

    with open('data/micro_spec_itemid_category_ids.json', 'w') as f:
        json.dump(micro_spec_itemid_category_ids, f)
    
    with open('data/micro_test_itemid_category_ids.json', 'w') as f:
        json.dump(micro_test_itemid_category_ids, f)
    
    with open('data/micro_org_itemid_category_ids.json', 'w') as f:
        json.dump(micro_org_itemid_category_ids, f)
    
    with open('data/micro_ab_itemid_category_ids.json', 'w') as f:
        json.dump(micro_ab_itemid_category_ids, f)
    
    with open('data/micro_dilution_comparison_category_ids.json', 'w') as f:
        json.dump(micro_dilution_comparison_category_ids, f)

    with open('data/patient_category_ids.json', 'w') as f:
        json.dump(patient_category_ids, f)

    with open('data/triage_category_ids.json', 'w') as f:
        json.dump(triage_category_ids, f)


def get_numerical_variable_stats():
    labevent_values = []
    microbiologyevent_values = []
    age = []
    triage = []
    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values

        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
        try:
            hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
        except:
            # file does not exist, disgard
            continue
        labevent_df = pd.read_csv(os.path.join(current_data_path, 'labevents.csv'))
        try:
            microbiologyevent_df = pd.read_csv(os.path.join(current_data_path, 'microbiologyevents.csv'))
        except:
            microbiologyevent_df = pd.DataFrame()

        if len(labevent_df) > 0:
            labevent_values += list(labevent_df['valuenum'].dropna().values)
        if len(microbiologyevent_df) > 0:
            microbiologyevent_values += list(microbiologyevent_df['dilution_value'].dropna().values)

        age += [hosp_ed_cxr_df.anchor_age.values[0]]
        triage.append(list(
            hosp_ed_cxr_df.loc[0, ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp']]))

    with open('numerical_variable_stats.pkl', 'wb') as f:
        pickle.dump({'labevent_values': labevent_values,
                     'microbiologyevent_values': microbiologyevent_values,
                     'age': age,
                     'triage': triage}, f)


# get_unique_diagnosis_with_abnormal_cxr()
get_unique_labevent_test_id()
get_unique_category_ids()

# get_numerical_variable_stats()

def prepare_discharge_summary_embeddings(data_split_idx_path, split_type='train'):
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    with open(data_split_idx_path, 'r') as f:
        data_split_idx = json.load(f)

    discharge_summary_embeddings = {}

    for idx, values in tqdm(data_split_idx.items()):


        subject_id, hamd_id, stay_id = values

        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
        hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
        discharge_summary = hosp_ed_cxr_df.loc[0, 'discharge_note_text']

        discharge_summary_tokens = tokenizer(discharge_summary, return_tensors='pt', padding='max_length',
                                                  truncation=True, max_length=512)
        discharge_summary_tokens = {key: values.cuda() for key, values in discharge_summary_tokens.items()}
        outputs = model(**discharge_summary_tokens)
        pooler_output = outputs.pooler_output
        discharge_summary_embeddings[idx] = pooler_output.detach().cpu()

    with open(os.path.join('data', split_type, 'discharge_summary_embeddings.pkl'), 'wb') as f:
        pickle.dump(discharge_summary_embeddings, f)


