import torch
from dataclasses import dataclass
from torch._C import dtype
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Optional, Tuple
from .dataset import MultimodalInput

import collections

def _collate_batch_for_multimodal_input(
    examples: List[MultimodalInput],
    tokenizer: PreTrainedTokenizerBase,
    mask=False
):
    '''
    mask_strategy:

    randomly mask out input types.
    '''

    bin_size = 2000
    input_batches=[]
    for i, example in enumerate(examples):
        multimodal_input_dict ={'image_feature' : example.image_feature,

        'microbiology_comment_embeddings' : example.microbiology_comment_embeddings,
        'medical_history_embeddings' : example.medical_history_embeddings,
        'family_history_embeddings' : example.family_history_embeddings,

        'chiefcomplaint_embedding' : example.chiefcomplaint_embedding,
        'labels': example.labels}



        current_labevents = example.labevents
        if len(current_labevents) >0:
            labevent_number_input = []
            labevent_category_input = []
            for i in range(min(len(current_labevents), 100)):
                labevent_number_input.append(torch.tensor(current_labevents[i][0]))
                labevent_category_input.append(torch.tensor([current_labevents[i][1]]))
            labevent_number_input = torch.concat(labevent_number_input,dim=0)
            labevent_category_input = torch.concat(labevent_category_input, dim=0)
        else:
            labevent_number_input = torch.tensor([])
            labevent_category_input = torch.tensor([])
        assert labevent_number_input.size(0) == labevent_category_input.size(0)

        number_of_labevents = 0
        if not mask:
            number_of_labevents = labevent_number_input.size(0) *2
        elif labevent_number_input.size(0) > 0:
            number_of_labevents = (labevent_number_input.size(0) - torch.randint(low=0, high=labevent_number_input.size(0),size=(1,1))[0])* 2



        labevent_attention_mask = torch.zeros([200])
        labevent_attention_mask[:number_of_labevents] = 1

        padded_labevent_number_input = torch.tensor([]).new_full([100,bin_size], fill_value=tokenizer.pad_token_id)
        padded_labevent_category_input = torch.tensor([]).new_full([100],
                                                                 fill_value=tokenizer.pad_token_id)

        if len(current_labevents) > 0:
            padded_labevent_number_input[:int(number_of_labevents/2), :] = labevent_number_input[:int(number_of_labevents/2)]

            padded_labevent_category_input[:int(number_of_labevents/2)] = labevent_category_input[:int(number_of_labevents/2)]

        multimodal_input_dict['labevent_number_input'] = padded_labevent_number_input
        multimodal_input_dict['labevent_category_input'] = padded_labevent_category_input

        microbiology_comment_input= torch.zeros([15, 512],dtype=torch.int)
        microbiology_comment_attention_mask_input = torch.zeros([15, 512],dtype=torch.int)
        # num_comments=0

        microbiology_comment_embeddings = example.microbiology_comment_embeddings[:15]
        microbiology_comment_attention_mask = example.microbiology_comment_attention_mask[:15]
        num_comments = len(microbiology_comment_embeddings)
        # there may not exist comments
        if len(microbiology_comment_embeddings) >0:
            microbiology_comment_embeddings = torch.concat(microbiology_comment_embeddings,dim=0)
            microbiology_comment_attention_mask = torch.concat(microbiology_comment_attention_mask,dim=0)
            microbiology_comment_input[:microbiology_comment_embeddings.size(0),:] = microbiology_comment_embeddings
            microbiology_comment_attention_mask_input[:microbiology_comment_attention_mask.size(0),:]= microbiology_comment_attention_mask
        multimodal_input_dict['microbiology_comment_embeddings'] = microbiology_comment_input
        multimodal_input_dict['microbiology_comment_attention_mask'] =microbiology_comment_attention_mask_input

        microbiology_category_input, microbiology_num_input = example.microbiology_input
        if len(microbiology_category_input) > 0:
            microbiology_category_input = microbiology_category_input[:15]
            microbiology_category_input = torch.concat([torch.tensor(i) for i in microbiology_category_input])
        else:
            microbiology_category_input = torch.tensor([])

        if len(microbiology_num_input) >0:
            microbiology_num_input = microbiology_num_input[:15]
            microbiology_num_input = torch.concat([torch.tensor(i) for i in microbiology_num_input])

        number_of_microbiology_events=0
        if not mask:
            number_of_microbiology_events = microbiology_category_input.size(0) + len(microbiology_num_input) + num_comments
        elif microbiology_category_input.size(0) > 0:
            # mask out a random number of microbiology measurements
            number_of_microbiology_events = microbiology_category_input.size(0) + len(microbiology_num_input) + num_comments - torch.randint(low=0, high=microbiology_category_input.size(0),size=(1,1))[0]* 15

        microbiology_attention_mask = torch.zeros([105])
        microbiology_attention_mask[:number_of_microbiology_events] = 1

        # max_microbiology=15
        padded_microbiology_category_input = torch.tensor([]).new_full([75], fill_value=tokenizer.pad_token_id)
        padded_microbiology_number_input = torch.tensor([]).new_full([15, bin_size],
                                                                 fill_value=tokenizer.pad_token_id)

        if len(microbiology_num_input) > 0:
            padded_microbiology_number_input[:microbiology_num_input.size(0),:] = microbiology_num_input

            padded_microbiology_category_input[:microbiology_category_input.size(0)] = microbiology_category_input

        multimodal_input_dict['microbiology_category_input'] = padded_microbiology_category_input
        multimodal_input_dict['microbiology_number_input'] = padded_microbiology_number_input



        # convert patient_input and triage_input into tensors
        patient_category_input = torch.tensor(example.patient_input[:-1])
        patient_number_input = torch.tensor(example.patient_input[-1])
        multimodal_input_dict['patient_category_input'] =patient_category_input
        multimodal_input_dict['patient_number_input'] = patient_number_input

        # print(example.triage_input)
        triage_category_input = torch.tensor(example.triage_input[-2:])
        triage_number_input = torch.concat([torch.tensor(i) for i in example.triage_input[:-2]],dim=0)
        multimodal_input_dict['triage_category_input'] = triage_category_input
        multimodal_input_dict['triage_number_input'] = triage_number_input

        ed_patient_attention_mask = torch.ones(len(example.patient_input) + len(example.triage_input))

        if mask:
            # 1/5 probability to mask all images
            if torch.bernoulli(torch.tensor(0.2)).bool():
                image_attention_mask = torch.zeros_like(example.image_attention_mask)
            else:
                image_attention_mask = example.image_attention_mask

            # 1/5 probability to mask all medical history
            if torch.bernoulli(torch.tensor(0.2)).bool():
                medical_history_attention_mask = torch.zeros_like(example.medical_history_attention_mask.squeeze())
            else:
                medical_history_attention_mask = example.medical_history_attention_mask.squeeze()

            # 1/5 probability to mask all family history
            if torch.bernoulli(torch.tensor(0.2)).bool():
                family_history_attention_mask = torch.zeros_like(example.family_history_attention_mask.squeeze())
            else:
                family_history_attention_mask = example.family_history_attention_mask.squeeze()

        else:
            image_attention_mask = example.image_attention_mask
            medical_history_attention_mask = example.medical_history_attention_mask.squeeze()
            family_history_attention_mask = example.family_history_attention_mask.squeeze()



        total_attention_mask = torch.concat([image_attention_mask,
                                             labevent_attention_mask,
                                             microbiology_attention_mask,
                                             medical_history_attention_mask,
                                             family_history_attention_mask,
                                             ed_patient_attention_mask,
                                             example.chiefcomplaint_attention_mask.squeeze()],dim=0) #example.chiefcomplaint_attention_mask.squeeze()

        multimodal_input_dict['medical_history_attention_mask']= example.medical_history_attention_mask.squeeze()
        multimodal_input_dict['family_history_attention_mask']=example.family_history_attention_mask.squeeze()
        multimodal_input_dict['chiefcomplaint_attention_mask']=example.chiefcomplaint_attention_mask.squeeze()

        multimodal_input_dict['total_attention_mask'] = total_attention_mask
        multimodal_input_type = torch.tensor([0]*347 + [1]*labevent_attention_mask.size(0) \
                                             + [2]* microbiology_attention_mask.size(0) + [3]*example.medical_history_attention_mask.squeeze().size(0) \
                                             + [4]*example.family_history_attention_mask.squeeze().size(0) + [5]*ed_patient_attention_mask.size(0) + [6]*example.chiefcomplaint_attention_mask.squeeze().size(0)) #(ed_patient_attention_mask.size(0)+example.chiefcomplaint_attention_mask.squeeze().size(0))
        multimodal_input_dict['multimodal_input_type'] = multimodal_input_type

        input_batches.append(multimodal_input_dict)

    # concat each individual samples
    batch = {}
    for key in input_batches[0].keys():
        batch[key] = torch.concat([sample[key].unsqueeze(0).cpu() for sample in input_batches],dim=0)
       
    batch['diagnosis_text_embeddings'] = examples[0].diagnosis_text_embeddings
    return batch


@dataclass
class DataCollatorForMultimodalInput:
    """
    Data collator for multimodal input

    Args:
        protein_tokenizer: the tokenizer used for encoding protein sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    # check tokenizer of bio-clinical bert
    tokenizer: PreTrainedTokenizerBase
    mask: bool = False

    def __call__(
            self,
            examples: List[MultimodalInput]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch_for_multimodal_input(examples, self.tokenizer, self.mask)

        return batch


@dataclass
class DataCollatorForMultimodalInput_LateFusion:
    tokenizer: PreTrainedTokenizerBase
    mask: bool = False
    def __call__(
            self,
            examples: List[MultimodalInput]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch_for_FLAVA(examples, self.tokenizer,self.mask)

        return batch
