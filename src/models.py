from math import gamma
import os
import json
import copy
import pickle
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.modules.sparse import Embedding
from torch.nn.functional import normalize
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, BertModel, BertPreTrainedModel, AutoModelForSequenceClassification,FlavaPreTrainedModel, FlavaModel, ViTModel
from transformers.models.bert.modeling_bert import SequenceClassifierOutput, BertEmbeddings

from einops import repeat
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import DistilBertConfig, BertForMaskedLM
from transformers import pipeline
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List

from src.numeric_features import DICE, bin_feature

class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None
    diagnosis_text_embeddings: Optional[torch.FloatTensor] = None

class MultimodalBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.modality_distribution = [197,50,50,50,200,105,277,61,34]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        all_position_ids =[]
        for length in self.modality_distribution:
            all_position_ids.append(self.position_ids[:, 0 :length ])
        all_position_ids = torch.concat(all_position_ids,dim=-1)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(all_position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Multimodal_data_Processor(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # categorical projectors
        self.labevent_category_projector = nn.Embedding(config.num_labevent_category, config.hidden_size)

        self.micro_spec_itemid_category_projector = nn.Embedding(config.num_micro_spec_itemid_category,
                                                                 config.hidden_size)
        self.micro_test_itemid_category_projector = nn.Embedding(config.num_micro_test_itemid_category,
                                                                 config.hidden_size)
        self.micro_org_itemid_category_projector = nn.Embedding(config.num_micro_org_itemid_category,
                                                                config.hidden_size)
        self.micro_ab_itemid_category_projector = nn.Embedding(config.num_micro_ab_itemid_category, config.hidden_size)
        self.micro_dilution_comparison_category_projector = nn.Embedding(config.num_micro_dilution_comparison_category,
                                                                         config.hidden_size)

        self.patient_category_projector = nn.Embedding(config.num_patient_category, config.hidden_size)
        self.triage_category_projector = nn.Embedding(config.num_triage_category, config.hidden_size)


        # numerical projectors
        self.n_bins=2000
        self.min_bin_value= -1000.0
        self.max_bin_value = 1000.0

        self.labevent_normalized_value_fc = nn.Sequential(nn.Linear(self.n_bins,config.hidden_size), nn.ReLU())

        self.microbiology_dilution_value_fc = nn.Sequential(nn.Linear(self.n_bins,config.hidden_size), nn.ReLU())

        self.patient_anchor_age_fc = nn.Sequential(nn.Linear(self.n_bins,config.hidden_size), nn.ReLU())

        self.triage_fc = nn.Sequential(nn.Linear(self.n_bins,config.hidden_size), nn.ReLU())



    def forward(self,
                image_feature,
                labevent_number_input,
                labevent_category_input,
                microbiology_category_input,
                microbiology_number_input,
                microbiology_comment_embeddings,
                medical_history_embeddings,
                family_history_embeddings,
                patient_category_input,
                patient_number_input,
                triage_category_input,
                triage_number_input,
                chiefcomplaint_embedding,
                total_attention_mask,
                multimodal_input_type):

        ##### text input

        b, _, _ = image_feature.shape

        # project labevent inputs
        labevents_test_id = labevent_category_input.long()
        labevents_normalized_value_feature = self.labevent_normalized_value_fc(labevent_number_input)
        labevents_test_id_feature = self.labevent_category_projector(labevents_test_id)


        # gather labevent features
        batch, labevent_len_number, n_bin = labevent_number_input.size()
        labevents_embedding = torch.zeros([batch, labevent_len_number*2, self.config.hidden_size],device=labevent_number_input.device)
        labevents_embedding[:, ::2, :] = labevents_normalized_value_feature
        labevents_embedding[:, 1::2, :] = labevents_test_id_feature


        # project microbiology inputs
        micro_spec_itemid = microbiology_category_input[:, 0::5].long()
        micro_test_itemid = microbiology_category_input[:, 1::5].long()
        micro_org_itemid = microbiology_category_input[:, 2::5].long()
        micro_ab_itemid = microbiology_category_input[:, 3::5].long()
        micro_dilution_comparison = microbiology_category_input[:, 4::5].long()

        micro_spec_itemid_feature = self.micro_spec_itemid_category_projector(micro_spec_itemid)
        micro_test_itemid_feature = self.micro_test_itemid_category_projector(micro_test_itemid)
        micro_org_itemid_feature = self.micro_org_itemid_category_projector(micro_org_itemid)
        micro_ab_itemid_feature = self.micro_ab_itemid_category_projector(micro_ab_itemid)
        micro_dilution_comparison_feature = self.micro_dilution_comparison_category_projector(micro_dilution_comparison)

        micro_dilute_value = microbiology_number_input
        micro_dilute_value_feature = self.microbiology_dilution_value_fc(micro_dilute_value)

        # gather microbiology features
        microbiology_embedding = torch.zeros([batch, 105, self.config.hidden_size],device=micro_dilute_value_feature.device)

        microbiology_embedding[:,0::7] = micro_spec_itemid_feature
        microbiology_embedding[:, 1::7] = micro_test_itemid_feature
        microbiology_embedding[:, 2::7] = micro_org_itemid_feature
        microbiology_embedding[:, 3::7] = micro_ab_itemid_feature
        microbiology_embedding[:, 4::7] = micro_dilution_comparison_feature
        microbiology_embedding[:, 5::7] = micro_dilute_value_feature
        microbiology_embedding[:, 6::7] = microbiology_comment_embeddings



        # project patient inputs
        # size (batch, 3)
        patient_gender_race_arrival_transport = patient_category_input.long()
        patient_gender_race_arrival_transport_feature = self.patient_category_projector(patient_gender_race_arrival_transport)
        patient_anchor_age = patient_number_input.float()

        patient_anchor_age_feature = self.patient_anchor_age_fc(patient_anchor_age)

        # gather patient features
        patient_embedding = torch.zeros([batch, 4, self.config.hidden_size],device=patient_anchor_age_feature.device)
        patient_embedding[:,:-1, :] = patient_gender_race_arrival_transport_feature
        patient_embedding[:, -1, :] = patient_anchor_age_feature.squeeze(1)

        # project triage inputs
        triage_vital_signs = triage_number_input.float()
        triage_vital_signs_feature = self.triage_fc(triage_vital_signs)

        triage_pain = triage_category_input[:, -2].long()
        triage_pain_feature = self.triage_category_projector(triage_pain)
        triage_accuity = triage_category_input[:,-1].long()
        triage_accuity_feature = self.triage_category_projector(triage_accuity)

        # gather triage inputs
        triage_embedding = torch.zeros([batch, 8, self.config.hidden_size],device=triage_number_input.device)
        triage_embedding[:, :-2, :] = triage_vital_signs_feature
        triage_embedding[:, -2, :] = triage_pain_feature
        triage_embedding[:, -1, :] = triage_accuity_feature


        input_embeds = torch.concat([image_feature, labevents_embedding, microbiology_embedding,
                                     medical_history_embeddings,
                                     family_history_embeddings, patient_embedding, triage_embedding,
                                     chiefcomplaint_embedding], dim=1)

        return input_embeds


class GEMINI_scratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.multimodal_data_processor = Multimodal_data_Processor(config)

        self.bert = BertModel(config)
        self.bert.embeddings = MultimodalBertEmbeddings(config)

        self.text_encoder = BertModel.from_pretrained(config.text_model_path)
        for name, para in self.text_encoder.named_parameters():
            para.requires_grad = False

        self.image_encoder16 = ViTModel.from_pretrained(config.img_model_path16)
        for name, para in self.image_encoder16.named_parameters():
            para.requires_grad = False

        self.image_encoder32 = ViTModel.from_pretrained(config.img_model_path32)
        for name, para in self.image_encoder32.named_parameters():
            para.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.diagnosis_text_embedding_fc = nn.Linear(config.hidden_size, config.projection_dim)
        self.cls_token_fc = nn.Linear(config.hidden_size, config.projection_dim)
  
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        image_feature,
        labevent_number_input=None,
        labevent_category_input=None,
        microbiology_category_input=None,
        microbiology_number_input=None,
        microbiology_comment_embeddings=None,
        microbiology_comment_attention_mask=None,
        medical_history_embeddings=None,
        family_history_embeddings=None,
        patient_category_input=None,
        patient_number_input=None,
        triage_category_input=None,
        triage_number_input=None,
        chiefcomplaint_embedding=None,
        total_attention_mask: Optional[torch.Tensor] = None,
        multimodal_input_type: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        medical_history_attention_mask = kwargs.pop('medical_history_attention_mask')
        family_history_attention_mask = kwargs.pop('family_history_attention_mask')
        chiefcomplaint_attention_mask = kwargs.pop('chiefcomplaint_attention_mask')

        # image_feature size (b, 4, 3, 224, 224)
        first_img = image_feature[:,0,:,:,:]
        second_img = image_feature[:,1,:,:,:]
        third_img = image_feature[:,2,:,:,:]
        fourth_img = image_feature[:,3,:,:,:]

        vit_image_feature = torch.zeros([image_feature.size(0), 347, 768],device=image_feature.device, dtype= image_feature.dtype)
        zero_img_tensor = torch.zeros_like(first_img)
        if not torch.equal(first_img, zero_img_tensor):
            first_img_feat = self.image_encoder16(first_img).last_hidden_state
            vit_image_feature[:,:197,:] = first_img_feat
        if not torch.equal(second_img, zero_img_tensor):
            second_img = self.image_encoder32(second_img).last_hidden_state
            vit_image_feature[:,197:247,:] = second_img
        if not torch.equal(third_img, zero_img_tensor):
            third_img = self.image_encoder32(third_img).last_hidden_state
            vit_image_feature[:,247:297,:] = third_img
        if not torch.equal(fourth_img, zero_img_tensor):
            fourth_img = self.image_encoder32(fourth_img).last_hidden_state
            vit_image_feature[:,297:347,:] = fourth_img

        b, num_comments, max_tokens = microbiology_comment_embeddings.size()
        microbiology_comment_embeddings = microbiology_comment_embeddings.resize(b*num_comments,max_tokens)
        microbiology_comment_attention_mask = microbiology_comment_attention_mask.resize(b*num_comments,max_tokens)
        microbiology_comment_embeddings = self.text_encoder(input_ids = microbiology_comment_embeddings.squeeze(1),
                                                        attention_mask= microbiology_comment_attention_mask).pooler_output
        microbiology_comment_embeddings = microbiology_comment_embeddings.resize(b, num_comments, self.config.hidden_size)

        medical_history_embeddings = self.text_encoder(input_ids = medical_history_embeddings.squeeze(1),
                                                        attention_mask= medical_history_attention_mask).last_hidden_state

        family_history_embeddings = self.text_encoder(input_ids= family_history_embeddings.squeeze(1),
                                                        attention_mask=family_history_attention_mask).last_hidden_state

        chiefcomplaint_embedding = self.text_encoder(input_ids=chiefcomplaint_embedding.squeeze(1),
                                                        attention_mask=chiefcomplaint_attention_mask).last_hidden_state

        inputs_embeds = self.multimodal_data_processor(vit_image_feature,
                                                        labevent_number_input,
                                                        labevent_category_input,
                                                        microbiology_category_input,
                                                        microbiology_number_input,
                                                        microbiology_comment_embeddings,
                                                        medical_history_embeddings,
                                                        family_history_embeddings,
                                                        patient_category_input,
                                                        patient_number_input,
                                                        triage_category_input,
                                                        triage_number_input,
                                                        chiefcomplaint_embedding,
                                                        total_attention_mask,
                                                        multimodal_input_type)

        outputs = self.bert(
            input_ids=None,
            attention_mask=total_attention_mask,
            token_type_ids=multimodal_input_type,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooled_output = outputs[1]


        pooled_output = self.cls_token_fc(pooled_output)

        diagnosis_text_embeddings = kwargs.pop('diagnosis_text_embeddings')
        diagnosis_text_embeddings = self.diagnosis_text_embedding_fc(diagnosis_text_embeddings)

        return SequenceClassifierOutput(
            loss=None,
            logits=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pooler_output=pooled_output,
            diagnosis_text_embeddings=diagnosis_text_embeddings,
        )


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        # intermediate_size = int(hidden_size/2)
        # intermediate_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(intermediate_size)

        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
    def forward(self,x):
        b, d = x.size()

        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))

        return x