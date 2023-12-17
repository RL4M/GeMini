import os
import pickle
import random
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ['TOKENIZERS_PARALLELISM']='false'

import logging
from tqdm import tqdm
import traceback
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import dataclasses
from transformers import HfArgumentParser, set_seed
from transformers import BertTokenizer, AutoTokenizer, logging, AutoConfig, BertModel, Trainer, TrainingArguments, ViTImageProcessor
import numpy as np

from src.models import *
from src.dataset import MMCaD
from src.dataloader import DataCollatorForMultimodalInput
from src.trainer import *
from src.training_args import ModelArguments, DataArguments, DiagnosisTrainingArguments
from src.vit import VisionTransformer

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
logger = logging.get_logger(__name__)
DEVICE = 'cuda'

from sklearn.metrics import classification_report, multilabel_confusion_matrix
def compute_metrics(logits,labels):

    probs = 1/(1 + np.exp(-logits))
    predictions = (probs>0.5).astype(int)

    return {"roc_auc_score_micro": roc_auc_score(labels.astype(int), probs, average='weighted'),
            "roc_auc_score_macro": roc_auc_score(labels.astype(int), probs, average='macro'),
    "precision_score": precision_score(labels.astype(int), predictions, average='weighted'),
    "recall_score": recall_score(labels.astype(int), predictions, average='weighted'),
    "f1_score": f1_score(labels.astype(int), predictions, average='weighted'),
            'accuracy': accuracy_score(labels.astype(int), predictions)}


def main():
    print('Start parsing args')
    parser = HfArgumentParser((DiagnosisTrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    print('Finish parsing args')

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info(f"Training parameters: {training_args}")

    set_seed(training_args.seed)

    bert_tokenizer = BertTokenizer.from_pretrained(model_args.text_model_path)
    vit_processor16 = ViTImageProcessor.from_pretrained(model_args.img_model_path16)
    vit_processor32 = ViTImageProcessor.from_pretrained(model_args.img_model_path32)
    print("pad token: ",bert_tokenizer.pad_token_id)

    test_dataset = MMCaD(data_args.data_dir,data_args.cxr_dir, data_args.test_idx_path,data_args.prepared_data_path,data_args.icd_diagnosis_threshold,model_args.text_model_path, bert_tokenizer, vit_processor16, vit_processor32)


    # init data collator
    data_collator = DataCollatorForMultimodalInput(tokenizer=bert_tokenizer,mask=training_args.mask_input)

    config = AutoConfig.from_pretrained(model_args.model_path)
    #
    config.max_position_embeddings = 1024
    config.type_vocab_size = 7

    # update config with data stats for nn.Embedding in model
    config.num_labels = test_dataset.num_labels
    config.num_labevent_category = test_dataset.num_labevent_category

    config.num_micro_spec_itemid_category = test_dataset.num_micro_spec_itemid_category
    config.num_micro_test_itemid_category = test_dataset.num_micro_test_itemid_category
    config.num_micro_org_itemid_category = test_dataset.num_micro_org_itemid_category
    config.num_micro_ab_itemid_category = test_dataset.num_micro_ab_itemid_category
    config.num_micro_dilution_comparison_category = test_dataset.num_micro_dilution_comparison_category

    config.num_patient_category = test_dataset.num_patient_category
    config.num_triage_category = test_dataset.num_triage_category

    config.model_path = model_args.model_path


    model = GEMINI_scratch.from_pretrained(model_args.model_path)

    '''
    Default optimizer is AdamW, linear Schedular with warmp up
    See https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
    '''

    trainer = MultilabelTrainer_KMIL(
        model=model,
        args=training_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Prediction
    logger.info("**** Test ****")

    results = trainer.predict(test_dataset)


    performance = compute_metrics(results.predictions,results.label_ids)
    print(performance)


if __name__ == "__main__":
    main()

