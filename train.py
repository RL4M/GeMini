import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ['TOKENIZERS_PARALLELISM']='false'

import json
import torch.nn as nn
import torchvision
from transformers import HfArgumentParser, set_seed
from transformers import BertTokenizer, AutoTokenizer, logging, AutoConfig, BertModel, Trainer, ViTForImageClassification, ViTImageProcessor
import numpy as np

from src.models import *
from src.dataset import *
from src.dataloader import DataCollatorForMultimodalInput, DataCollatorForMultimodalInput_LateFusion
from src.trainer import MultilabelTrainer_KMIL
from src.training_args import ModelArguments, DataArguments, DiagnosisTrainingArguments
from src.vit import VisionTransformer

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
logger = logging.get_logger(__name__)
DEVICE = 'cuda'

from sklearn.metrics import classification_report, multilabel_confusion_matrix

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    probs = 1/(1 + np.exp(-logits))
    predictions = (probs>0.5).astype(int)

    try:
        roc_auc_score(labels.astype(int), probs, average='weighted')
    except:
        labels = np.array(labels.astype(int))
        np.savetxt('validataion_labels.txt', labels, fmt='%d')

    return {"roc_auc_score": roc_auc_score(labels.astype(int), probs, average='weighted'),
            "roc_auc_score_macro": roc_auc_score(labels.astype(int), probs, average='macro'),
        "precision_score": precision_score(labels.astype(int), predictions, average='macro'),
            "recall_score": recall_score(labels.astype(int), predictions, average='macro'),
            "f1_score": f1_score(labels.astype(int), predictions, average='macro'),
            'accuracy': accuracy_score(labels.astype(int), predictions)}


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def main():
    parser = HfArgumentParser((DiagnosisTrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # check output_dir
    os.makedirs(training_args.output_dir, exist_ok=True)

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

    tokenizer = BertTokenizer.from_pretrained(model_args.text_model_path)
    vit_processor16 = ViTImageProcessor.from_pretrained(model_args.img_model_path16)
    vit_processor32 = ViTImageProcessor.from_pretrained(model_args.img_model_path32)

    train_dataset = MMCaD(data_args.data_dir, data_args.cxr_dir, data_args.train_idx_path,data_args.prepared_data_path,data_args.icd_diagnosis_threshold, model_args.text_model_path,tokenizer, vit_processor16, vit_processor32)

    val_dataset = MMCaD(data_args.data_dir, data_args.cxr_dir, data_args.val_idx_path,data_args.prepared_data_path,data_args.icd_diagnosis_threshold,model_args.text_model_path, tokenizer, vit_processor16, vit_processor32)

    print("pad token: ",tokenizer.pad_token_id)
    print("Number of label:",train_dataset.num_labels)


    # init data collator
    data_collator = DataCollatorForMultimodalInput(tokenizer=tokenizer,mask=training_args.mask_input)
    config = AutoConfig.from_pretrained(model_args.text_model_path)

    #
    config.max_position_embeddings = 1024
    config.type_vocab_size = 7

    # update config with data stats for nn.Embedding in model
    config.num_labels = train_dataset.num_labels
    config.num_labevent_category = train_dataset.num_labevent_category

    config.num_micro_spec_itemid_category = train_dataset.num_micro_spec_itemid_category
    config.num_micro_test_itemid_category = train_dataset.num_micro_test_itemid_category
    config.num_micro_org_itemid_category = train_dataset.num_micro_org_itemid_category
    config.num_micro_ab_itemid_category = train_dataset.num_micro_ab_itemid_category
    config.num_micro_dilution_comparison_category = train_dataset.num_micro_dilution_comparison_category

    config.num_patient_category = train_dataset.num_patient_category
    config.num_triage_category = train_dataset.num_triage_category
    config.text_model_path = model_args.text_model_path
    config.img_model_path16 = model_args.img_model_path16
    config.img_model_path32 = model_args.img_model_path32
    config.keys_to_ignore_at_inference=['hidden_states','attentions']
    config.projection_dim = 512

    model = GEMINI_scratch(config)

    '''
    Default optimizer is AdamW, linear Schedular with warmp up
    See https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
    '''

    trainer = MultilabelTrainer_KMIL(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, #val_dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tuning
    if training_args.do_train:
        trainer.train()

    trainer.save_model(training_args.output_dir)


    # Prediction
    logger.info("**** Test ****")

if __name__ == "__main__":
    main()