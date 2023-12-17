from collections import defaultdict
from json import decoder
import math
from dataclasses import dataclass, field
from transformers import logging
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    model_path: str = field(
        default="None",
        metadata={"help": "The directory of the checkpoint GeMini model."}
    )

    text_model_path: str = field(
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        metadata={"help": "The directory of the text model."}
    )

    img_model_path16: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "The directory of patch16 image model."}
    )

    img_model_path32: str = field(
        default="google/vit-base-patch32-224-in21k",
        metadata={"help": "The directory of patch32 image model."}
    )

    config_path: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_path"}
    )




@dataclass
class DiagnosisTrainingArguments(TrainingArguments):

    logging_dir: str = field(
        default=None,
        metadata={"help": "Logging directory"}
    )

    resume_path: str = field(
        default=None,
        metadata={"help": "Resume training directory"}
    )

    train_epochs: int = field(
        default=100,
        metadata={"help": "Number of epochs to run"}
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )

    learning_rate: float = field(
        default=0.0001,
        metadata={"help": "Learning rate for training"}
    )

    mask_input: bool = field(
        default=False,
        metadata={"help": "Whether to randomly mask input tokens during training"}
    )

    seed: int = field(
        default=2023,
        metadata={"help": "Set seed for reproducibility"}
    )


    # distinguish steps of linear warmup on LM and KE.
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps for LM."}
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps for LM."}
    )

    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for constrastive loss."}
    )

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def get_lm_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup on LM.
        """
        warmup_steps = (
            self.lm_warmup_steps if self.lm_warmup_steps > 0 else math.ceil(num_training_steps * self.lm_warmup_ratio)
        )
        return warmup_steps


@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={"help": "the directory path of MMCaD."}
    )
    cxr_dir: str = field(
        metadata={"help": "the directory path of chest X-ray images."}
    )

    prepared_data_path: str = field(
        default=None,
        metadata={"help": "Directory containing files from prepare_data.py to initialize Dataset object"}
    )

    train_idx_path: str = field(
        default=None,
        metadata={"help": "the path of file containing list of train indexes."}
    )

    val_idx_path: str = field(
        default=None,
        metadata={"help": "the path of file containing list of val indexes."}
    )

    test_idx_path: str = field(
        default=None,
        metadata={"help": "the path of file containing list of test indexes."}
    )

    icd_diagnosis_threshold: int = field(
        default=1000,
        metadata={"help": "Threshold for filtering diagnosis label"}
    )
