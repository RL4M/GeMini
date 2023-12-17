#!/bin/bash

OUTPUT_DIR="output/GEMINI"
DATA_DIR='data/MMCaD'
CXR_DIR='data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

PREPARED_DATA_PATH='data/'
TRAIN_IDX_PATH='data/train_idx.json'
VAL_IDX_PATH='data/val_idx.json'
TEST_IDX_PATH='data/test_idx.json'

ICD_DIAGNOSIS_THRESHOLD=100

# model path
TEXT_MODEL_PATH="models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
IMG_MODEL_PATH16="models/vit-base-patch16-224-in21k"
IMG_MODEL_PATH32="models/vit-base-patch32-224-in21k"
CHECKPOINT="output/GEMINI"

# Train
TRAIN_EPOCHS=30
MASK_INPUT=true
# per gpu batch size
BATCH_SIZE=8

ACCUMULATION_STEPS=1
SCHEDULER_TYPE="linear"
WEIGHT_DECAY=0.01

LEARNING_RATE=5e-5
WARMUP_RATIO=0.1
TEMPERATURE=0.07

EVALUATION_STRATEGY="epoch"
SAVE_STRATEGY="epoch"

# Train
TRAIN_EPOCHS=50
# per gpu batch size
BATCH_SIZE=4

ACCUMULATION_STEPS=1
SCHEDULER_TYPE="linear"
WEIGHT_DECAY=0.01

LEARNING_RATE=1e-3
WARMUP_RATIO=0.1

EVALUATION_STRATEGY="epoch"
SAVE_STRATEGY="epoch"

MASK_INPUT=false

python test.py \
  --do_train \
  --mask_input $MASK_INPUT \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --cxr_dir $CXR_DIR \
  --prepared_data_path $PREPARED_DATA_PATH \
  --train_idx_path $TRAIN_IDX_PATH \
  --val_idx_path $VAL_IDX_PATH \
  --test_idx_path $TEST_IDX_PATH \
  --icd_diagnosis_threshold $ICD_DIAGNOSIS_THRESHOLD \
  --text_model_path $TEXT_MODEL_PATH \
  --img_model_path16 $IMG_MODEL_PATH16 \
  --img_model_path32 $IMG_MODEL_PATH32 \
  --model_path $CHECKPOINT \
  --per_device_train_batch_size $BATCH_SIZE \
  --num_train_epochs $TRAIN_EPOCHS \
  --evaluation_strategy $EVALUATION_STRATEGY \
  --save_strategy $SAVE_STRATEGY \
  --weight_decay $WEIGHT_DECAY \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --lr_scheduler_type $SCHEDULER_TYPE \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --dataloader_num_workers 4 \
  --seed 2023 \
  --fp16=True