#!/bin/bash
set -eux

MODEL_NAME="xlm_roberta"
MODEL_PATH="/mnt/hdd/dl_ensae/models/xlm_roberta-finetuned/best_model.pt"
OUTPUT_DIR="/mnt/hdd/dl_ensae/models"

LANGUAGES=('xquad.en' 'xquad.ar' 'xquad.de' 'xquad.zh' 'xquad.vi' 'xquad.es' 'xquad.hi' 'xquad.el' 'xquad.th' 'xquad.tr' 'xquad.ru' 'xquad.ro')

for lang in "${LANGUAGES[@]}"; do \
  python3 main.py \
      --model_name "$MODEL_NAME" \
      --learning_rate 2e-5 \
      --weight_decay 0.0001 \
      --type_lr_scheduler cosine \
      --warmup_ratio 0.1 \
      --save_strategy steps \
      --save_steps 5000 \
      --num_epochs 5 \
      --early_stopping_patience 3 \
      --output_dir "$OUTPUT_DIR" \
      --device cuda \
      --batch_size 12 \
      --max_length 384 \
      --doc_stride 128 \
      --n_best_size 20 \
      --max_answer_length 30 \
      --squad_v2 False \
      --dataset_name xquad \
      --xsquad_subdataset_name "$lang" \
      --eval_only True \
      --path_to_finetuned_model "$MODEL_PATH" \

done
