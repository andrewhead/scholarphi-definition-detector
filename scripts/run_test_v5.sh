#!/bin/bash

DATA_DIR=$HOME/data/ScholarPhi/v4/data/
MODEL_DIR=$HOME/data/ScholarPhi/v5/model/
OUTPUT_DIR=$HOME/data/ScholarPhi/v5/output/
GPU=0
VERSION=v5
DATASETS=(W00+AI2020+DocDefQueryInplaceFixedMIA) #W00+AI2020+DocDef2) #QueryInplaceFixedMIA) #DocDefQueryInplaceFixedMIA DocDefQueryInplaceFixed DocDefQueryInplace DocDef2)
MODELS=(roberta-large)


for DATASET in "${DATASETS[@]}"
do
for MODEL in "${MODELS[@]}"
do
    #NOTE for longformer: change max_seq_len from 80 to 512, batch_size from 32 to 8
    if [[ $MODEL = allenai/longformer* ]]
    then
        BATCH=8
        MAXLEN=512
    else
        BATCH=12 #8 #32
        MAXLEN=100
    fi

    echo "============================================"
    echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
    #-m torch.distributed.launch
    CUDA_VISIBLE_DEVICES=$GPU \
    python -W ignore main.py \
        --model_name_or_path=$MODEL \
        --data_dir ${DATA_DIR} \
        --output_dir ${MODEL_DIR}/${VERSION}/${DATASET}/MAXLEN=${MAXLEN}/${MODEL} \
        --prediction_dir ${OUTPUT_DIR}/predictions/${VERSION}/${DATASET}/MAXLEN=${MAXLEN}/${MODEL} \
        --result_dir ${DATA_DIR}/results/${VERSION}/${DATASET}/MAXLEN=${MAXLEN}/${MODEL} \
        --task ${DATASET} --dataset ${DATASET} \
        --use_crf \
        --use_heuristic \
        --do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
        --per_device_eval_batch_size ${BATCH} \
        --max_seq_len ${MAXLEN}  \
        --overwrite_cache \
        --joint_learning
    echo "============================================"
    #exit
done
done
