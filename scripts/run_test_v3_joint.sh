#!/bin/bash

#DATASET=W00+WFM
#DATASET=DocDef2 #AI2020 #W00+WFM
DATASETS=(DocDef2+AI2020+W00) # DocDef2+W00 W00+AI2020 DocDef2+AI2020+W00)
DATA_DIR=$HOME/data/ScholarPhi/
GPU=1
VERSION=v3.2
MODELS=(roberta-large) # bert-large-uncased allenai/scibert_scivocab_uncased) #roberta-large_finetuned_s2orc10K)

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
        BATCH=16 #8 #32
        MAXLEN=80
    fi

    K=0
    echo "============================================"
    echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
    #-m torch.distributed.launch
    CUDA_VISIBLE_DEVICES=$GPU \
    python -W ignore main.py \
        --model_name_or_path=$MODEL \
        --data_dir ${DATA_DIR} \
		--output_dir ${DATA_DIR}/models/${VERSION}/${DATASET}/${K}/${MODEL} \
		--prediction_dir ${DATA_DIR}/predictions/${VERSION}/${DATASET}/${K}/${MODEL} \
		--result_dir ${DATA_DIR}/results/${VERSION}/${DATASET}/${K}/${MODEL} \
        --task ${DATASET} \
        --use_crf \
        --use_heuristic \
		--do_eval --use_pos --use_np --use_vp --use_entity --use_acronym \
        --per_device_eval_batch_size ${BATCH} \
        --max_seq_len ${MAXLEN} \
        --use_test_set_for_validation \
        --joint_learning #--overwrite_cache
#            --use_nickname_detector --data_limit -1 \
#            --merge_predictions_for_symbol union

    echo "============================================"
    #exit
done
done
