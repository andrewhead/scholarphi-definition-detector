#!/bin/bash

DATASET=W00
EVALDATASET=s2orc # DocDef #W00 #WFM
DATA_DIR=$HOME/data/ScholarPhi/
KFOLD=(0 1 2 3 4 5 6 7 8 9)
GPU=0

#bert-base-uncased bert-large-uncased roberta-large roberta-base) #
MODELS=(roberta-large) #allenai/cs_roberta_base) #albert-large-v2) #allenai/scibert_scivocab_uncased) # albert-base-v2

# NOTE: new features for v2.0 system
# --do_ensemble         for model ensembling
# --get_confidence      getting confidence scores of model prediction

for MODEL in "${MODELS[@]}"
do
    #NOTE for longformer: change max_seq_len from 80 to 512, batch_size from 32 to 8
    if [[ $MODEL = allenai/longformer* ]]
    then
        BATCH=8
        MAXLEN=512
    else
        BATCH=32
        MAXLEN=80
    fi

    #TODO replace latex to SYMBOL
    #TODO scale up to 1M papers.
    #TODO featurize 1M papers.

    for K in "${KFOLD[@]}"
    do
        echo "============================================"
        echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python -W ignore main.py \
            --model_name_or_path=$MODEL \
            --data_dir ${DATA_DIR} \
            --output_dir ${DATA_DIR}/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${EVALDATASET} \
            --eval_data_file "features_s2orc_external_cs_limit=10000.json" \
            --kfold $K \
            --use_crf \
            --do_eval \
            --use_nickname_detector --use_acronym_detector \
            --use_heuristic --use_pos --use_np --use_vp --use_entity --use_acronym \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --overwrite_cache \
            --data_limit -1
        echo "============================================"

        exit
    done

done
