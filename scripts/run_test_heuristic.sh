#!/bin/bash

DATASET=W00_v3
KFOLD=(0 1 2 3 4 5 6 7 8 9)
GPU=

#bert-base-uncased bert-large-uncased roberta-large
MODELS=(roberta-base) # allenai/scibert_scivocab_uncased albert-base-v2 albert-large-v2)


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

    for K in "${KFOLD[@]}"
    do
        echo "============================================"
        echo "Testing..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_eval --use_heuristic \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --overwrite_cache
        echo "============================================"

        echo "============================================"
        echo "Testing (POS)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_eval --use_heuristic --use_pos \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --overwrite_cache
        echo "============================================"

        echo "============================================"
        echo "Testing (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_eval --use_heuristic --use_pos --use_np --use_vp --use_entity --use_acronym \
            --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --overwrite_cache
        echo "============================================"

    done

done
