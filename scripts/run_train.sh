#!/bin/bash

DATASET=W00
KFOLD=(0 1 2 3 4 5 6 7 8 9)
GPU=0

#TODO albert-xxlarge-v2
#bert-base-uncased bert-large-uncased
MODELS=(roberta-base) #allenai/scibert_scivocab_uncased) #roberta-base) # roberta-base allenai/scibert_scivocab_uncased) # albert-base-v2 albert-large-v2)
#bert-base-uncased bert-large-uncased roberta-large roberta-base
#allenai/cs_roberta_base) #albert-xxlarge-v2) #
# allenai/longformer-base-4096

#Neoligsm detection
#Term/Def clustering t-SNE.
#Term type classification (Symbol, Term, neologism)
#Length / other features of the terms/definitions detected.


for MODEL in "${MODELS[@]}"
do
    #NOTE for longformer: change max_seq_len from 80 to 512, batch_size from 32 to 8
    if [[ $MODEL = allenai/longformer* ]]
    then
        BATCH=8
        MAXLEN=512
    else
        BATCH=8 #32
        MAXLEN=80
    fi

    for K in "${KFOLD[@]}"
    do
        echo "============================================"
        echo "Training..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_train \
            --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
            --num_train_epochs 30 --logging_steps 100 --save_steps 100 \
            --overwrite_cache --overwrite_output_dir
        echo "============================================"

        echo "============================================"
        echo "Training (POS)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python -W ignore main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
            --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
            --num_train_epochs 30 --logging_steps 100 --save_steps 100 \
            --overwrite_cache --overwrite_output_dir
        echo "============================================"

        echo "============================================"
        echo "Training (POS,NP,VP,Entity,Acronym)..." $MODEL $K ${DATASET} ${BATCH} ${MAXLEN}
        CUDA_VISIBLE_DEVICES=$GPU \
        python -W ignore main.py \
            --model_name_or_path=$MODEL \
            --data_dir $HOME/data/ \
            --output_dir $HOME/data/joint_bert/${DATASET}/${K}/${MODEL} \
            --task ${DATASET} --kfold $K \
            --use_crf \
            --do_train --use_pos --use_np --use_vp --use_entity --use_acronym \
            --use_heuristic \
            --per_device_train_batch_size ${BATCH} --per_device_eval_batch_size ${BATCH} \
            --max_seq_len ${MAXLEN} --learning_rate 2e-5 \
            --num_train_epochs 30 --logging_steps 100 --save_steps 100 \
            --overwrite_cache --overwrite_output_dir
        echo "============================================"


    done

# --use_pos --use_np

done
