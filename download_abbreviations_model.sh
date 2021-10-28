#!/bin/bash

DATA=./model_files/
mkdir -p $DATA

# Joint Model (DocDef2+AI2020+W00)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/joint_symnick_abbrexp_termdef.zip -P $DATA/
mkdir -p $DATA/joint_symnick_abbrexp_termdef/
unzip $DATA/joint_symnick_abbrexp_termdef.zip -d $DATA/joint_symnick_abbrexp_termdef/
rm $DATA/joint_symnick_abbrexp_termdef.zip


echo "Successfully downloaded model files."


