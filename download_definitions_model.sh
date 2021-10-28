#!/bin/bash

DATA=./model_files/
mkdir -p $DATA

# DocDef Query format (DocDefQueryInplaceFixeDMIA)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/symnick_query_mia.zip -P $DATA/
mkdir -p $DATA/symnick_query_mia/
unzip $DATA/symnick_query_mia.zip -d $DATA/symnick_query_mia/
rm $DATA/symnick_query_mia.zip

echo "Successfully downloaded model files."


