#!/bin/bash

DATA=./model_files/
mkdir -p $DATA

# AI2020
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/abbrexp.zip -P $DATA/
mkdir -p $DATA/abbrexp/
unzip $DATA/abbrexp.zip -d $DATA/abbrexp/
rm $DATA/abbrexp.zip

# W00
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/termdef.zip -P $DATA/
mkdir -p $DATA/termdef/
unzip $DATA/termdef.zip -d $DATA/termdef/
rm $DATA/termdef.zip

# DocDef2
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v4/symnick.zip -P $DATA/
mkdir -p $DATA/symnick/
unzip $DATA/symnick.zip -d $DATA/symnick/
rm $DATA/symnick.zip

# DocDef Query format (DocDefQueryInplace)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/symnick_query.zip -P $DATA/
mkdir -p $DATA/symnick_query/
unzip $DATA/symnick_query.zip -d $DATA/symnick_query/
rm $DATA/symnick_query.zip

# DocDef Query format (DocDefQueryInplaceFixeDMIA)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/symnick_query_mia.zip -P $DATA/
mkdir -p $DATA/symnick_query_mia/
unzip $DATA/symnick_query_mia.zip -d $DATA/symnick_query_mia/
rm $DATA/symnick_query_mia.zip

# Joint Model (DocDef2+AI2020+W00)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/joint_symnick_abbrexp_termdef.zip -P $DATA/
mkdir -p $DATA/joint_symnick_abbrexp_termdef/
unzip $DATA/joint_symnick_abbrexp_termdef.zip -d $DATA/joint_symnick_abbrexp_termdef/
rm $DATA/joint_symnick_abbrexp_termdef.zip

# Joint Model (DocDefQueryInplace+AI2020+W00)
wget https://scholarphi4nlp.s3-us-west-2.amazonaws.com/model/v5/joint_symnickquery_abbrexp_termdef.zip -P $DATA/
mkdir -p $DATA/joint_symnickquery_abbrexp_termdef/
unzip $DATA/joint_symnickquery_abbrexp_termdef.zip -d $DATA/joint_symnickquery_abbrexp_termdef/
rm $DATA/joint_symnickquery_abbrexp_termdef.zip


echo "Successfully downloaded model files."


