#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # nic add
echo "$DIR"

# 1. preprocess dataset by the following. It will produce data/sparc_data_removefrom/

python3 "$DIR/preprocess.py" --dataset=sparc --remove_from

# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR

# GLOVE_PATH="/home/lily/rz268/dialog2sql/word_emb/glove.840B.300d.txt" # you need to change this
GLOVE_PATH="$DIR/glove/glove.840B.300d.txt" 
LOGDIR="$DIR/logs/logs_sparc_editsql"


CUDA_VISIBLE_DEVICES=0 python3 "$DIR/run.py" --raw_train_filename="$DIR/data/sparc_data_removefrom/train.pkl" \
          --raw_validation_filename="$DIR/data/sparc_data_removefrom/dev.pkl" \
          --database_schema_filename="$DIR/data/sparc_data_removefrom/tables.json" \
          --embedding_filename="$GLOVE_PATH" \
          --data_directory="$DIR/processed_data_sparc_removefrom_test" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --discourse_level_lstm=1 \
          --use_utterance_attention=1 \
          --use_previous_query=1 \
          --use_query_attention=1 \
          --use_copy_switch=1 \
          --use_schema_encoder=1 \
          --use_schema_attention=1 \
          --use_encoder_attention=1 \
          --use_bert=1 \
          --bert_type_abb=uS \
          --fine_tune_bert=1 \
          --use_schema_self_attention=1 \
          --use_schema_encoder_2=1 \
          --interaction_level=1 \
          --reweight_batch=1 \
          --freeze=1 \
          --evaluate=1 \
          --logdir="$LOGDIR" \
          --evaluate_split="valid" \
          --use_predicted_queries=1 \
          --save_file="$LOGDIR/save_8"

# 3. get evaluation result

python3 "$DIR/postprocess_eval.py" --dataset=sparc --split=dev --pred_file "$LOGDIR/valid_use_predicted_queries_predictions.json" --remove_from



#          --interaction_level="False" \
          # --evaluate_split="test" \ # nic
          # --interaction_level=1 \
#          --data_directory="$DIR/processed_data_sparc_removefrom_test" \               #           --data_directory="$DIR/processed_data_sparc_removefrom" \

#         --save_file="$LOGDIR/save_31_sparc_editsql"
#         --evaluate_split="test" \