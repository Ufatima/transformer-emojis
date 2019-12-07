export DATA_DIR=./data/tweet_emoji_dataset
export TASK_NAME=emoji

python run_train.py \
    --model_type distilbert \
    --model_name_or_path distilbert-base-multilingual-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
