# transformer-emojis
Predicting emojis with transformers

## Installation
### With pip
```bash
$ pip install -r requirements.
```
### With conda
```bash
$ conda env create -f environment.yml
```

## Dataset creation
To train a model, you first need to get the training data. You can do so by utilizing the `create_dataset.py` script.

First, you need to get your [twitter app credentials](https://developer.twitter.com/en/apps) and save them to a json file like the following:
```json
{
    "consumer_key": "<put consumer key here>",
    "consumer_secret": "<cosumer secret here>",
    "access_token_key": "<acces token key belongs here>",
    "access_token_secret": "<and access token secret should go here>"
}
```

Then, run the following command to collect the data:
```bash
$ python create_dataset.py --out_dir=./data/emoji_dataset/ --langs=en --secrets=./path/to/the/secrets.json
```
The collection of the data might take some time especially if you select a language that doesn't get much tweets.

Once you think you have collected enough data you can stop the collection script with `CTRL+C` and run the following command to split the data into train, dev, and test sets:
```bash
$ python split_dataset.py --input=./data/emoji_dataset/en/all.tsv --output=./data/emoji_dataset/en --dev_size=1000 --test_size=1000 --random_seed=42
```

Make sure that the dev and test size reflect the size of your collected data. The rest of the dataset is allocated to the train set.

That's it! You now have a dataset to train your emoji predictor with 🥳

## Running training
Now to the exiting part: training the model. Once you have your dataset ready, you simply run the following command to train the model:
```bash
export DATA_DIR=./data/emoji_dataset/en/ && \
export TASK_NAME=emoji && \
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
```

Or change the params to you your liking.

## Running in Colab
You refer to the `/notebooks/train_colab.ipynb` to train the model using Google Colab and

## Acknowledgements
I'd like to thank the Huggingface team for building the [Transformers](https://github.com/huggingface/transformers) library and [nikhilno1](https://github.com/nikhilno1) for [this](https://medium.com/@nikhil.utane/running-pytorch-transformers-on-custom-datasets-717fd9e10fe2) article for helping me use custom data with the Transformers.