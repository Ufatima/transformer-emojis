import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
import numpy as np

from transformer_emojis.utils import (
    processors,
    get_label_to_emoji_map,
    InputExample,
    convert_examples_to_features,
)



def texts_to_dataloader(texts):
    examples = [InputExample(guid=0, text_a=text, text_b=None, label="1")]
    features = convert_examples_to_features(examples, tokenizer, TASK, DATA_DIR)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=1)


def inference(dataloader, label_map=get_label_to_emoji_map("./emoji_index.txt")):
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        _, logits = model(**inputs)[:2]
        preds = logits.detach().cpu().numpy()
        yield label_map[np.argmax(preds, axis=1)[0]]


if __name__ == '__main__':

    MODEL_DIR = "./outputs/emoji"
    TASK = "emoji"
    DATA_DIR = "./data/tweet_emoji_dataset_tiny"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = processors[TASK]("./data/tweet_emoji_dataset_tiny")
    label_list = processor.get_labels()
    config = DistilBertConfig.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    model.to(device)

    while True:
        print("Enter text and I'll give you an emoji :)")
        text = input(">>> ")

        dataloader = texts_to_dataloader([text])
        emoji = list(inference(dataloader))[0]
        print(emoji)



