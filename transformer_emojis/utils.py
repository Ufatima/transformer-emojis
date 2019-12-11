# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classification fine-tuning: utilities to work with various datasets """

from __future__ import absolute_import, division, print_function

import csv
import pickle
import logging
import os
import sys
from io import open
import enum

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf

    assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
    _tf_available = True  # pylint: disable=invalid-name
    logger.info("TensorFlow version {} available.".format(tf.__version__))
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name


def is_tf_available():
    return _tf_available


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, input_ids, attention_mask, segment_ids, label_id):
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class EmojiProcessor(DataProcessor):
    """ Processor for the Tweet-Emoji dataset """

    def get_train_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev"
        )

    def get_labels(self):
        labels_path = os.path.join("./emoji_index.txt")
        with open(labels_path, "r") as label_file:
            return [str(i) for i, _ in enumerate(label_file.read().split(","))]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            guid = f"{set_type}-{i}"
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


def get_label_to_emoji_map(labels_path):
    with open(labels_path, "r") as f:
        return {label: emoji.strip() for label, emoji in enumerate(f.read().split(","))}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# def convert_example_to_feature(example_row):
#     # return example_row
#     example, label_map, max_seq_length, tokenizer, output_mode = example_row

#     tokens_a = tokenizer.tokenize(example.text_a)

#     tokens_b = None
#     if example.text_b:
#         tokens_b = tokenizer.tokenize(example.text_b)
#         # Modifies `tokens_a` and `tokens_b` in place so that the total
#         # length is less than the specified length.
#         # Account for [CLS], [SEP], [SEP] with "- 3"
#         _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#     else:
#         # Account for [CLS] and [SEP] with "- 2"
#         if len(tokens_a) > max_seq_length - 2:
#             tokens_a = tokens_a[: (max_seq_length - 2)]

#     tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#     segment_ids = [0] * len(tokens)

#     if tokens_b:
#         tokens += tokens_b + ["[SEP]"]
#         segment_ids += [1] * (len(tokens_b) + 1)

#     input_ids = tokenizer.convert_tokens_to_ids(tokens)

#     # The mask has 1 for real tokens and 0 for padding tokens. Only real
#     # tokens are attended to.
#     input_mask = [1] * len(input_ids)

#     # Zero-pad up to the sequence length.
#     padding = [0] * (max_seq_length - len(input_ids))
#     input_ids += padding
#     input_mask += padding
#     segment_ids += padding

#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length

#     if output_mode == "classification":
#         label_id = label_map[example.label]
#     elif output_mode == "regression":
#         label_id = float(example.label)
#     else:
#         raise KeyError(output_mode)

#     return InputFeatures(
#         input_ids=input_ids,
#         input_mask=input_mask,
#         segment_ids=segment_ids,
#         label_id=label_id,
#     )


def convert_examples_to_features(
    examples,
    tokenizer,
    task,
    data_dir,
    max_length=512,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        task: Current task f.e. "emoji"
        data_dir: Where the training data lies
        max_length: Maximum example length
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    # TODO: Pass processor as an argument instead of creating a new one
    if task is not None:
        processor = processors[task](data_dir)
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                },
                tf.int64,
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "imdb":
        return acc_and_f1(preds, labels)
    elif task_name == "quora":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {"emoji": EmojiProcessor}
output_modes = {"emoji": "classification"}

