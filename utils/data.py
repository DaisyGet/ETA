import json
import csv
import os
import string
from collections import Counter, defaultdict
from utils import constant


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b):
        self.text_a = text_a
        self.text_b = text_b


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
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
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data


class ACOSProcessor(DataProcessor):
    def __init__(self):
        super(ACOSProcessor, self).__init__()
        self.class_weight = Counter()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json").replace('\\', '/')), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json").replace('\\', '/')), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json").replace('\\', '/')), "test")

    def get_labels(self):
        """See base class."""
        pass

    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line['sentence']
            text_b = line['structured']

            examples.append(InputExample(text_a=text_a, text_b=text_b))
        return examples


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids