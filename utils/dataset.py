from collections import OrderedDict
from utils import constant
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.constant import DEPREL_TO_ID


class FeaturizedDataset(Dataset):
    def __init__(self, examples, opt=None, tokenizer=None, cached_features=False):
        super().__init__()
        if cached_features:
            self.data = examples
        else:
            self.opt = opt
            self.max_length = opt.max_length
            self.max_target_length = opt.max_target_length
            self.data = self.convert_examples_to_features(examples, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def convert_examples_to_features(self, examples, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
        """
        data = []

        for (ex_index, example) in enumerate(examples):

            input_ids = torch.tensor(tokenizer.encode(example.text_a, padding='max_length',
                                                        truncation='longest_first', max_length=self.max_length))

            input_mask = torch.tensor(tokenizer.encode_plus(example.text_a, padding='max_length',
                                                             truncation='longest_first',
                                                             max_length=self.max_length)["attention_mask"])

            output_seqs = torch.tensor(tokenizer.encode(example.text_b, padding='max_length',
                                                        truncation='longest_first', max_length=self.max_target_length))

            target_mask = torch.tensor(tokenizer.encode_plus(example.text_b, padding='max_length',
                                                             truncation='longest_first',
                                                             max_length=self.max_target_length)["attention_mask"])

            data.append((input_ids,
                         input_mask,
                         output_seqs,
                         target_mask))

        return data