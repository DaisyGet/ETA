from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import torch
import torch.nn as nn


class FeaturizedDataLoader(DataLoader):
    def __init__(self, dataset, opt, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        self.eval = eval
        self.opt = opt
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch = list(zip(*batch_data))
        tensorized = OrderedDict()
        tensorized['input_ids'] = torch.LongTensor([list(i) for i in batch[0]])
        tensorized['attention_mask'] = torch.LongTensor([list(i) for i in batch[1]])
        tensorized['labels'] = torch.LongTensor([list(i) for i in batch[2]])
        tensorized['decoder_attention_mask'] = torch.LongTensor([list(i) for i in batch[3]])

        return tensorized

