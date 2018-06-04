import torch
import json
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader


BatchTuple = namedtuple("BatchTuple", ["id", "q1", "q2", "q1_len", "q2_len", "label"])

class TrainData(Dataset):
    def __init__(self, fp, vocab, max_q_len=None):
        self.max_q_len = max_q_len
        data = json.load(open(fp))
        self._ids = data['ids']
        self._q1 = [vocab.sentence_to_id(item) for item in data['q1']]
        self._q2 = [vocab.sentence_to_id(item) for item in data['q2']]
        self._label = data["label"]

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        if self.max_q_len is not None:
            return (
                self._ids[index],
                self._q1[index][:self.max_q_len],
                self._q2[index][:self.max_q_len],
                min(len(self._q1[index]), self.max_q_len),
                min(len(self._q2[index]), self.max_q_len),
                self._label[index]
            )
        else:
            return (
                self._ids[index],
                self._q1[index],
                self._q2[index],
                len(self._q1[index]),
                len(self._q2[index]),
                self._label[index],
            )


class InferenceData(Dataset):
    def __init__(self, ids, docs, q, vocab, max_q_len=None):
        if max_q_len is not None:
            q = [vocab.sentence_to_id(q)[:max_q_len]] * len(docs)
            docs = [vocab.sentence_to_id(item)[:max_q_len] for item in docs]
        else:
            q = [vocab.sentence_to_id(q)] * len(docs)
            docs = [vocab.sentence_to_id(item) for item in docs]
        self._q1 = q
        self._q2 = docs
        self._ids = ids

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, index):
        return (
            self._ids[index],
            self._q1[index],
            self._q2[index],
            len(self._q1[index]),
            len(self._q2[index]),
            None
        )


def collect_fn_wrapper(pad_id=0):
    def collect_fn(batch):
        batch.sort(key=lambda k: -k[3])
        max_q1_length = max([item[3] for item in batch])
        max_q2_length = max([item[4] for item in batch])
        q1= np.zeros((len(batch), max_q1_length), dtype=np.int64)
        q2 = np.zeros((len(batch), max_q2_length), dtype=np.int64)
        q1_len = np.zeros(len(batch), dtype=np.int64)
        q2_len = np.zeros(len(batch), dtype=np.int64)
        infer = True
        if batch[0][-1] is not None:
            label = np.zeros(len(batch), dtype=np.int64)
            infer = False
        else:
            label = None
        ids = []
        for i, item in enumerate(batch):
            ids.append(item[0])
            s_len = item[3]
            t_len = item[4]
            q1[i, :s_len] = item[1]
            q2[i, :t_len] = item[2]
            q1_len[i] = s_len
            q2_len[i] = t_len
            if not infer: label[i] = item[5]

        nbatch = BatchTuple(
            ids,
            torch.from_numpy(q1),
            torch.from_numpy(q2),
            torch.from_numpy(q1_len),
            torch.from_numpy(q2_len),
            torch.from_numpy(label) if label is not None else None
        )
        return nbatch

    if pad_id == 0:
        return collect_fn
    else:
        raise ValueError("pad id not zero is not supported now!")


def get_data_loader(dataset, batch_size=2, num_workers=2, shuffle=False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collect_fn_wrapper(),
        pin_memory=False
    )