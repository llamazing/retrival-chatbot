import os
import itertools
from collections import Counter


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocab(object):
    def __init__(self, vocab_path, max_num=None):
        if not os.path.exists(vocab_path):
            raise ValueError("Vocab file %s not exists!" %vocab_path)
        self.id2word = [PAD_TOKEN, UNK_TOKEN]
        self.word2id = dict(zip(self.id2word, itertools.count()))
        self.wcount = len(self.id2word)

        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_num is not None and i == max_num:
                    break
                word = line.split('\t\t')[0]
                if word not in self.word2id:
                    self.id2word.append(word)
                    self.word2id[word] = self.wcount
                    self.wcount += 1
                else:
                    raise Exception("Word %s duplicate!" %word)

    def __len__(self):
        return len(self.id2word)

    def word_to_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[UNK_TOKEN]

    def sentence_to_id(self, sent):
        words = sent.split()
        word_ids = [ self.word_to_id(word) for word in words]
        return word_ids

    def id_to_word(self, id):
        if id < len(self.id2word):
            return self.id2word[id]
        else:
            return UNK_TOKEN

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word.append(word)
            self.word2id[word] = self.wcount
            self.wcount += 1
        else:
            raise ValueError("Word %s already in vocab." %word)

    def id_to_sentence(self, sent_id, eos_id, filter_unk=False):
        sent = []
        for i, id in enumerate(sent_id):
            if id == eos_id:
                break
            if filter_unk and id == self.word_to_id(UNK_TOKEN):
                continue
            sent.append(self.id_to_word(id))
        if len(sent) == 0: return ""
        return " ".join(sent)

    @staticmethod
    def build_vocab_from_sentences(sents, out_path, rewrite=False):
        if rewrite and os.path.exists(out_path):
            raise ValueError("vocab already exists! If you want to rewrite set rewirte True." %out_path)
        word_count = Counter()
        for item in sents:
            word_count.update(item.split())
        with open(out_path, 'w', encoding='utf-8') as f:
            i = 0
            for key, val in word_count.most_common():
                if i != 0: f.write('\n')
                f.write(key + '\t\t' + str(val))
                i += 1
        print("Saving %s words to file." %len(word_count))