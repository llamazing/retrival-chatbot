import torch
import sys
import json
import jieba
from Vocab import Vocab
from drmm_tks import Predictor
from data_load import InferenceData, get_data_loader
from Checkpoint import Checkpoint

vocab_path = "data/vocab.txt"
qa_path = "data/qa.json"
expt_dir = "save/"
qa_data = json.load(open(qa_path))
vocab = Vocab(vocab_path)
batch_size = 20

q = sys.argv[1]
question = " ".join(jieba.cut(q.strip()))

inf_set = InferenceData(qa_data['ids'], qa_data['questions'], question, vocab)

model = Checkpoint.load(expt_dir, Checkpoint.get_latest_step(expt_dir)).model
model.use_cuda = torch.cuda.is_available()
predict = Predictor(batch_size=batch_size)
test_data = get_data_loader(inf_set, batch_size=batch_size)
loc = predict.evaluate(test_data, model)
print(qa_data['answers'][int(loc)])