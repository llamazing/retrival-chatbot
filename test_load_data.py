import json
from Vocab import Vocab
from data_load import TrainData, InferenceData, get_data_loader




vocab = Vocab("data/vocab.txt")
train_set = TrainData("data/dev.json", vocab)
qa_data = json.load(open("data/qa.json"))
inf_set = InferenceData(qa_data['ids'], qa_data['questions'], qa_data['questions'][0], vocab)

train = get_data_loader(train_set)
dev = get_data_loader(inf_set)

for i, t in enumerate(train):
    if i == 3: break
    print(t)
for i, d in enumerate(dev):
    if i == 3: break
    print(d)
