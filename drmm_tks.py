import torch
import torch.nn as nn
import torch.nn.functional as F


def dot2(a, b, normalize=False):
    mm = torch.bmm(a, b.transpose(2, 1))
    if normalize:
        q1_norm = torch.norm(a, 2, dim=2, keepdim=True)
        q2_norm = torch.norm(b, 2, dim=2, keepdim=True).transpose(1, 2)
        norm = torch.bmm(q1_norm, q2_norm)
        mm = mm / (norm + 1e-8)
    return mm

class DRMM_TKS(nn.Module):
    def __init__(self, config):
        super(DRMM_TKS, self).__init__()
        self.name = "DRMM_TKS"
        self.use_cuda = config.cuda
        self.topk = config.topk
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.gating = nn.Linear(config.emb_dim, 1)
        hidden_list = [config.topk] + config.hidden_sizes
        layers = []
        for i in range(1, config.num_layers+1):
            layers.append(nn.Linear(hidden_list[i-1], hidden_list[i]))
            layers.append(nn.Softplus())
        self.deep_layer = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout_p)
        self.out_layer = nn.Linear(1, 2)

    def get_attn_mask(self, q1_len, q2_len, max_len, extend_size):
        batch_size = q1_len.size(0)
        mask = torch.zeros(batch_size, max_len, extend_size)
        if self.use_cuda:
            mask = mask.cuda()
        for i, l in enumerate(zip(q1_len, q2_len)):  # skip the first sentence
            l1, l2 = l
            if l1 < max_len:
                mask[i, l1:, :] = 1
            if l2 < extend_size:
                mask[i, :, l2:] = 1
        mask = mask.byte()
        return mask

    def forward(self, q1, q2, q1_length, q2_length):
        batch_size, max_q1_len = q1.size()
        q1_embed = self.embedding(q1)
        q2_embed = self.embedding(q2)
        attn_mask = self.get_attn_mask(q1_length, q2_length, max_q1_len, q2.size(1))
        # (batch_size, max_q1_len, max_q2_len)
        mm = dot2(q1_embed, q2_embed, True)
        mm.data.masked_fill_(attn_mask, 0.0)
        w_g = self.gating(q1_embed).squeeze(2)
        # (batch_size, max_q1_len)
        g = F.softmax(w_g, dim=1)
        mmk, _ = torch.topk(mm, self.topk, dim=2)
        mmk = self.deep_layer(mmk).squeeze(2)
        # (batch_size, max_q1_len)
        mmk_drop = self.dropout(mmk)
        mean = torch.sum(mmk_drop * g, dim=1).view(batch_size, 1)
        out = F.softmax(self.out_layer(mean), dim=1)
        return out


class Evaluator(object):
    def __init__(self, loss="ce", batch_size=64, pad_id=0):
        if loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        self.batch_size = batch_size
        self.pad_id = pad_id
        if torch.cuda.is_available():
            self.loss.cuda()

    def evaluate(self, eval_data, model):
        model.eval()
        total_loss = 0
        total_step = 0
        correct = 0
        nums = 0
        for batch in eval_data:
            nums += self.batch_size
            if torch.cuda.is_available():
                q1 = batch.q1.cuda()
                q2 = batch.q2.cuda()
                q1_len = batch.q1_len.cuda()
                q2_len = batch.q2_len.cuda()
                label = batch.label.cuda()
            else:
                q1 = batch.q1
                q2 = batch.q2
                q1_len = batch.q1_len
                q2_len = batch.q2_len
                label = batch.label
            logits = model(q1, q2, q1_len, q2_len)
            loss = self.loss(logits, label)
            _, pred = torch.topk(logits, 1, dim=1)
            correct += torch.sum(torch.eq(label, pred.squeeze(1)).int()).item()
            total_loss += loss.item()
            total_step += self.batch_size 
        return total_loss / total_step, correct / nums


class Predictor(object):
    def __init__(self, batch_size=64, pad_id=0):
        self.batch_size = batch_size
        self.pad_id = pad_id

    def evaluate(self, eval_data, model):
        model.eval()
        all = []
        all_ids = []
        for batch in eval_data:
            if torch.cuda.is_available():
                q1 = batch.q1.cuda()
                q2 = batch.q2.cuda()
                q1_len = batch.q1_len.cuda()
                q2_len = batch.q2_len.cuda()
            else:
                q1 = batch.q1
                q2 = batch.q2
                q1_len = batch.q1_len
                q2_len = batch.q2_len
            logits = model(q1, q2, q1_len, q2_len)
            all.extend(logits[:, 1].cpu().tolist())
            all_ids.extend(batch.id)
        id_max = 0
        v_max = 0
        for id, v in zip(all_ids, all):
            if v > v_max:
                v_max = v
                id_max = id
        return id_max