import argparse
import torch
from drmm_tks import DRMM_TKS


parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", type=int, default=5)
parser.add_argument("--vocab_size", type=int, default=10)
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--cuda", type=bool, default=False)
parser.add_argument("--num_layers", type=int, default=3 )
parser.add_argument("--dropout_p", type=float, default=0.2)
parser.add_argument("--hidden_sizes", nargs='+', type=int, default=[2, 2, 1])


args = parser.parse_args()
model = DRMM_TKS(args)

q1 = torch.LongTensor(3, 4) % args.vocab_size
q2 = torch.LongTensor(3, 5) % args.vocab_size
q2_length = torch.LongTensor([2, 3, 4])
q1_length = torch.LongTensor([3, 4, 5])

out = model(q1, q2, q1_length, q2_length)
# out = model.get_attn_mask(q1_length, q2_length, 4, 5)
print(out)