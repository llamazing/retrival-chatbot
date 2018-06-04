import argparse
import os
import torch
from drmm_tks import DRMM_TKS
from data_load import get_data_loader, TrainData
from Vocab import Vocab
from supervised_trainer import Trainer
from Optim import Optimizer
from Checkpoint import Checkpoint


def add_parameters(parser: argparse.ArgumentParser):
    parser.add_argument("data_dir", type=str, help="data dir.")
    parser.add_argument("expt_dir", type=str, help="experiment dir.")
    parser.add_argument("--log_file", type=str, default=None, help="log_file")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="initial learning rate.")
    parser.add_argument("--epoches", type=int, default=10, help="training epoches.")
    parser.add_argument("--print_every", type=int, default=100, help="print loss every step.")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="save model every step.")
    parser.add_argument("--opt", type=str, default="Adam", help="training method.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="max gradient norm.")
    parser.add_argument("--mode", type=str, default="train", help="mode: train|test.")
    parser.add_argument("--num_workers", type=int, default=2, help="read data workers.")
    parser.add_argument("--resume", type=bool, default=False, help="resotre model to train.")
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--hidden_sizes", nargs='+', type=int, default=[50, 20, 1])


def train(args):
    train_path = os.path.join(args.data_dir, "train.json")
    dev_path = os.path.join(args.data_dir, "dev.json")
    vocab_path = os.path.join(args.data_dir, "vocab.txt")

    vocab = Vocab(vocab_path)

    vocab_size = len(vocab)
    args.vocab_size = vocab_size
    args.cuda = torch.cuda.is_available()
    train_set = TrainData(train_path, vocab)
    dev_set = TrainData(dev_path, vocab)
    train_iter = get_data_loader(train_set,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True)
    dev_iter = get_data_loader(dev_set,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False)
    model = DRMM_TKS(args)
    if torch.cuda.is_available():
        model.cuda()

    if not args.resume:
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)
    optim = Optimizer(args.opt, model.parameters(), args.learning_rate, max_grad_norm=args.max_grad_norm)
    trainer = Trainer(args.batch_size, args.checkpoint_every, args.print_every, args.expt_dir, args.log_file,
                      optimizer=optim, max_grad_norm=args.max_grad_norm)

    trainer.train(model, train_iter, args.epoches, eval_data=dev_iter, resume=args.resume)


def test(args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sequence to Sequence client.")
    add_parameters(parser)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        raise NotImplementedError
    else:
        raise NotImplementedError
