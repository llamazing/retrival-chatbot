import logging
import torch
import torch.optim as optim
import os
import torch.nn as nn
from utils import logger_set
from Checkpoint import Checkpoint
from Optim import Optimizer
from drmm_tks import Evaluator


class Trainer(object):
    def __init__(self, batch_size, checkpoint_every, print_every, expt_dir, log_file=None, loss="ce",optimizer=None, max_grad_norm=5, pad_id=0):
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        if loss == "ce":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        if torch.cuda.is_available():
            self.loss.cuda()
        self.pad_id = pad_id
        if not os.path.exists(expt_dir):
            os.makedirs(expt_dir)
        self.evaluator = Evaluator()
        self.expt_dir = expt_dir
        self.logger = logging.getLogger(__name__)
        logger_set(self.logger, log_file)

    def _train_batch(self, model, batch):
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
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_epoches(self, train_data, model, n_epoches, start_epoch, start_step, eval_data=None):
        print_loss_total = 0
        epoch_loss_total = 0

        step = start_step
        step_elapse = 0
        steps_per_epoch = len(train_data)
        total_batch = n_epoches * steps_per_epoch
        self.logger.info("Training total steps: %d, One Epoch steps: %d" %(total_batch, steps_per_epoch))
        acc = 0
        for epoch in range(start_epoch, n_epoches + 1):
            self.logger.info("Epoch: %d, Step: %d" %(epoch, step))
            model.train(mode=True)
            for batch in train_data:
                step += 1
                step_elapse += 1
                loss = self._train_batch(model, batch)
                print_loss_total += loss
                epoch_loss_total += loss
                if step % self.print_every == 0 and step_elapse >= self.print_every:
                    avg_print_loss = print_loss_total / self.print_every
                    print_loss_total = 0
                    self.logger.info("Progress: %d%%, Epoch: %d, Train loss: %.4f" %(step / total_batch * 100, epoch, avg_print_loss))

            if step_elapse == 0: continue
            avg_epoch_loss = epoch_loss_total / min(steps_per_epoch, step-start_step)
            epoch_loss_total = 0
            self.logger.info("At epoch: %d, Train loss: %.4f" %(epoch, avg_epoch_loss))
            if eval_data is not None:
                with torch.no_grad():
                    eval_loss, eval_acc = self.evaluator.evaluate(eval_data, model)
                self.optimizer.update(eval_loss)
                model.train(mode=True)
                self.logger.info("At epoch: %d, Eval loss: %.4f, Eval accuracy: %.2f%%" %(epoch, eval_loss, eval_acc*100))
            else:
                eval_acc = 0
                self.optimizer.update(avg_epoch_loss, epoch)
            if eval_acc > acc:
                acc = eval_acc
                Checkpoint(model=model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        step=step
                        ).save(self.expt_dir, step)
            self.logger.info("Finished epoch %d, average loss: %.4f" %(epoch, avg_epoch_loss))

    def train(self, model, train_data, n_epoches, eval_data=None, resume=False):
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            resume_optim = resume_checkpoint.optimizer.optimizer
            defaults = resume_optim.param_grous[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if self.optimizer is None:
                self.optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=self.max_grad_norm)
        self.logger.info("Optimizer: %s, Schedule: %s" %(self.optimizer.optimizer, self.optimizer.scheduler))
        self._train_epoches(train_data, model, n_epoches, start_epoch, step, eval_data)

        return model
