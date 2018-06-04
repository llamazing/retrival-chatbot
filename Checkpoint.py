from __future__ import print_function
import os
import torch


class Checkpoint(object):

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, optimizer, epoch, step, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir, step):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        # date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME)
        path = self._path

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                   },
                   os.path.join(path, str(step) + "_" + self.TRAINER_STATE_NAME))
        model_name = str(step) + "_" + self.MODEL_NAME
        torch.save(self.model, os.path.join(path, model_name))

        return path

    @classmethod
    def load(cls, path, step):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
            step (int): step
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        lstep = str(step)
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.CHECKPOINT_DIR_NAME, lstep + "_" + cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.CHECKPOINT_DIR_NAME, lstep + "_" + cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.CHECKPOINT_DIR_NAME, lstep + "_" + cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.CHECKPOINT_DIR_NAME, lstep + "_" + cls.MODEL_NAME), map_location=lambda storage, loc: storage)

        # model.flatten_parameters() # make RNN parameters contiguous
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          path=path)

    @classmethod
    def get_latest_step(cls, experiment_path):
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_steps = [int(item.split('_')[0]) for item in os.listdir(checkpoints_path)]
        return max(all_steps)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])