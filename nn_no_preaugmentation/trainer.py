import torch as t
import os
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 checkpoint_folder="checkpoints",
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._checkpoint_folder = checkpoint_folder
        self.predictions = []
        self.labels = []
        self.train_losses = []
        self.validation_losses = []
        self.current_epoch = 0
        self.best_validation_loss = np.finfo(np.float32).max

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        isExist = os.path.exists(self._checkpoint_folder)
        if not isExist:
            try:
                os.mkdir(self._checkpoint_folder)
            except FileExistsError:
                print("Directory ", self._checkpoint_folder, " already exists")
        t.save({'state_dict': self._model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': self._optim.state_dict(),
                'train_loss': self.train_losses,
                'validation_loss': self.validation_losses},
               '{}/checkpoint_{:03d}.ckp'.format(self._checkpoint_folder, epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('{}/checkpoint_{:03d}.ckp'.format(self._checkpoint_folder, epoch_n),
                     'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        self.current_epoch = epoch_n + 1
        if "optimizer_state_dict" in ckp:
            self._optim.load_state_dict(ckp['optimizer_state_dict'])
        if "train_loss" in ckp:
            self.train_losses = ckp['train_loss']
        if "validation_loss" in ckp:
            self.validation_losses = ckp['validation_loss']

    def train_step(self, x, y):
        # y = y.to(t.long)########################################################################################################
        self._optim.zero_grad()
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        loss.backward()
        self._optim.step()
        return loss

    def val_test_step(self, x, y):
        # y = y.to(t.long)#########################################################################################################
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        return loss

    def train_epoch(self):
        train_loss = []
        size = len(self._train_dl.dataset)
        self._model.train(mode=True)
        for batch, (data, labels) in enumerate(self._train_dl):
            # Transfer Data to GPU if available
            if t.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            loss = self.train_step(data, labels)
            train_loss.append(loss.item())
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data)
                print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        mean_train_loss = sum(train_loss) / len(train_loss)
        return mean_train_loss

    def val_test(self):
        valid_loss = []
        self._model.train(mode=False)
        size = len(self._val_test_dl.dataset)
        for batch, (data, labels) in enumerate(self._val_test_dl):
            if t.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            loss = self.val_test_step(data, labels)
            valid_loss.append(loss.item())
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data)
                print(f"Validation loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # pred.append(prediction)
            # lab.append(labels)
            # print("Loss:", loss)
        # self.predictions.append(pred)
        # self.labels.append(lab)
        mean_loss = sum(valid_loss) / len(valid_loss)
        return mean_loss

    def test_model(self, input_batch):
        self._model.eval()
        pred = self._model(input_batch)
        return pred

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch

        non_decreasing_epochs = 0
        is_exiting = False

        while True and not is_exiting:
            print("Current epoch = ", self.current_epoch)
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            validation_loss = self.val_test()
            # append the losses to the respective lists
            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)
            print("Average train loss in epoch:", train_loss)
            print("Average validation loss in epoch: ", validation_loss)

            if validation_loss <= self.best_validation_loss:
                self.save_checkpoint(self.current_epoch)
                self.best_validation_loss = validation_loss
                non_decreasing_epochs = 0
            else:
                non_decreasing_epochs += 1
                if non_decreasing_epochs == self._early_stopping_patience:
                    is_exiting = True

            self.current_epoch += 1
            # stop by epoch number
            if self.current_epoch >= epochs:
                is_exiting = True
        return self.train_losses, self.validation_losses
