from typing import Tuple, Optional

import mlflow
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import consts
from util import should_use_cuda
from tqdm import tqdm


class Coach(object):
    def __init__(self, model: nn.Module,
                 optimizer: Optimizer,
                 lr_scheduler: object,
                 criterion: nn.Module,
                 train_dataset: DataLoader,
                 val_dataset: DataLoader,
                 val_freq: int,
                 max_epoch_len: int,
                 label: str,
                 fold: Optional[int] = None,
                 clip_grad_norm: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_epoch_len = max_epoch_len
        self.label = label
        self.clip_grad_norm = clip_grad_norm
        self.val_freq = val_freq
        self.fold = fold

    def train(self, epochs: int) -> Tuple[float, float]:
        loss, acc = self.run_epoch(train=False)
        val_loss = loss
        val_acc = acc
        mlflow.log_metrics({f'{self.label} {self.fold} Val loss': loss, f'{self.label} {self.fold} Val acc at 1': acc}, step=0)
        self.model.train(True)
        for i in range(1, epochs+2):
            # Run a train epoch
            print(f'Epoch={i+1}')
            loss, acc = self.run_epoch(train=True)
            print(f'{self.fold} Train loss={loss}, Train acc at 1={acc}')
            mlflow.log_metrics({f'{self.label} {self.fold} Train loss': loss, f'{self.label} {self.fold} Train acc at 1': acc}, step=i)
            
            # Run a validation epoch
            if i % self.val_freq:
                print(f'{self.fold} Val epoch={i+1}')
                loss, acc = self.run_epoch(train=False)
                print(f'{self.fold} Val loss={loss}, Val acc at 1={acc}')
                mlflow.log_metrics({f'{self.label} {self.fold} Val loss': loss, f'{self.label} {self.fold} Val acc at 1': acc}, step=i)
                val_loss = loss
                val_acc = acc
        return val_loss, val_acc

    def run_epoch(self, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        dataset = self.train_dataset
        if not train:
            dataset = self.val_dataset
        with torch.set_grad_enabled(train):
            epoch_len = min(self.max_epoch_len, len(dataset))
            avg_loss, avg_acc = 0, 0
            data_loader_iter = iter(dataset)
            pbar = tqdm(range(epoch_len))
            for _ in pbar:
                x, subj_idx, y = next(data_loader_iter)
                batch_loss, batch_acc = self.train_batch(x, subj_idx, y)
                pbar.set_description(f'Loss={batch_loss}, Acc={batch_acc}')
                avg_loss += batch_loss / epoch_len
                avg_acc += batch_acc / epoch_len
            return avg_loss, avg_acc

    def train_batch(self, x, subj_idx, y):
        if should_use_cuda():
            for scan in x:
                scan.cuda(non_blocking=True)
            y.cuda(non_blocking=True)
        y_hat = self.model(x, subj_idx)

        # Calculate acc@1
        _, preds = torch.max(y_hat, 1)
        batch_acc = torch.sum(preds == y.data).item() / y.shape[0]

        self.model.zero_grad()
        loss = self.criterion(y_hat, y)
        
        if self.model.training:
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), consts.MAX_GRAD_NORM)
            self.optimizer.step()

        return loss.item(), batch_acc
