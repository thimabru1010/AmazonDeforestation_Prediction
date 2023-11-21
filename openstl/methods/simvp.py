import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method
from segmentation_models_pytorch.losses import FocalLoss


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch, nclasses):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.nclasses = nclasses
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        if nclasses is None:
            self.criterion = nn.MSELoss()
            
        if args.loss == 'ce':
            if args.loss_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(args.loss_weights).to(device), ignore_index=50)
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=50)
        elif args.loss == 'focal':
            self.criterion = FocalLoss("multiclass", gamma=3, ignore_index=50).to(device)
            # self.criterion = FocalLoss("binary", gamma=3, ignore_index=50).to(device)
        
    def _build_model(self, args):
        return SimVP_Model(nclasses=self.nclasses, **args).to(self.device)

    def _predict(self, batch_x, batch_y=None, classify=True, **kwargs):
        """Forward the model"""
        # if classify:
        #     pred_y = self.model(batch_x)
        #     preds = []
        #     for i in range(true.shape[2]):
        #         preds.append(np.argmax(pred[:, :, i:i+true.shape[2]], axis=2))
        #     # pred = np.argmax(pred, axis=2)
        #     pred = np.stack(preds, axis=2).reshape(-1)

        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            #! ALTERADO
            pred_y = self.model(batch_x)
            # print('DEBUG 2')
            # print(pred_y.shape)
            #! As outras camadas são jogadas fora, pois só temos labels para o número de aft_seq_length.
            #! Sendo assim, ao aplicar a função de custo e o backpropagation o modelo irá aprender a gerar o tensor pred_y
            #! Com os valores corretos para o canal self.args.aft_seq_length.
            pred_y = pred_y[:, :self.args.aft_seq_length]
            # print(pred_y.shape)
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m]) #???
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                #! ALTERADO
                # print('DEBUG train_one_epoch')
                # print(pred_y.shape, batch_y.shape)
                B, T, C, H, W = pred_y.shape
                pred_y = pred_y.reshape(B, -1, H, W)
                batch_y = batch_y.reshape(B, -1, H, W)[:, 0]
                # print(pred_y.shape, batch_y.shape)
                # print(pred_y)
                # loss = 0
                # window_size = batch_y.shape[1]
                # for deep_slice in range(self.nclasses):
                    # loss += self.criterion(pred_y[:, deep_slice*window_size:(deep_slice+1)*window_size], batch_y[:, deep_slice].type(torch.cuda.LongTensor))
                # loss /= self.nclasses
                loss = self.criterion(pred_y.contiguous() , batch_y.long().contiguous() )

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
