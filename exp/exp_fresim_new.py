from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix, get_record
from utils.augmentations import masked_data
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.FAT_resi import moving_average

warnings.filterwarnings('ignore')
class Exp_fresim(Exp_Basic):
    def __init__(self, args):
        super(Exp_fresim, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs/{args.data}/{args.model}/{args.pretrain_mode}/{args.exp_name}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            if self.args.trs:
                print("train from scratch")
            else:
              print("Loading ckpt: {}".format(self.args.load_checkpoints))
              model = transfer_weights(self.args.load_checkpoints, model, device=self.device, freeze=self.args.freeze)

        # if torch.cuda.device_count() > 1:
        # #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
        #       model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_classification_loader(self):
        if self.args.task_type == "clf":
            # 加载训练数据集
            pre_train_data = cls_data.DefaultGenerator(
                cls_data.DatasetName.__members__[self.args.data],
                flag='train',
                x_len=178
            )

            # 加载验证数据集
            val_data = cls_data.DefaultGenerator(
                cls_data.DatasetName.__members__[self.args.data],
                flag='val',
                x_len=178
            )

            # 加载验证数据集
            test_data = cls_data.DefaultGenerator(
                cls_data.DatasetName.__members__[self.args.data],
                flag='test',
                x_len=178
            )

            # 创建DataLoader
            train_loader = DataLoader(
                pre_train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )

            val_loader = DataLoader(
                val_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            ) ###
            test_loader = DataLoader(
                test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            )

            return train_loader, val_loader, test_loader

        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _forward_model(self, batch_x, batch_x_mark=None):
        if self.args.model == 'FAT_res_trend_gate' and batch_x_mark is not None:
            return self.model(batch_x, batch_x_mark)
        elif self.args.model == 'FAT_res_trend_gate_new' and getattr(self.args, 'res_pretrain_use_time', 0) == 1 and batch_x_mark is not None:
            return self.model(batch_x, batch_x_mark)
        return self.model(batch_x)

    def _collect_pretrain_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            if 'module.' in k:
                k = k.replace('module.', '')

            if self.args.model == 'FAT_res_trend_gate':
                if 'head' not in k:
                    state_dict[k] = v
            elif 'encoder' in k or 'enc_embedding' in k:
                state_dict[k] = v
        return state_dict

    def pretrain(self):

        if self.args.task_type == "clf":
            train_loader, vali_loader, test_loader = self._get_classification_loader()
        else:
            # data preparation
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data, self.args.exp_name)
        if not os.path.exists(path):
            os.makedirs(path)
        # optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.pretrain_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.pretrain_epochs):
            start_time = time.time()
            if self.args.task_type == "clf":
                train_loss, train_cl_loss, train_rb_loss, train_gate = self.pretrain_one_epoch(train_loader,
                                                                                                 model_optim,
                                                                                                 model_scheduler)
                vali_loss, valid_cl_loss, valid_rb_loss, vali_gate = self.valid_one_epoch(vali_loader)
                test_loss, test_cl_loss, test_rb_loss, test_gate = self.valid_one_epoch(test_loader)

            else:
                train_loss, train_cl_loss, train_rb_loss, train_gate = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
                vali_loss, valid_cl_loss, valid_rb_loss, vali_gate = self.valid_one_epoch(vali_loader)
                test_loss, test_cl_loss, test_rb_loss, test_gate = self.valid_one_epoch(test_loader)

            # log and Loss
            end_time = time.time()

            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f} Test Loss: {9:.4f}/{10:.4f}/{11:.4f} | Gate Keep Score: {12:.4f}/{13:.4f}/{14:.4f}\n"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                        train_rb_loss,
                        vali_loss, valid_cl_loss, valid_rb_loss,  test_loss, test_cl_loss, test_rb_loss,
                        train_gate, vali_gate, test_gate))

            pretrain_txt = path + "/" + "pretrain_loss.txt"
            pretrain_content = "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f} Test Loss: {9:.4f}/{10:.4f}/{11:.4f} | Gate Keep Score: {12:.4f}/{13:.4f}/{14:.4f}\n".format(
                epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                train_rb_loss,
                vali_loss, valid_cl_loss, valid_rb_loss,  test_loss, test_cl_loss, test_rb_loss,
                train_gate, vali_gate, test_gate)
            get_record(pretrain_txt, pretrain_content)

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'train_gate_keep_score': train_gate,
                'vali_loss': vali_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
                'vali_gate_keep_score': vali_gate,
                'test_loss': test_loss,
                'test_cl_loss': test_cl_loss,
                'test_rb_loss': test_rb_loss,
                'test_gate_keep_score': test_gate,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...\n".format(min_vali_loss, vali_loss, epoch))
                save_info = "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...\n".format(min_vali_loss, vali_loss, epoch)
                get_record(pretrain_txt, save_info)

                min_vali_loss = vali_loss
                self.encoder_state_dict = self._collect_pretrain_state_dict()
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best_ptmode_{self.args.pretrain_mode}.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                get_record(pretrain_txt, "Saving model at epoch {}...\n".format(epoch + 1))

                self.encoder_state_dict = self._collect_pretrain_state_dict()
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))


    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []
        train_gate_score = []

        self.model.train()
        for i, (batch_x, batch_y ,*others) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = others[0].float().to(self.device) if len(others) > 0 else None
            loss, loss_cl, loss_rb, gate_score ,_ ,_ = self._forward_model(batch_x, batch_x_mark)

            # backward
            loss.backward()
            model_optim.step()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())
            if gate_score is not None:
                train_gate_score.append(gate_score.item() if hasattr(gate_score, 'item') else gate_score)

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)
        avg_gate_score = np.average(train_gate_score) if len(train_gate_score) > 0 else 0.0

        return train_loss, train_cl_loss, train_rb_loss, avg_gate_score

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []
        valid_gate_score = []

        self.model.eval()
        for i, (batch_x, batch_y, *others) in enumerate(vali_loader):

            batch_x = batch_x.float().to(self.device)
            batch_x_mark = others[0].float().to(self.device) if len(others) > 0 else None
            # encoder
            loss, loss_cl, loss_rb, gate_score, _, _ = self._forward_model(batch_x, batch_x_mark)

            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())
            if gate_score is not None:
                valid_gate_score.append(gate_score.item() if hasattr(gate_score, 'item') else gate_score)

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)
        avg_gate_score = np.average(valid_gate_score) if len(valid_gate_score) > 0 else 0.0

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss, avg_gate_score

    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.freeze == 0:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, *others) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = others[0].float().to(self.device) if len(others) > 0 else None

                # encoder
                outputs = self._forward_model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                trend_y = moving_average(batch_y, kernel_size=self.args.decomp_kernel)
                trend_hat = moving_average(outputs, kernel_size=self.args.decomp_kernel)  # [B, pred_len, N]
                #
                loss_trend = criterion(trend_hat, trend_y)
                #
                # # 施加分布一致性
                # eps = 1e-6
                # mu_y = trend_y.mean(dim=1)  # [B, N]
                # std_y = trend_y.std(dim=1, unbiased=False) + eps  # [B, N]
                # mu_hat = trend_hat.mean(dim=1)  # [B, N]
                # std_hat = trend_hat.std(dim=1, unbiased=False) + eps  # [B, N]
                # loss_mu = criterion(mu_hat, mu_y)
                # loss_std = criterion(std_hat, std_y)
                # loss_var = loss_mu + loss_std

                # loss
                # loss = criterion(outputs, batch_y) + self.args.lambda_trend * loss_trend + self.args.lambda_var * loss_var
                # loss = criterion(outputs, batch_y)
                loss = criterion(outputs, batch_y) + self.args.lambda_trend * loss_trend
                loss.backward()
                model_optim.step()
                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))

            finetune_txt = path + '/' + 'finetune_loss.txt'
            finetune_content = "lr_{0}_dp_{1}_{2}_Epoch: {3}, Steps: {4}, Time: {5:.2f}s | Train Loss: {6:.7f} Vali Loss: {7:.7f} Test Loss: {8:.7f} \n".format(
                self.args.learning_rate, self.args.dropout, self.args.pred_len, epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss)
            get_record(finetune_txt, finetune_content)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.freeze == 0:
               adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, *others) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = others[0].float().to(self.device) if len(others) > 0 else None

                # encoder
                outputs = self._forward_model(batch_x, batch_x_mark)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, *others) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = others[0].float().to(self.device) if len(others) > 0 else None

                # encoder
                outputs = self._forward_model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))

        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)
        text_txt = path + '/' + 'test_loss.txt'
        text_content = 'lr_{0}_dp_{1}_{2}___{3}->{4}, {5:.5f}, {6:.5f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.exp_name, self.args.seq_len, self.args.pred_len, mse, mae)
        get_record(text_txt, text_content)


        if self.args.trs:
            filename = "{}/score_{}_train_from_scratch.txt".format(folder_path,self.args.model,)
        else:
            filename = "{}/score_{}_{}_{}.txt".format(folder_path,self.args.model,self.args.lm,self.args.pretrain_mode)
        f = open(filename,'a')
        f.write('lr_{0}_dp_{1}_{2}->{3}, {4:.3f}, {5:.3f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()

    def show(self, num, epoch, type='valid'):

        # show cases
        if type == 'valid':
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show

        # data augumentation
        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                      self.args.positive_nums)
        batch_x_om = torch.cat([batch_x, batch_x_m], 0)

        # masking matrix
        mask = mask.to(self.device)
        mask_o = torch.ones(size=batch_x.shape).to(self.device)
        mask_om = torch.cat([mask_o, mask], 0).to(self.device)

        # to device
        batch_x = batch_x.float().to(self.device)
        batch_x_om = batch_x_om.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Encoder
        with torch.no_grad():
            loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

        for i in range(num):

            if i >= batch_x.shape[0]:
                continue

        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)
        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)
        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)
        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)
