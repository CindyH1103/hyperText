import torch
from scipy import stats
import numpy as np
import models
import data_loader


class HyperIQASolver_All(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_hyper = models.HyperNet_All(16, 112, 672, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, config.hyper_text, config.hyper_all, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, config.hyper_text, config.hyper_all,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc_align = 0.0
        best_plcc_align = 0.0
        best_srcc_qua = 0.0
        best_plcc_qua = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, text_feature, label1, label2 in self.train_data:
                img = (img.cuda()).clone().detach()
                text_feature = (text_feature.cuda()).clone().detach()
                label1 = (label1.cuda()).clone().detach()
                label2 = (label2.cuda()).clone().detach()
                label = (torch.stack([label1, label2], dim=1).cuda()).cuda().detach()

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                # pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred = model_target(torch.cat([text_feature, paras['target_in_vec']], dim=1))
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc_qua, test_plcc_qua, test_srcc_align, test_plcc_align = self.test(self.test_data)
            if test_srcc_align > best_srcc_align:
                best_srcc_align = test_srcc_align
                best_plcc_align = test_plcc_align
            if test_srcc_qua > best_srcc_qua:
                best_srcc_qua = test_srcc_qua
                best_plcc_qua = test_plcc_qua
            print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, train_srcc[2][3], train_srcc[0][1], test_srcc_align, test_srcc_qua, test_plcc_align,
                   test_plcc_qua))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores_align = []
        gt_scores_align = []
        pred_scores_qua = []
        gt_scores_qua = []

        for img, text_feature, label1, label2 in data:
            img = (img.cuda()).clone().detach()
            text_feature = (text_feature.cuda()).clone().detach()
            label1 = (label1.cuda()).clone().detach()
            label2 = (label2.cuda()).clone().detach()

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            # pred = model_target(paras['target_in_vec'])
            pred = model_target(torch.cat([text_feature, paras['target_in_vec']], dim=1))

            pred_scores_align.append(pred[0].cpu().tolist())
            pred_scores_qua.append(pred[1].cpu().tolist())
            gt_scores_align.append(label1.cpu().tolist()[0])
            gt_scores_qua.append(label2.cpu().tolist()[0])

        test_srcc_align, _ = stats.spearmanr(pred_scores_align, gt_scores_align)
        test_srcc_qua, _ = stats.spearmanr(pred_scores_qua, gt_scores_qua)
        test_plcc_align, _ = stats.pearsonr(pred_scores_align, gt_scores_align)
        test_plcc_qua, _ = stats.pearsonr(pred_scores_qua, gt_scores_qua)

        self.model_hyper.train(True)
        return test_srcc_qua, test_plcc_qua, test_srcc_align, test_plcc_align


class HyperIQASolver_All_2023(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_hyper = models.HyperNet_All_2023(16, 112, 672, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, config.hyper_text, config.hyper_all, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, config.hyper_text, config.hyper_all,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc_align = 0.0
        best_plcc_align = 0.0
        best_srcc_qua = 0.0
        best_plcc_qua = 0.0
        best_srcc_auth = 0.0
        best_plcc_auth = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, text_feature, label1, label2, label3 in self.train_data:
                img = (img.cuda()).clone().detach()
                text_feature = (text_feature.cuda()).clone().detach()
                # label1 = (label1.cuda()).clone().detach()
                # label2 = (label2.cuda()).clone().detach()
                label = (torch.stack([label1, label2, label3], dim=1).cuda()).cuda().detach()

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                # pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred = model_target(torch.cat([paras['target_in_vec'], text_feature], dim=1))
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc_qua, test_plcc_qua, test_srcc_auth, test_plcc_auth, test_srcc_align, test_plcc_align = self.test(self.test_data)
            if test_srcc_align > best_srcc_align:
                best_srcc_align = test_srcc_align
                best_plcc_align = test_plcc_align
            if test_srcc_qua > best_srcc_qua:
                best_srcc_qua = test_srcc_qua
                best_plcc_qua = test_plcc_qua
            if test_srcc_auth > best_srcc_auth:
                best_srcc_auth = test_srcc_auth
                best_plcc_auth = best_plcc_auth
            print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, train_srcc[2][3], train_srcc[0][1], test_srcc_align, test_srcc_auth, test_srcc_qua, test_plcc_qua, test_plcc_auth, test_plcc_align))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores_align = []
        gt_scores_align = []
        pred_scores_qua = []
        gt_scores_qua = []
        pred_scores_auth = []
        gt_scores_auth = []

        for img, text_feature, label1, label2, label3 in data:
            img = (img.cuda()).clone().detach()
            text_feature = (text_feature.cuda()).clone().detach()
            label1 = (label1.cuda()).clone().detach()
            label2 = (label2.cuda()).clone().detach()
            label3 = (label3.cuda()).clone().detach()

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(torch.cat([paras['target_in_vec'], text_feature], dim=1))

            pred_scores_qua.append(pred[0].cpu().tolist())
            pred_scores_auth.append(pred[1].cpu().tolist())
            pred_scores_align.append(pred[2].cpu().tolist())
            gt_scores_align.append(label1.cpu().tolist()[0])
            gt_scores_qua.append(label2.cpu().tolist()[0])
            gt_scores_auth.append(label3.cpu().tolist()[0])

        test_srcc_align, _ = stats.spearmanr(pred_scores_align, gt_scores_align)
        test_srcc_qua, _ = stats.spearmanr(pred_scores_qua, gt_scores_qua)
        test_srcc_auth, _ = stats.spearmanr(pred_scores_auth, gt_scores_auth)
        test_plcc_align, _ = stats.pearsonr(pred_scores_align, gt_scores_align)
        test_plcc_qua, _ = stats.pearsonr(pred_scores_qua, gt_scores_qua)
        test_plcc_auth, _ = stats.pearsonr(pred_scores_auth, gt_scores_auth)

        self.model_hyper.train(True)
        return test_srcc_qua, test_plcc_qua, test_srcc_auth, test_plcc_auth, test_srcc_align, test_plcc_align