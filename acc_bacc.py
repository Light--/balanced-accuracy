# -*- coding: utf-8 -*-
# @Time    : 2020/8/6 14:36
# @Author  : Light--
# @FileName: conf_metric.py
# @Software: PyCharm

"""
Description： use confusion matrix to compute acc, bacc

"""

import torch.nn.parallel
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

class Conf_Metric():
    def __init__(self, classes=None):
        self.classesOriginal = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                       'Big_Lips',
                       'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                       'Double_Chin',
                       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                       'Mouth_Slightly_Open',
                       'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                       'Receding_Hairline',
                       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                       'Wearing_Hat',
                       'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.classes = classes if classes else self.classesOriginal

        self.classNum = len(self.classes)
        assert self.classNum == 40 # comment this for other DL tasks instead of the 40-attribute facial attribute estimaion

        self.correct_samples = 0
        self.tns, self.tps, self.fns, self.fps, self.nns, self.nps = [0] * self.classNum, [0] * self.classNum, \
                                [0] * self.classNum, [0] * self.classNum, [0] * self.classNum, [0] * self.classNum
        self.batch_size = 0
        self.samples_num = 0
        self.samples_acc = 0.0
        self.labels_acc = 0.0
        self.labels_bacc = 0.0

    def cal_acc(self, batch_pred, batch_target):
        # num of correct samples / num of all samples
        if self.batch_size==0:
            self.batch_size = batch_target.size(0)
        else:
            try:
                assert batch_target.size(0) == self.batch_size
            except:
                # samples num cannot be divided by batch_size with no remainder
                import traceback
                traceback.print_exc()
                self.batch_size = batch_target.size(0)
        self.samples_num += self.batch_size
        # torch.equal: If both are identical (Exactly the same), return True
        self.correct_samples += sum([torch.equal(batch_pred[i], batch_target[i]) for i in range(self.batch_size)])
        self.samples_acc = 100 * self.correct_samples / self.samples_num
        # print('\ncorrect samples:', self.correct_samples, 'samples num:', self.samples_num)
        return self.samples_acc

    def list_acc_bacc(self):
        # return the acc & bacc list of all labels (classes, attributes)
        tps, Nps, tns, Nns = self.tps, self.nps, self.tns, self.nns
        # print('\ntps[:10]:', tps[:10])
        # print('tns[:10]:', tns[:10])
        taskNum = len(tns)
        assert taskNum == len(tps) == len(Nns) == len(Nps)

        precision = [tps[i] / Nps[i] if Nps[i] != 0 else 0 for i in range(taskNum)]  # tp/Np
        tnNn = [tns[i] / Nns[i] if Nns[i] != 0 else 0 for i in range(taskNum)]  # tn/Nn
        self.labels_bacc = [100 * 0.5 * (precision[i] + tnNn[i]) for i in range(taskNum)]  # 1/2 * (tp/Np + tn/Nn)
        tp_tn = [tps[i] + tns[i] for i in range(taskNum)]  # tp + tn
        Np_Nn = [Nps[i] + Nns[i] for i in range(taskNum)]  # Np + Nn
        self.labels_acc = [100 * (tp_tn[i] / Np_Nn[i]) for i in range(taskNum)]
        return self.labels_acc, self.labels_bacc

    def cal_batch(self, pred, target):
        assert type(pred) == type(target), 'input should be same type'
        assert pred.dtype == target.dtype, 'input should have same data type'
        pred, target = pred.cpu(), target.cpu()
        # print('\npred0', pred[0])
        # print('target0', target[0])
        from sklearn.metrics import multilabel_confusion_matrix
        cm = multilabel_confusion_matrix(target, pred)
        matrixNum, rowsNum, colsNum = cm.shape
        assert matrixNum == self.classNum and 2 == rowsNum == colsNum
        for idx, label in enumerate(self.classes):
            cmi = cm[idx]
            [[tn, fp], [fn, tp]] = cmi
            self.tns[idx] += tn
            self.tps[idx] += tp
            self.fns[idx] += fn
            self.fps[idx] += fp
            self.nns[idx] += (tn + fn)
            self.nps[idx] += (tp + fp)
        acc_list, bacc_list = self.list_acc_bacc()
        # print('\nacc list:', acc_list)
        # print('bacc list:', bacc_list)
        avg_acc, avg_bacc = sum(acc_list) / self.classNum, sum(bacc_list) / self.classNum
        # print(pred, target)
        acc = self.cal_acc(pred, target)
        # print('Batch:\tSamples (Acc. %.5f%%)\tLabels (Avg. Acc.  %.5f%%, Avg. BAcc.  %.5f%%)' % (acc, avg_acc, avg_bacc,))
        # print('Batch:\tAcc. %s\tBAcc. %s' % (acc_list, bacc_list,))
        return acc, avg_acc, avg_bacc
