#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import precision_score, recall_score

"""
---- Input:
    predict:
        type, numpy.ndarray
        shape, (BS=batch_size, C=class_num, W, H, D), onehot encoding
    target:
        type, numpy.ndarray
        shape, (batch_size, W, H, D)
---- Return
    iou, Intersection over Union
    precision,
    recall
"""
####################################################################################################
#      Channel first, process in CPU with numpy
#  (BS, C, W, H, D) or (BS, C, D, H, W)均可，注意predict和target两者统一即可
####################################################################################################


def get_iou(iou_sum, cnt_class):
    # iou = np.divide(iou_sum, cnt_class)  # what if cnt_class[i]==0, 当测试集中，某些样本类别缺失
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx]/cnt_class[idx] if cnt_class[idx] else 0

    # mean_iou = np.mean(iou, dtype=np.float32)  # what if cnt_class[i]==0
    # mean_iou = np.sum(iou) / np.count_nonzero(cnt_class)
    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])  # 去掉第一类empty
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]   # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
    predict = np.argmax(predict, axis=1)

    correct = (predict == target)  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
                # weight_k[i, n] = weight[target[i, n]]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc


def get_occupancy_calibration(predict, target):
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]   # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(predict, axis=1)  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
    calib_occupied = np.zeros(_C, dtype=np.int32) # calibrate occupancy per class.
    calib_total = np.zeros(_C, dtype=np.int32)

    # Go through all predictions and check the occupancy
    occupied = (target != 0)  # (_bs, 129600)

    # TEST
    inv = (target > _C)
    print(f"Number of classes > {_C}: {np.sum(inv)}.")

    for i in range(_bs):
        for n in range(target.shape[1]):
            label = predict[i,n]
            calib_total[label] = calib_total[label] + 1
            calib_occupied[label] = calib_occupied[label] + occupied[i,n]

    return calib_occupied, calib_total


def get_score_semantic_and_completion(predict, target, nonempty=None):
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    # ---- one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
    predict = np.argmax(predict, axis=1)
    # ---- check empty
    if nonempty is not None:
        predict[nonempty == 0] = 0     # 0 empty
        nonempty = nonempty.reshape(_bs, -1)
    # ---- ignore
    # predict[target == 255] = 0
    # target[target == 255] = 0
    # ---- flatten
    target = target.reshape(_bs, -1)    # (_bs, 129600)
    predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

    cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    tp_sum = np.zeros(_C, dtype=np.int32)  # tp
    fp_sum = np.zeros(_C, dtype=np.int32)  # fp
    fn_sum = np.zeros(_C, dtype=np.int32)  # fn

    acc = 0.0

    for idx in range(_bs):
        y_true = target[idx, :]  # GT
        y_pred = predict[idx, :]
        # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        # y_pred = y_pred[y_true != 255]  # ---- ignore
        # y_true = y_true[y_true != 255]
        # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        if nonempty is not None:
            nonempty_idx = nonempty[idx, :]
            # y_pred = y_pred[nonempty_idx == 1]
            # y_true = y_true[nonempty_idx == 1]
            y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]  # 去掉需ignore的点
            y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            # print('y_true.shape, y_pred.shape', y_true.shape, y_pred.shape)
        acc += accuracy_score(y_true, y_pred)  # pixel accuracy
        for j in range(_C):  # for each class
            tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
            fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
            fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size
            u_j = np.array(np.where(y_true == j)).size
            cnt_class[j] += 1 if u_j else 0
            # iou = 1.0 * tp/(tp+fp+fn) if u_j else 0
            # iou_sum[j] += iou
            iou_sum[j] += 1.0*tp/(tp+fp+fn) if u_j else 0  # iou = tp/(tp+fp+fn)

            tp_sum[j] = tp
            fp_sum[j] = fp
            fn_sum[j] = fn

    acc = acc / _bs
    # return acc, iou_sum, cnt_class
    return acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum


def get_score_completion(predict, target, nonempty=None):  # on both observed and occluded voxels
    """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
    _bs = predict.shape[0]  # batch size
    # _C = predict.shape[1]  # _C = 12
    # ---- one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.
    predict = np.argmax(predict, axis=1)
    # ---- check empty
    if nonempty is not None:
        predict[nonempty == 0] = 0     # 0 empty
        nonempty = nonempty.reshape(_bs, -1)
    # ---- ignore
    predict[target == 255] = 0
    target[target == 255] = 0
    # ---- flatten
    target = target.reshape(_bs, -1)    # (_bs, 129600)
    predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
    # ---- treat all non-empty object class as one category, set them to label 1
    b_pred = np.zeros(predict.shape)
    b_true = np.zeros(target.shape)
    b_pred[predict > 0] = 1
    b_true[target > 0] = 1
    p, r, iou = 0.0, 0.0, 0.0
    for idx in range(_bs):
        y_true = b_true[idx, :]  # GT
        y_pred = b_pred[idx, :]
        if nonempty is not None:
            nonempty_idx = nonempty[idx, :]
            y_true = y_true[nonempty_idx == 1]
            y_pred = y_pred[nonempty_idx == 1]
        # print('From [get_score_completion]: y_true.shape', y_true.shape)
        # ---- Way 2: default pos_label=1, average='binary'
        # _acc = accuracy_score(y_true, y_pred)  # pixel accuracy
        _p, _r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')  # labels=[0, 1] pos_label=1,
        _iou = 1 / (1 / _p + 1 / _r - 1) if _p else 0  # 1/iou = (tp+fp+fn)/tp = (tp+fp)/tp + (tp+fn)/tp - 1
        p += _p
        r += _r
        iou += _iou
        # acc += _acc
        # print('_p, _r, _iou', _p, _r, _iou)
    # acc = 100.0 * acc / _bs
    p = p / _bs
    r = r / _bs
    iou = iou / _bs
    return p, r, iou
