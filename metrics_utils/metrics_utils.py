##### EDITING THIS VERSION ADD TO CVPR CLEAN ON DESKTOP ####

from __future__ import division, print_function
import numpy as np
import cv2 as cv
import glob
import os
import pdb
import torch
import os
import datetime
import shutil
import tensorflow as tf

import math
from math import sqrt

import scipy.misc
from scipy.ndimage import distance_transform_edt

import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix

from skimage.morphology import binary_dilation, disk


major = cv.__version__.split('.')[0]     # Get opencv version
bDebug = False






#calculate Dice hard
#calculate IoU
'''
Note: From our CVPR code (helper_func.py)

@pram y_pred   numpy array
@param y_true  numpy array

@return: hard dice for this instance
'''
def dice_hard(y_pred, y_true):
    import pdb
    # pdb.set_trace()
    f = f1_score(y_true, y_pred, labels=None, average='micro', sample_weight=None)
    return f


#calculate soft dice
'''
Note: From our CVPR code (helper_func.py)

@pram y_pred   numpy array
@param y_true  numpy array

@return: soft dice for this instance
'''
def dice_soft(y_pred, y_true, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


#calculate IoU
'''
Note: From Darnet code (data_utils.py)

@pram y_pred   numpy array
@param y_true  numpy array

@return: IoU for this instance
'''
def compute_iou(y_pred, y_true):
    intersection = np.count_nonzero(
        np.logical_and(y_pred, y_true)
    )
    union = np.count_nonzero(
        np.logical_or(y_pred, y_true)
    )

    return intersection, union, intersection / union

#calculate Bound F
'''
Note: helper to the bfscore fuction
From: https://github.com/minar09/bfscore_python
'''
def calc_precision_recall(contours_a, contours_b, threshold):

    top_count = 0

    try:
        for b in range(len(contours_b)):

            # find the nearest distance
            for a in range(len(contours_a)):
                dist = (contours_a[a][0] - contours_b[b][0]) * \
                    (contours_a[a][0] - contours_b[b][0])
                dist = dist + \
                    (contours_a[a][1] - contours_b[b][1]) * \
                    (contours_a[a][1] - contours_b[b][1])
                if dist < threshold*threshold:
                    top_count = top_count + 1
                    break

        precision_recall = top_count/len(contours_b)
    except Exception as e:
        precision_recall = 0

    return precision_recall, top_count, len(contours_b)


"""
    NOTE: calc_boundF_darNet helper

    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
"""
def db_eval_boundary(foreground_mask, gt_mask, bound_th=2):
    
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F, precision, recall, np.sum(fg_match), n_fg, np.sum(gt_match), n_gt

"""
    NOTE: calc_boundF_darNet helper

    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
"""
def seg2bmap(seg,width=None,height=None):
    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap


#Compute boundary F score
'''
Note: from https://github.com/minar09/bfscore_python
computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation

@param predict_mask  numpy array
@param gt_mask  numpy array

@return  boundF score per sample averaged over a 1-5 pixel threshold

'''
def calc_boundF_darNet(predict_mask, gt_mask):
    running_intersection = 0
    running_union = 0
    example_iou = 0

    f_bound_n_fg = [0] * 5
    f_bound_fg_match = [0] * 5
    f_bound_gt_match= [0] * 5
    f_bound_n_gt = [0] * 5
    # import pdb
    # pdb.set_trace()
    intersection, union, iou = compute_iou(predict_mask, gt_mask)
    running_intersection += intersection
    running_union += union
    example_iou += iou

    for bounds in range(5):
        _, _, _, fg_match, n_fg, gt_match, n_gt = db_eval_boundary(predict_mask, gt_mask, bound_th=bounds + 1)
        f_bound_fg_match[bounds] += fg_match
        f_bound_n_fg[bounds] += n_fg
        f_bound_gt_match[bounds] += gt_match
        f_bound_n_gt[bounds] += n_gt

    # example_iou /= len(dataset)
    text = "IOU: {}".format(example_iou)
    # print(text)
    # f.write(text + "\n")

    f_bound = [None] * 5
    for bounds in range(5):
        precision = f_bound_fg_match[bounds] / f_bound_n_fg[bounds]
        recall = f_bound_gt_match[bounds] / f_bound_n_gt[bounds]
        f_bound[bounds] = 2 * precision * recall / (precision + recall)

    text = ""
    for bounds in range(5):
        text += "F({})={},".format(bounds + 1, f_bound[bounds])
    # pdb.set_trace()
    f_bound = np.array(f_bound)
    num_nans = sum(np.isnan(f_bound))
    if num_nans > 0:
        f_bound = f_bound[~np.isnan(f_bound)]
    if len(f_bound) > 0:
        text += "F(avg) = {}\n".format(sum(f_bound) / f_bound.shape[0])
            # f.write(text)
        # print ("Bound F: " + str(sum(f_bound) / 5))
        return sum(f_bound) / f_bound.shape[0]
    else:
        return 0


#calculate weighted coverage
'''
NOTE: Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). 
Check: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score

@param y_pred  numpy array
@param y_true  numpy array

@return weigthed coverage per sample
'''
def calc_weighted_cov(y_pred, y_true):
    w_ious = sklearn.metrics.jaccard_score(y_true, y_pred, pos_label=1, average='weighted')
    return w_ious








