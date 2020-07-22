'''
The purpose of this script is to calculate metrics for inferences done on the Vaihingen and Bing Datasets
Commands to run script:
- for Vaihingen:
python calculate_metrics.py --path /Users/dsengupta/Desktop/pred_npys --dataset V
- for Bing:
python calculate_metrics.py --path /Users/dsengupta/Desktop/pred_npys --dataset B

Save predictions and ground truths as .np arrays. pred_npys should have the following directory structure:

pred_npys/
	- 101_pred.npy
	- 101_gt.npy
	.
	.
	.
	- 168_pred.npy
	- 168_gt.npy
'''

from metrics_utils import dice_hard, calc_weighted_cov, calc_boundF_darNet, compute_iou
from sklearn.metrics import recall_score, precision_score
import argparse
import numpy as np
import os


################################# CALCULATE METRICS FOR ALL TEST CASES ##################################
# these cases did not produce center masks for vaihingen so discard
# for fair comparison for single and multi

# exclude = [106, 119, 124, 161] 
# exclude = []

'''
Calculate the average dice over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''
def calc_avg_dice(input_dir, lower, upper, exclude):
    wcs = []
    for i in range(lower, upper):
        # print (i)
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        wc = dice_hard(y_pred, y_true)
        if wc == 0: #missed middle building
            print (i)
            continue
        wcs.append(wc)

    return np.mean(wcs)

'''
Calculate the average weighted coverage over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''
def calc_avg_weighted_cov(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    wcs = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        wc = calc_weighted_cov(y_pred, y_true)
        if i in exclude: #missed middle building
            print (i)
            continue
        wcs.append(wc)

    return np.mean(wcs)

'''
Calculate the average bound F over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''
def calc_avg_bound_f(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    boundFs = []
    for i in range(lower, upper):
        # print(i)
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        boundF = calc_boundF_darNet(y_pred, y_true)
        if i in exclude: #missed middle building
            print (i)
            continue
        # if boundF != -1:
        boundFs.append(boundF)
    # pdb.set_trace()

    return np.mean(boundFs)

'''
Calculate the average IoU over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''
def calc_avg_iou(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    ious = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        iou = compute_iou(y_pred, y_true)[2]
        if i in exclude: #missed middle building
            print (i)
            continue
        ious.append(iou)

    return np.mean(ious)

################ FOR BING ####################

def calc_avg_dice_b(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    wcs = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        if y_pred.shape != (64,64):
            print(i)
            continue
        wc = dice_hard(y_pred, y_true)
        wcs.append(wc)

    return np.mean(wcs)


def calc_avg_weighted_cov_b(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    wcs = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        if y_pred.shape != (64,64):
            print(i)
            continue
        wc = calc_weighted_cov(y_pred, y_true)
        wcs.append(wc)

    return np.mean(wcs)

def calc_avg_bound_f_b(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    boundFs = []
    for i in range(lower, upper):
        # print(i)
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        if y_pred.shape != (64,64):
            print(i)
            continue
        boundF = calc_boundF_darNet(y_pred, y_true)
        boundFs.append(boundF)
    # pdb.set_trace()

    return np.mean(boundFs)

def calc_avg_iou_b(input_dir, lower, upper, exclude):
    import pdb
    # pdb.set_trace()
    ious = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        y_pred = np.round(y_pred)
        y_pred[y_pred != 0] = 1

        y_true = np.round(y_true)
        y_true[y_true != 0] = 1
        if y_pred.shape != (64,64):
            print(i)
            continue
        iou = iou_calc(y_pred, y_true)
        ious.append(iou)

    return np.mean(ious)

'''
Calculate the average precision over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''

def calc_avg_precision(input_dir, lower, upper, exclude):
    ious = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        x = precision_score(y_pred, y_true, average='weighted')
        ious.append(x)
    
    return np.average(ious)

'''
Calculate the average recall over all test images

@param input_dir  directory with predictions and gts stored as .npy files
@param lower  lower bound image id number of test set
@param upper  upper bound image id numbr of test set

NOTE: The input dir should consist of .npy files of the predictions and the gts. 
The format of the prediction file names should be #_pred.npy
The format of the ground truth file should be #_gt.npy 

Example: If the building ID of an image is 101 then input dir should contain 101_pred.npy and 101_gt.npy
'''

def calc_avg_recall(input_dir, lower, upper, exclude):
    ious = []
    for i in range(lower, upper):
        y_pred = np.load(os.path.join(input_dir, str(i) + "_pred.npy"))
        y_true = np.load(os.path.join(input_dir, str(i) + "_gt.npy"))
        x = recall_score(y_pred, y_true, average='weighted')
        ious.append(x)
    
    return np.average(ious)



if __name__ == "__main__":
    print("########### METRICS ##########")
    # for Vaihingen, the test set is image 101-168
    lower_v = 101
    upper_v = 168

    # for Bing, the test set is image 101-168
    lower_b = 335
    upper_b = 605

    parser = argparse.ArgumentParser()
    # parser.add_argument("--path", help="path to test set predictions and groud truth set.", "--dataset", help="Please pick V for Vaihingen or B for Bing", action="store_true")
    parser.add_argument('--path', required=True,
                        metavar="/path/to/predictions/",
                        help='Directory of prediction and ground truth .npy files')
    parser.add_argument('--dataset', required=True,
                        metavar="Vaihingen or Bing",
                        help='Specify V for Vaihingen and B for Bing')
    args = parser.parse_args()

    if args.path == None:
        print ("please input path to predictions directory")
    if args.dataset == None:
        print("please select dataset to run metrics on")
    else:
        if args.dataset == "V":
            print ("Metrics for Vaihingen")
            print("Vaihingen Dice")
            print(calc_avg_dice(args.path, lower_v, upper_v, []))
            print("Vaihingen IoU")
            print(calc_avg_iou(args.path, lower_v, upper_v, []))
            print("Vaihingen Weighted Cov")
            print(calc_avg_weighted_cov(args.path, lower_v, upper_v, []))
            print("Vaihingen BoundF")
            print(calc_avg_bound_f(args.path, lower_v, upper_v, []))
        if args.dataset == "B":
            print ("Metrics for Bing")
            print ("Bing Dice")
            print(calc_avg_dice(args.path, lower_b, upper_b, []))
            print ("Bing IoU")
            print(calc_avg_iou(args.path, lower_b, upper_b, []))
            print("Bing Weighted Cov")
            print(calc_avg_weighted_cov(args.path, lower_b, upper_b, []))
            print ("Bing BoundF")
            print(calc_avg_bound_f(args.path, lower_b, upper_b, []))

