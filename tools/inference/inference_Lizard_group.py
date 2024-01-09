from argparse import ArgumentParser

from PGT.apis.inference import init_detector, inference_detector
import matplotlib.pyplot as plt
# import mmcv
import numpy as np
import scipy.io as sio
import os
import glob
import scipy
from scipy.optimize import linear_sum_assignment
import re
import sys
import cv2

sys.path.append('/storage/homefs/jg23p152/project')
from MCSpatNet.utils import get_npy_files as get_file_list
from MCSpatNet.utils import parse_json_file

unpaired_true_all = []
unpaired_pred_all = []
true_inst_type_all = []
pred_inst_type_all = []
paired_all = []
true_idx_offset = 0
pred_idx_offset = 0
colors = {1: (0, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)} # bgr

def filter_with_json(img_list, json_file):
    json_data = parse_json_file(json_file)
    img_names = [sample['img_id'] for sample in json_data]
    img_list = [img for img in img_list if os.path.basename(os.path.split(img)[0]) in img_names]
    return img_list

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_file',
                        default="/Path to Dataset/Lizard/Test/Images/",
                        help='Image file')
    parser.add_argument('--img_file_256',
                        default="/Path to Dataset/Lizard/Lizard_COCO_6Classes/Lizard_Test/",
                        help='Image file')
    parser.add_argument('--config',
                        default="./configs/Swin_Lizard_PGT.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="/Path to Dataset/PGT_Lizard.pth",
                        help='Checkpoint file')
    parser.add_argument('--gt-file',
                        default="/Path to Dataset/Test/Labels/",
                        help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--dataset', type=str, default="CoNSeP", help='inference on')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--num_classes', 
        type=int, 
        default=6, 
        help='num_classes')
    parser.add_argument(
        '--mode', 
        type=str, 
        default="globals", 
        help='inference on')
    parser.add_argument('--json_file', 
                        default="/Path to Dataset/Lizard/Labels/cv0/Lizard_test.json",
                        help='Image file')
    parser.add_argument('--save_path', 
                        default="/Path to Dataset/Lizard/Labels/cv0/")
    parser.add_argument('--save_patches', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    return args


def det(args, pred_centroid, pred_inst_type, img_name, img_idx, full_img_list, patch_list, pred_centroid_orig):
    global pred_idx_offset, pred_inst_type_all, unpaired_pred_all, unpaired_true_all, true_idx_offset, true_inst_type_all, paired_all
    if 'endo' in args.gt_file.lower():
        img_path = os.path.join(args.gt_file, img_name[:-2] + ".mat")
    else:
        img_path = os.path.join(args.gt_file, img_name + ".mat")
    true_info = sio.loadmat(img_path)
    true_centroid = (true_info["inst_centroid"]).astype("float32")
    true_inst_type = (true_info["inst_type"]).astype("int32")

    if true_centroid.shape[0] != 0:
        true_inst_type = true_inst_type[:, 0] if args.dataset != 'TCGA' else true_inst_type[0,:] + 1
    else:  # no instance at all
        true_centroid = np.array([[0, 0]])
        true_inst_type = np.array([0])

    pred_centroid = np.asarray(pred_centroid).astype("float32")
    pred_inst_type = np.asarray(pred_inst_type).astype("int32")

    if pred_centroid.shape[0] != 0:
        pred_inst_type = pred_inst_type[:, 0]
    else:  # no instance at all
        pred_centroid = np.array([[0, 0]])
        pred_inst_type = np.array([0])

    paired, unpaired_true, unpaired_pred = pair_coordinates(
        true_centroid, pred_centroid, 6
    )

    true_idx_offset = (
        true_idx_offset + true_inst_type_all[-1].shape[0] if img_idx != 0 else 0
    )
    pred_idx_offset = (
        pred_idx_offset + pred_inst_type_all[-1].shape[0] if img_idx != 0 else 0
    )
    true_inst_type_all.append(true_inst_type)
    pred_inst_type_all.append(pred_inst_type)

    if paired.shape[0] != 0:  # ! sanity
        paired[:, 0] += true_idx_offset
        paired[:, 1] += pred_idx_offset
        paired_all.append(paired)

    unpaired_true += true_idx_offset
    unpaired_pred += pred_idx_offset
    unpaired_true_all.append(unpaired_true)
    unpaired_pred_all.append(unpaired_pred)

    # Save Predict Result
    pred_mat = {}
    pred_mat["inst_centroid"] = pred_centroid
    pred_mat["inst_type"] = pred_inst_type
    mat_save_path = os.path.join(args.save_path, 'mat')
    os.makedirs(mat_save_path, exist_ok=True)
    sio.savemat(os.path.join(mat_save_path, img_name + ".mat"),
                pred_mat)

    # Save Visual Result
    file_ext = 'png' if args.dataset != 'TCGA' else 'tif'
    full_img_name = [im for im in full_img_list if img_name+'.'+file_ext in os.path.normpath(im).split(os.path.sep)]
    full_img = cv2.imread(full_img_name[0])
    overlay_save_path = os.path.join(args.save_path, 'overlay')
    os.makedirs(overlay_save_path, exist_ok=True)
    cv2.imwrite(os.path.join(overlay_save_path, img_name + ".png"), full_img)


    full_img_pred = full_img.copy()
    for i, centroid in enumerate(pred_centroid):
        full_img_pred = cv2.circle(full_img_pred, (int(centroid[0]), int(centroid[1])),3, colors[pred_inst_type[i]], cv2.FILLED)
    cv2.imwrite(os.path.join(overlay_save_path, img_name + ".png"), full_img_pred)

    full_img_gt = full_img.copy()
    for i, centroid in enumerate(true_centroid):
        full_img_gt = cv2.circle(full_img_gt, (int(centroid[0]), int(centroid[1])), 3, colors[true_inst_type[i]], cv2.FILLED)
    cv2.imwrite(os.path.join(overlay_save_path, img_name + "_gt.png"), full_img_gt)

    if args.save_patches:
        patch_save_path = os.path.join(args.save_path, 'patch')
        patch_save_path = os.path.join(patch_save_path, img_name)
        os.makedirs(patch_save_path, exist_ok=True)
        
        for j, patch in enumerate(patch_list):
            patch_name = os.path.basename(patch)
            patch = cv2.imread(patch)
            for i, centroid in enumerate(pred_centroid_orig[j]):
                patch = cv2.circle(patch, (int(centroid[0]), int(centroid[1])), 3, colors[pred_inst_type[i]], cv2.FILLED)
            cv2.imwrite(os.path.join(patch_save_path, patch_name), patch)
            
            




def main(args):
    global pred_idx_offset, pred_inst_type_all, unpaired_pred_all, unpaired_true_all, true_idx_offset, true_inst_type_all, paired_all
    num_classes = args.num_classes
    
    if os.path.isdir(args.checkpoint):
        args.checkpoint = glob.glob(os.path.join(args.checkpoint, "best_*_iter_*.pth"))[-1]

    model = init_detector(
        args.config, args.checkpoint, device=args.device
    )
    model.inference_on = args.mode
    thr = args.score_thr
    
    if thr:
        img_list = get_file_list(args.img_file_256, "png")
        img_list = filter_with_json(img_list, args.json_file)2        file_ext = 'png' if args.dataset != 'TCGA' else 'tif'
        full_img_list = get_file_list(args.img_file, file_ext)
        img_ids = [os.path.basename(os.path.normpath(img)).split('.')[0] for img in full_img_list]
        predicted = [f.path for f in os.scandir(os.path.join(args.save_path, 'mat')) if f.is_file()]
        missing = [img for img in img_ids if os.path.join(args.save_path, 'mat',img+'.mat') not in predicted]

        img_list = [img for img in img_list if os.path.basename(os.path.dirname(img)) in missing]
        full_img_list = [img for img in full_img_list if os.path.basename(os.path.normpath(img)).split('.')[0] in missing]
        # img_list.sort()
        img_name_oral_last = None
        img_idx = 0
        pred_centroid = []
        pred_inst_type = []
        patch_list = []
        pred_centroid_orig = []
        idx = 0
        for img_path in img_list:
            img_name_oral, img_name = os.path.split(img_path)
            ret = re.match(r'(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*).png$', img_name)
            img_name_oral = os.path.basename(img_name_oral)
            if (not args.overwrite) and (os.path.isfile(os.path.join(args.save_path, 'mat', img_name_oral + ".mat"))):
                continue
            if img_name_oral != img_name_oral_last:
                print("predicting: ", os.path.dirname(img_path))
            x_start = int(ret.group(3))
            y_start = int(ret.group(6))

            result = inference_detector(model, img_path)
            # Visual Latent Group
            # result, latent = inference_detector(model, img_path, return_attn=True)

            if idx == 0:
                pred_centroid = []
                pred_inst_type = []
                pred_centroid_orig = []
                patch_list = []
                img_name_oral_last = img_name_oral

            elif x_start == 0 and y_start == 0:
                det(args, pred_centroid, pred_inst_type, img_name_oral_last, img_idx, full_img_list, patch_list, pred_centroid_orig)
                img_name_oral_last = img_name_oral
                # pred info
                pred_centroid = []
                pred_inst_type = []
                pred_centroid_orig = []
                patch_list = []

                img_idx += 1
            for i in range(num_classes):
                classes = result[i][np.where(result[i][:, 2] > thr)]
                classes = classes[:, :2]
                pred_centroid_orig.append(classes)
                classes[:, 0] += y_start
                classes[:, 1] += x_start
                pred_centroid.extend(classes)
                pred_type = np.full((classes.shape[0], 1), i + 1)
                pred_inst_type.extend(pred_type)
                patch_list.append(img_path)

            if idx == len(img_list) - 1:
                det(args, pred_centroid, pred_inst_type, img_name_oral_last, img_idx, full_img_list, patch_list, pred_centroid_orig)
            idx += 1

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        w = [1, 1]
        tp_d = paired_pred_type.shape[0]
        fp_d = unpaired_pred_type.shape[0]
        fn_d = unpaired_true_type.shape[0]

        tp_tn_dt = (paired_pred_type == paired_true_type).sum()
        fp_fn_dt = (paired_pred_type != paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        # if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

        results_list = [f1_d, acc_type]
        for type_uid in type_uid_list:
            f1_type = _f1_type(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        results_list = np.array(results_list)

        Avg = 0
        print("\n===========================Evaluation=======================")
        print("Thr: {}".format(thr))
        print("F1 Detection:{}".format(results_list[0]))
        for i in range(num_classes):
            print("F1 Type{}:{}".format(i, results_list[i + 2]))
            Avg += results_list[i+2]
        print("F1c Avg:{}".format(Avg/num_classes))
        print("===========================Evaluation=======================\n")
    return


def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
            type_samples = (paired_true == type_id) | (paired_pred == type_id)

            paired_true = paired_true[type_samples]
            paired_pred = paired_pred[type_samples]

            tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
            tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
            fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
            fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

            fp_d = (unpaired_pred == type_id).sum()
            fn_d = (unpaired_true == type_id).sum()

            f1_type = (2 * (tp_dt + tn_dt)) / (
                    2 * (tp_dt + tn_dt)
                    + w[0] * fp_dt
                    + w[1] * fn_dt
                    + w[2] * fp_d
                    + w[3] * fn_d
            )
            return f1_type


if __name__ == '__main__':

    args = parse_args()
    main(args)
