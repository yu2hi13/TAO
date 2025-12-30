import torch
from tqdm import tqdm
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from UCSDped_dataset import UCSDPed2Dataset
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score

transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = UCSDPed2Dataset(root_dir='/data/cxli/yuzhi/datasets/UCSDped2/UCSDped2/test', transform=transform)

import pickle

# 加载分割结果
def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 定义计算指标的函数
def calculate_metrics(gt_mask, pred_mask):
    if gt_mask.sum() == 0:
        return None, None, None, None  # 如果只有一个类别，返回 None
    if pred_mask.sum() == 0:
        return 0, 0, 0, 0

    # 计算指标
    
    pixel_auroc = roc_auc_score(gt_mask.ravel(), pred_mask.ravel())
    pixel_ap = average_precision_score(gt_mask.ravel(), pred_mask.ravel())

    # 计算 precision-recall 曲线
    precision, recall, _ = precision_recall_curve(gt_mask.ravel(), pred_mask.ravel())
    pixel_aupro = auc(recall, precision)

    # 计算 F1 分数
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pixel_f1 = f1_score(gt_mask.ravel(), pred_mask.ravel())

    return pixel_auroc, pixel_ap, pixel_aupro, pixel_f1

segmentation_results = load_results('/data/cxli/yuzhi/segment-anything-2/processed_video_segments_sam_10.pkl')
video_metrics = {}  # 存储每个视频的指标
overall_metrics = {'auroc': [], 'ap': [], 'aupro': [], 'f1': []}  # 存储总体的指标

for i, sample in enumerate(tqdm(dataset, desc='Evaluating')):
    image, mask, anomaly, anomalous_boxes, video_index, frame_index = sample['image'], sample['mask'], sample['anomaly'], sample['anomaly_box'], sample['video_index'], sample['frame_index']
    
    if video_index + 1 in segmentation_results and frame_index in segmentation_results[video_index + 1]:
            # 获取指定视频和帧的掩码数据
            pred_masks = segmentation_results[video_index + 1][frame_index].values()
            
            try:
                # 初始化一个全零的掩码，大小与第一个掩码相同
                first_mask = next(iter(pred_masks))
                combined_mask = np.zeros_like(first_mask, dtype=np.float32)

                # 遍历所有掩码并取最大值
                for mask_iter in pred_masks:
                    combined_mask = np.maximum(combined_mask, mask_iter)
                
                # 归一化到0-1之间
                if combined_mask.max() > 0:
                    pred_mask = combined_mask / combined_mask.max()
                else:
                    pred_mask = combined_mask
            except StopIteration:
                # 如果 pred_masks 为空
                pred_mask = np.zeros(mask.shape, dtype=np.float32)
    else:
        pred_mask = np.zeros(mask.shape, dtype=np.float32)
    # 计算指标
    pixel_auroc, pixel_ap, pixel_aupro, pixel_f1 = calculate_metrics(mask, pred_mask)
    
    if pixel_auroc is not None and pixel_ap is not None and pixel_aupro is not None and pixel_f1 is not None:
        # 累积当前视频的指标
        if video_index not in video_metrics:
            video_metrics[video_index] = {'auroc': [], 'ap': [], 'aupro': [], 'f1': []}
        video_metrics[video_index]['auroc'].append(pixel_auroc)
        video_metrics[video_index]['ap'].append(pixel_ap)
        video_metrics[video_index]['aupro'].append(pixel_aupro)
        video_metrics[video_index]['f1'].append(pixel_f1)
        
        # 累积总体的指标
        overall_metrics['auroc'].append(pixel_auroc)
        overall_metrics['ap'].append(pixel_ap)
        overall_metrics['aupro'].append(pixel_aupro)
        overall_metrics['f1'].append(pixel_f1)

# 计算每个视频的平均指标
for video_index, metrics in video_metrics.items():
    video_metrics[video_index]['auroc'] = np.mean(metrics['auroc'])
    video_metrics[video_index]['ap'] = np.mean(metrics['ap'])
    video_metrics[video_index]['aupro'] = np.mean(metrics['aupro'])
    video_metrics[video_index]['f1'] = np.mean(metrics['f1'])

# 计算总体平均指标
if overall_metrics['auroc']:
    overall_metrics['auroc'] = np.mean(overall_metrics['auroc'])
if overall_metrics['ap']:
    overall_metrics['ap'] = np.mean(overall_metrics['ap'])
if overall_metrics['aupro']:
    overall_metrics['aupro'] = np.mean(overall_metrics['aupro'])
if overall_metrics['f1']:
    overall_metrics['f1'] = np.mean(overall_metrics['f1'])

# 打印视频级别的指标
for video_index, metrics in video_metrics.items():
    print(f"Video {video_index} Metrics:")
    print(f"  AUROC: {metrics['auroc']:.4f}, AP: {metrics['ap']:.4f}, AUPRO: {metrics['aupro']:.4f}, F1: {metrics['f1']:.4f}")

print("Overall Metrics:")
if overall_metrics['auroc'] is not None:
    print(f"  AUROC: {overall_metrics['auroc']:.4f}")
if overall_metrics['ap'] is not None:
    print(f"  AP: {overall_metrics['ap']:.4f}")
if overall_metrics['aupro'] is not None:
    print(f"  AUPRO: {overall_metrics['aupro']:.4f}")
if overall_metrics['f1'] is not None:
    print(f"  F1: {overall_metrics['f1']:.4f}")