import numpy as np
import argparse
import faiss
from video_dataset import VideoDatasetWithFlows, img_tensor2numpy
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
import sys
import cv2
import torch
import csv

# 设置 CUDA 设备为 GPU 8
os.environ["CUDA_VISIBLE_DEVICES"] = "5"



def gaussian_video(video, lengths, sigma=3):
    scores = np.zeros_like(video)
    prev = 0
    for cur in lengths:
        scores[prev: cur] = gaussian_filter1d(video[prev: cur], sigma)
        prev = cur
    return scores

def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

def draw_bounding_box(image, box, color, thickness=2):
    """在图像上绘制边框"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def evaluate(args, root):


    # 在代码开始部分创建CSV文件
    csv_file = 'anomalous_scores.csv'
    fieldnames = ['frame_index', 'top1_score', 'top1_bbox', 'top2_score', 'top2_bbox', 'top3_score', 'top3_bbox']
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    
    test_clip_lengths = np.load(os.path.join(root, args.dataset_name, 'test_clip_lengths.npy'))

    # 加载检测框数据
    bbox_data = np.load('/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_test.npy', allow_pickle=True)

    train_velocity = np.load('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    train_velocity = np.concatenate(train_velocity, 0)
    train_deep_features = np.load('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), allow_pickle=True)
    train_deep_features = np.concatenate(train_deep_features, 0)

    train_pose = np.load('extracted_features/{}/train/pose.npy'.format(args.dataset_name), allow_pickle=True)
    without_empty_frames = []
    for i in tqdm(range(len(train_pose))):
        if len(train_pose[i]):
            without_empty_frames.append(train_pose[i])
    train_pose = np.concatenate(without_empty_frames, 0)

    test_velocity = np.load('extracted_features/{}/test/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    test_pose = np.load('extracted_features/{}/test/pose.npy'.format(args.dataset_name), allow_pickle=True)
    test_deep_features = np.load('extracted_features/{}/test/deep_features.npy'.format(args.dataset_name), allow_pickle=True)

    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                         train=False, sequence_length=0, all_bboxes=None, normalize=False, mode='last')

    if args.dataset_name == 'ped2':
        velocity_density_estimator = GaussianMixture(n_components=2, random_state=0).fit(train_velocity)
    else:
        velocity_density_estimator = GaussianMixture(n_components=5, random_state=0).fit(train_velocity)

    train_velocity_scores = -velocity_density_estimator.score_samples(train_velocity)

    train_pose_scores = np.load('extracted_features/{}/train_pose_scores.npy'.format(args.dataset_name))
    train_deep_features_scores = np.load('extracted_features/{}/train_deep_features_scores.npy'.format(args.dataset_name))

    min_deep_features = np.min(train_deep_features_scores)
    max_deep_features = np.max(train_deep_features_scores)

    min_pose = np.min(train_pose_scores)
    max_pose = np.percentile(train_pose_scores, 99.9)

    min_velocity = np.min(train_velocity_scores)
    max_velocity = np.percentile(train_velocity_scores, 99.9)

    res = faiss.StandardGpuResources()
    
    gpu_index = 0  # 当前代码会默认使用 GPU 8，因为我们已经设置了 `CUDA_VISIBLE_DEVICES`
    
    index = faiss.IndexFlatL2(train_deep_features.shape[1])
    index_deep_features = faiss.index_cpu_to_gpu(res, gpu_index, index)
    index_deep_features.add(train_deep_features.astype(np.float32))

    index = faiss.IndexFlatL2(train_pose.shape[1])
    index_pose = faiss.index_cpu_to_gpu(res, gpu_index, index)
    index_pose.add(train_pose.astype(np.float32))

    test_velocity_scores = []
    test_deep_features_scores = []
    test_pose_scores = []

    frame_object_scores = []

    # 初始化列表来保存每帧最大对象的索引
    max_velocity_indices = []
    max_deep_features_indices = []

    # 创建保存图像的文件夹
    output_folder = 'anomalous_frames'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        cur_pose = test_pose[i]
        cur_velocity = test_velocity[i]
        cur_deep_features = test_deep_features[i]

        object_scores = {"velocity": [], "pose": [], "deep_features": []}

        if cur_pose.shape[0]:
            D, I = index_pose.search(cur_pose.astype(np.float32), 1)
            score_pose = np.mean(D, axis=1)
            max_score_pose = np.max(score_pose)
            test_pose_scores.append(max_score_pose)
            object_scores['pose'] = score_pose
        else:
            test_pose_scores.append(0)
            object_scores['pose'] = [0]

        D, I = index_deep_features.search(cur_deep_features.astype(np.float32), 1)
        score_features = np.mean(D, axis=1)
        max_score_features = np.max(score_features)
        test_deep_features_scores.append(max_score_features)
        object_scores['deep_features'] = score_features

        max_score_velocity = np.max(-velocity_density_estimator.score_samples(cur_velocity))
        test_velocity_scores.append(max_score_velocity)
        object_scores['velocity'] = -velocity_density_estimator.score_samples(cur_velocity)

        # 保存每帧图像中最大对象的速度和深度特征相似性对应的索引
        if len(object_scores['velocity']) > 0:
            max_velocity_index = np.argmax(object_scores['velocity'])
            max_velocity_indices.append(max_velocity_index)
        else:
            max_velocity_indices.append(None)  # 处理无对象的情况

        if len(object_scores['deep_features']) > 0:
            max_deep_features_index = np.argmax(object_scores['deep_features'])
            max_deep_features_indices.append(max_deep_features_index)
        else:
            max_deep_features_indices.append(None)  # 处理无对象的情况

        frame_object_scores.append(object_scores)

        if test_dataset.all_gt[i] == 1:  # 只输出异常帧的对象分数
            print(f"异常帧 {i}:")
            # 获取图像
            img = test_dataset[i][0].squeeze(0)  # 假设第一个元素为图像
            img = img_tensor2numpy(img)  # 转换为 NumPy 格式
            
            # 还原图像尺寸从 224x224 到 360x240
            original_size = (360, 240)
            img_resized = cv2.resize(img, original_size)  # 还原到原始尺寸
            
            # 执行逆标准化
            img_resized = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # 转换为 (C, H, W) 并转换为 float
            
            # 确保图像是 uint8 类型
            img_resized = img_resized.permute(1, 2, 0).numpy()  # 转换回 (H, W, C)
            img_resized = img_resized.astype(np.uint8)

            img_with_boxes = img_resized.copy()

            # # 获取最大异常速度的边界框
            # if max_velocity_indices[i] is not None:
            #     max_velocity_bbox = bbox_data[i][max_velocity_indices[i]]
            #     img_with_boxes = draw_bounding_box(img_with_boxes, max_velocity_bbox, color=(0, 255, 0))  # 绿色框

            # # 获取最大异常深度特征的边界框
            # if max_deep_features_indices[i] is not None:
            #     max_deep_features_bbox = bbox_data[i][max_deep_features_indices[i]]
            #     img_with_boxes = draw_bounding_box(img_with_boxes, max_deep_features_bbox, color=(255, 0, 0))  # 蓝色框

            if len(object_scores['velocity']) > 0 and len(object_scores['deep_features']) > 0:
                # 归一化
                norm_velocity = (object_scores['velocity'] - min_velocity) / (max_velocity - min_velocity)
                norm_deep_features = (object_scores['deep_features'] - min_deep_features) / (max_deep_features - min_deep_features)

                # 计算总分数
                total_scores = norm_velocity + norm_deep_features
                
                # 找到最大总分数的索引
                top_indices = np.argsort(total_scores)[-3:]

                
                # 将前三最大总分数写入CSV文件
                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for idx in top_indices:
                        bbox = tuple(bbox_data[i][idx])  # 将边界框的四个点转换为元组
                        writer.writerow([i, total_scores[idx], bbox, '', '', '', ''])
                        
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # 红色，蓝色，绿色
                # 获取该对象的 bbox
                for idx, color in zip(top_indices, colors):
                    bbox = bbox_data[i][idx]
                    img_with_boxes = draw_bounding_box(img_with_boxes, bbox, color=color)  # 使用不同的颜色

                # 用红色边框标记最大总分数的对象

            # 保存包含所有对象检测框的图像
            img_save_path = os.path.join(output_folder, f"anomalous_frame_{i}_highlighted.png")
            cv2.imwrite(img_save_path, img_with_boxes)
            print(f"保存图像到: {img_save_path}")

    test_velocity_scores = np.array(test_velocity_scores)
    test_deep_features_scores = np.array(test_deep_features_scores)
    test_pose_scores = np.array(test_pose_scores)

    test_velocity_scores = (test_velocity_scores - min_velocity) / (max_velocity - min_velocity)
    test_pose_scores = (test_pose_scores - min_pose) / (max_pose - min_pose)
    test_deep_features_scores = (test_deep_features_scores - min_deep_features) / (max_deep_features - min_deep_features)

    if args.dataset_name == 'shanghaitech':
        final_scores = gaussian_video(test_velocity_scores + test_pose_scores,
                                      test_clip_lengths, sigma=args.sigma)
    else:
        final_scores = gaussian_video(test_velocity_scores + test_pose_scores + test_deep_features_scores,
                                      test_clip_lengths, sigma=args.sigma)

    print('Micro AUC: ', roc_auc_score(test_dataset.all_gt, final_scores) * 100)
    print('Macro AUC: ', macro_auc(final_scores, test_dataset.all_gt, test_clip_lengths) * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--sigma", type=int, default=3, help='sigma for gaussian1d smoothing')
    args = parser.parse_args()
    root = 'data/'
    evaluate(args, root)
