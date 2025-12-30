import pickle
import numpy as np

# 读取pkl文件
input_pickle_path = "/data/cxli/yuzhi/segment-anything-2/all_video_segments_10.pkl"
with open(input_pickle_path, 'rb') as f:
    all_video_segments = pickle.load(f)

# 创建一个新的字典来存储处理后的数据
processed_video_segments = {}

# 遍历所有视频
for video_index, video_data in all_video_segments.items():
    # 将视频序号加1
    new_video_index = int(video_index) 
    processed_video_segments[new_video_index] = {}

    # # 遍历每一帧
    # for frame_idx, frame_data in video_data.items():
    #     # 初始化一个全零的掩码，大小与第一个obj_id的掩码相同
    #     # 假设所有obj_id的掩码大小相同
    #     first_obj_id = next(iter(frame_data))
    #     combined_binary_mask = np.zeros_like(frame_data[first_obj_id], dtype=np.uint8)

    #     # 遍历每个obj_id的分割结果
    #     for obj_id, mask in frame_data.items():
    #         # 将分割结果转成二值化
    #         binary_mask = (mask > 0).astype(np.uint8)
    #         # 合并到总的二值化掩码中
    #         combined_binary_mask = np.logical_or(combined_binary_mask, binary_mask).astype(np.uint8)

    #     # 将处理后的帧数据存储到新的字典中
    #     processed_video_segments[new_video_index][frame_idx] = combined_binary_mask
    for frame_idx, frame_data in video_data.items():
        # 创建一个新的字典来存储二值化的分割结果
        binary_frame_data = {}

        # 遍历每个obj_id的分割结果
        for obj_id, mask in frame_data.items():
            # 将分割结果转成二值化
            
            # 将二值化结果存储到字典中
            binary_frame_data[obj_id] = mask

        # 将处理后的帧数据存储到新的字典中
        processed_video_segments[new_video_index][frame_idx] = binary_frame_data

# 保存处理后的数据到新的pkl文件
output_pickle_path = "/data/cxli/yuzhi/segment-anything-2/processed_video_segments_sam_10.pkl"
with open(output_pickle_path, 'wb') as f:
    pickle.dump(processed_video_segments, f)

print("Processing complete. Results are saved in 'processed_video_segments_b+.pkl'.")