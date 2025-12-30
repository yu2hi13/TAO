import pickle
import os
import cv2
import numpy as np
# 指定 all_video_segments.pkl 文件路径
input_pickle_path = "all_video_segments_10.pkl"

# 从文件加载 all_video_segments 字典
with open(input_pickle_path, 'rb') as f:
    all_video_segments = pickle.load(f)

# 指定保存的输出目录
output_dir = "video_segments_boxes_sam_10"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个视频和相应的分割结果
for video_name, video_data in all_video_segments.items():
    # 创建一个与视频名称匹配的输出文件
    output_file_path = os.path.join(output_dir, f"{video_name}.txt")
    
    # 收集每个对象的边界框信息
    with open(output_file_path, 'w') as f:
        for frame_idx, objects in video_data.items():
            for obj_id, mask in objects.items():
                # 将 mask 转换为 uint8 类型
                mask_uint8 = (mask[0] > 0).astype(np.uint8)
                
                # 获取非零点的坐标位置
                non_zero_y, non_zero_x = np.where(mask_uint8 > 0)
                
                # 如果非零坐标存在，计算最小和最大值
                if non_zero_x.size > 0 and non_zero_y.size > 0:
                    x_min, x_max = non_zero_x.min(), non_zero_x.max()
                    y_min, y_max = non_zero_y.min(), non_zero_y.max()
                    
                    # 将结果写入文件，格式为: frame_idx, x_min, y_min, x_max, y_max
                    f.write(f"{frame_idx+1},{x_min},{y_min},{x_max},{y_max},1\n")


    print(f"Processed and saved segmentation results for {video_name} to {output_file_path}")
