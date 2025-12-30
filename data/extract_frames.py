# import cv2
# import numpy as np
# import os
# from pathlib import *

# path_videos = "shanghaitech/training/videos/"
# path_frames = "shanghaitech/training/frames/"


# films = list()
# files = (x for x in Path(path_videos).iterdir() if x.is_file())
# for file in files:
#     print(str(file.name).split(".")[0], "is a file!")
#     films.append(file)

# for i, film in enumerate(films):
#     count = 0
#     vidcap = cv2.VideoCapture(str(film))
#     success, image = vidcap.read()
#     mapp = str(film.name).split(".")[0]
#     while success:
#         name = "shanghaitech/training/frames/%s/%d.jpg" % (mapp, count)
#         if not os.path.isdir("shanghaitech/training/frames/%s" % mapp):
#             os.mkdir("shanghaitech/training/frames/%s" % mapp)
#         cv2.imwrite(name, image)     # save frame as JPEG file
#         success, image = vidcap.read()
#         count += 1


import cv2
import os
from pathlib import Path

path_videos = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/avenue/testing_videos/"
path_frames = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/avenue/testing/frames/"

films = list()
files = (x for x in Path(path_videos).iterdir() if x.is_file())
for file in files:
    print(str(file.name).split(".")[0], "is a file!")
    films.append(file)

for i, film in enumerate(films):
    count = 0
    vidcap = cv2.VideoCapture(str(film))
    success, image = vidcap.read()
    mapp = str(film.name).split(".")[0]
    while success:
        name = f"{path_frames}{mapp}/{count}.jpg"
        if not os.path.isdir(f"{path_frames}{mapp}"):
            os.makedirs(f"{path_frames}{mapp}")
        cv2.imwrite(name, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1