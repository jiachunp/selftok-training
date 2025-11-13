# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

sys.path.append(".")
from mimogpt.utils import walk_all_files, mkdirs, get_dirs


def draw_tag(img, text, left, top, text_color=(255, 0, 0), text_size=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("tools/common/fonts/msyh.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


root_dir = "/ssd/ssd1/MIMO_BACKUP/visualize/wny/multi_modal_gpt_f16/iter_27339.pth"

src_dirs = [
    "cfg6.0_top32",
    "cfg6.0_top64",
    "cfg6.0_top128",
    "cfg6.0_top256",
    "cfg6.0_top512",
    "cfg6.0_top1024",
    "cfg6.0_top2048",
    "cfg6.0_top4096",
]

tags = [
    "cfg6.0_top32",
    "cfg6.0_top64",
    "cfg6.0_top128",
    "cfg6.0_top256",
    "cfg6.0_top512",
    "cfg6.0_top1024",
    "cfg6.0_top2048",
    "cfg6.0_top4096",
]

dst_dir = "comp_topk"
use_hor_concat = False  # 是否水平拼接
is_leave_dir = True
shuffle = False
shuffle_tag = False
text_color = (255, 0, 0)
text_size = 40
left = 35
top = 55

src_dirs = [os.path.join(root_dir, src_dir) for src_dir in src_dirs]

dst_dir = os.path.join(root_dir, dst_dir)
mkdirs(dst_dir)

if is_leave_dir:
    sub_dir_paths = [src_dirs[0]]
else:
    _, _, sub_dir_paths = get_dirs(src_dirs[0])

cnt = 0
for sub_dir in sub_dir_paths:
    file_num, file_names, file_paths = walk_all_files(sub_dir, sort=True)
    for i in tqdm(range(file_num)):
        img = cv2.imread(file_paths[i])
        img = draw_tag(
            img,
            tags[0],
            left + 15 * (img.shape[1] // 250 - 2),
            top + 25 * (img.shape[0] // 250 - 2),
            text_color=text_color,
            text_size=text_size + 10 * (img.shape[0] // 250 - 2),
        )
        relative_path = file_paths[i][len(src_dirs[0]) + 1 :]
        imgs = [img]
        indexs = [m for m in range(len(src_dirs) - 1)]
        if shuffle:
            indexs = random.sample(indexs, len(indexs))

        for j in range(len(src_dirs)):
            if j > 0:
                temp_path = os.path.join(src_dirs[indexs[j - 1] + 1], relative_path)
                temp_img = cv2.imread(temp_path)
                if temp_img is None:
                    continue
                if use_hor_concat:
                    scale = img.shape[0] / temp_img.shape[0]
                else:
                    scale = img.shape[1] / temp_img.shape[1]
                temp_img = cv2.resize(temp_img, (0, 0), fx=scale, fy=scale)
                if shuffle and shuffle_tag:
                    tag = tags[indexs[j - 1] + 1]
                else:
                    tag = tags[j]
                temp_img = draw_tag(
                    temp_img,
                    tag,
                    left + 15 * (img.shape[1] // 250 - 2),
                    top + 25 * (img.shape[0] // 250 - 2),
                    text_color=text_color,
                    text_size=text_size + 10 * (img.shape[0] // 250 - 2),
                )
                imgs.append(temp_img)
        if len(imgs) == len(src_dirs):
            cnt += 1
            dst_path = os.path.join(dst_dir, relative_path)
            mkdirs(os.path.dirname(dst_path))
            if shuffle:
                with open(os.path.join(os.path.dirname(dst_path), "indexs.txt"), "a") as f:
                    indexs_str = [str(index) for index in indexs]
                    f.write(dst_path + "\t")
                    f.write(" ".join(indexs_str) + "\n")
            if use_hor_concat:
                dst_img = np.concatenate(imgs, axis=1)
            else:
                dst_img = np.concatenate(imgs, axis=0)
            cv2.imwrite(dst_path, dst_img)
print("Overall {:.0f} files processed".format(cnt / len(src_dirs)))
