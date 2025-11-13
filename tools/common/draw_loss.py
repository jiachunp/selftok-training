# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, "valid")
    return re


def parsing_ldm(log_file, sample=-1, ratio=1):
    lines = open(log_file, "r", encoding="UTF-8").readlines()

    iters = []
    losses = []
    for line in lines:
        if "Step:" in line and "avg_loss" in line:
            step = int(line.strip().split("Step:")[-1].split("/")[0].split("[")[-1])
            loss = float(line.strip().split("avg_loss")[-1].split("(")[0])

            iters.append(step * ratio)
            losses.append(loss)
    iters = np.array(iters[:sample])
    losses_av = moving_average(np.array(losses[:sample]), 10)
    return iters, losses_av, losses_av


def parsing_mimo(log_file, sample=-1, ratio=1):
    lines = open(log_file, "r", encoding="UTF-8").readlines()

    iters = []
    text_losses = []
    image_losses = []
    for line in lines:
        if "loss:" in line and "mse:" in line:
            info = line.strip().split(",")
            step = int(info[1].split("/")[0].split("[")[-1]) + int(info[1].split("]")[-3].split("[")[-1]) * int(
                info[1].split("/")[1].split("]")[0]
            )

            text_loss = float(info[5].split(":")[-1].split("(")[0])
            image_loss = float(info[4].split(":")[-1].split("(")[0])

            iters.append(step * ratio)
            text_losses.append(text_loss)
            image_losses.append(image_loss)
    iters = np.array(iters[:sample])
    text_losses_av = moving_average(np.array(text_losses[:sample]), 20)
    image_losses_av = moving_average(np.array(image_losses[:sample]), 20)
    return iters, text_losses_av, image_losses_av


plt.switch_backend("Agg")
plt.figure()

log_file = "D:\data\logs\\mmdit\\mmdit_base.log"
save_file = log_file[:-4] + ".jpg"
iters, _, image_losses_av = parsing_mimo(log_file)
plt.plot(iters[: len(image_losses_av)], image_losses_av, "g", label="mmdit_base")

# log_file = 'D:\data\logs\\mmdit\\mmdit_sizectr_abspos.log'
# iters, _, image_losses_av = parsing_mimo(log_file)
# plt.plot(iters[:len(image_losses_av)], image_losses_av, 'b', label='mmdit_sizectr_abspos')


log_file = "D:\data\logs\\mmdit\\ldm.log"
iters, _, image_losses_av = parsing_ldm(log_file, sample=2000, ratio=0.2667)
plt.plot(iters[: len(image_losses_av)], image_losses_av, "r", label="ldm")

plt.ylabel("loss")
plt.xlabel("step")
plt.ylim((0.05, 0.45))
plt.legend()  # 个性化图例（颜色、形状等）
plt.savefig(save_file)
