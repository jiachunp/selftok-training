#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import argparse
import matplotlib.pyplot as plt

# 去掉 ANSI 颜色转义，如：[[34m2025-09-07 08:59:36[0m]
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# 解析一行训练日志（时间戳可有可无，关键是 step / selftok_ce / ce / steps/sec）
LINE_RE = re.compile(
    r'\[(?P<ts>[^\]]+)\].*?\(step=(?P<step>\d+)\)\s*'
    r'.*?Train\s+Loss\s+selftok_ce:\s*(?P<stce>-?\d+(?:\.\d+)?)\s*,\s*'
    r'Train\s+Loss\s+ce:\s*(?P<ce>-?\d+(?:\.\d+)?)\s*,\s*'
    r'Train\s+Steps/Sec:\s*(?P<sps>-?\d+(?:\.\d+)?)',
    re.IGNORECASE
)

def parse_log(path):
    steps, selftok_ce, ce, sps, tss = [], [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = ANSI_RE.sub('', raw.strip())
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group('step')))
                selftok_ce.append(float(m.group('stce')))
                ce.append(float(m.group('ce')))
                sps.append(float(m.group('sps')))
                tss.append(m.group('ts'))
    return steps, selftok_ce, ce, sps, tss

def moving_average(vals, k=5):
    if k <= 1:
        return vals[:]
    out, q, s = [], [], 0.0
    for v in vals:
        q.append(v); s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def main():
    parser = argparse.ArgumentParser(description="Parse training log and plot loss curves.")
    parser.add_argument("log_path", help="日志 txt 文件路径")
    parser.add_argument("--smooth", type=int, default=1, help="移动平均窗口（点数），默认 5；设为 1 关闭平滑")
    parser.add_argument("--out", type=str, default=None, help="输出图片路径（默认：同名 _loss.png）")
    args = parser.parse_args()

    steps, stce, ce, sps, tss = parse_log(args.log_path)
    if not steps:
        raise SystemExit("未在日志中匹配到训练条目，请检查格式/正则。")

    # 排序（防止日志乱序）
    idx = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in idx]
    stce  = [stce[i]  for i in idx]
    ce    = [ce[i]    for i in idx]
    sps   = [sps[i]   for i in idx]

    stce_ma = moving_average(stce, args.smooth)
    ce_has_nonzero = any(x != 0.0 for x in ce)
    if ce_has_nonzero:
        ce_ma = moving_average(ce, args.smooth)

    # 画图：只用 matplotlib，单图，无指定颜色
    plt.figure(figsize=(9, 5))
    plt.plot(steps, stce, label="selftok_ce (raw)", linewidth=1, alpha=0.35)
    plt.plot(steps, stce_ma, label=f"selftok_ce (MA{args.smooth})", linewidth=2)

    if ce_has_nonzero:
        plt.plot(steps, ce, label="ce (raw)", linewidth=1, alpha=0.35, linestyle=':')
        plt.plot(steps, ce_ma, label=f"ce (MA{args.smooth})", linewidth=2, linestyle='--')

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Step")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = args.out or os.path.splitext(args.log_path)[0] + "_loss.png"
    plt.savefig(out_path, dpi=150)
    print(f"已保存图片到：{out_path}")
    plt.show()

if __name__ == "__main__":
    main()
