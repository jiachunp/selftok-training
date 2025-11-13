# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd

DIR_A = "/data/data/imagenet/data_with_parquet_name"                 # 组A
DIR_B = "/data/data/Unified_Parquet/ImageNet-Long-Caption_renamed"   # 组B
OUT_DIR = "/data/data/Unified_Parquet/ImageNet-Long-Caption_merged_by_image_id"
MERGE_HOW = "inner"  # 可改为 "outer" / "left" / "right"
USE_ARROW = True     # True 使用 pyarrow 读写，速度更快

os.makedirs(OUT_DIR, exist_ok=True)

def read_parquet(path):
    # 统一用 pyarrow 引擎，兼容性更好；必要时可换成 fastparquet
    return pd.read_parquet(path, engine="pyarrow" if USE_ARROW else None)

# 收集两边的同名文件
files_a = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_A, "*.parquet"))}
files_b = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_B, "*.parquet"))}

shared_names = sorted(set(files_a.keys()) & set(files_b.keys()))
missing_in_b = sorted(set(files_a.keys()) - set(files_b.keys()))
missing_in_a = sorted(set(files_b.keys()) - set(files_a.keys()))

if missing_in_b:
    print(f"[警告] 下列文件只在A中存在，将会跳过（数量 {len(missing_in_b)}）：前5个示例 -> {missing_in_b[:5]}")
if missing_in_a:
    print(f"[警告] 下列文件只在B中存在，将会跳过（数量 {len(missing_in_a)}）：前5个示例 -> {missing_in_a[:5]}")

total_rows_out = 0
for i, name in enumerate(shared_names, 1):
    path_a = files_a[name]
    path_b = files_b[name]
    print(f"[{i}/{len(shared_names)}] 合并 {name}")

    df_a = read_parquet(path_a)
    df_b = read_parquet(path_b)

    if "image_id" not in df_a.columns:
        raise KeyError(f"{path_a} 缺少 `image_id` 列")
    if "image_id" not in df_b.columns:
        raise KeyError(f"{path_b} 缺少 `image_id` 列")

    # 统一类型，避免一个是int一个是str导致无法匹配
    df_a["image_id"] = df_a["image_id"].astype(str)
    df_b["image_id"] = df_b["image_id"].astype(str)

    # 如果分片内出现重复 image_id，先去重，避免笛卡尔乘积
    before_a = len(df_a)
    before_b = len(df_b)
    df_a = df_a.sort_values("image_id").drop_duplicates(subset=["image_id"], keep="last")
    df_b = df_b.sort_values("image_id").drop_duplicates(subset=["image_id"], keep="last")
    if len(df_a) != before_a or len(df_b) != before_b:
        print(f"  - 去重: A {before_a}->{len(df_a)}, B {before_b}->{len(df_b)}")

    # 合并；重名列自动加后缀
    merged = pd.merge(
        df_a, df_b,
        on="image_id",
        how=MERGE_HOW,
        suffixes=("_a", "_b"),
        copy=False,
        validate=None  # 如需严格校验可设为 "one_to_one"
    )

    out_path = os.path.join(OUT_DIR, name)
    # 保存
    merged.to_parquet(out_path, index=False, engine="pyarrow" if USE_ARROW else None)

    print(f"  - 结果行数: {len(merged)}，已保存 -> {out_path}")
    total_rows_out += len(merged)

print(f"\n✅ 完成。输出目录: {OUT_DIR}")
print(f"合并的分片数: {len(shared_names)}，总输出行数: {total_rows_out}")
print(f"Join 类型: {MERGE_HOW}；若需保留任一侧全部样本，请把 MERGE_HOW 改为 'outer'")
