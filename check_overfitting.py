# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# import numpy as np
# from typing import List, Dict, Tuple

# def count_mismatches(a: np.ndarray, b: np.ndarray, *,
#                      use_isclose: bool = False,
#                      rtol: float = 1e-5,
#                      atol: float = 1e-8,
#                      treat_nan_equal: bool = True) -> int:
#     """
#     返回 a 与 b 逐元素不一致的数量。
#     - use_isclose=False：精确匹配；NaN 视作相等（可配置）。
#     - use_isclose=True ：使用 np.isclose（equal_nan 受 treat_nan_equal 控制）。
#     """
#     if a.shape != b.shape:
#         # 形状不同直接按总元素数计全不一致
#         return max(a.size, b.size)

#     if use_isclose:
#         eq = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=treat_nan_equal)
#     else:
#         if treat_nan_equal:
#             eq = (a == b) | (np.isnan(a) & np.isnan(b))
#         else:
#             eq = (a == b)
#     mismatches = int(a.size - np.count_nonzero(eq))
#     return mismatches

# def compare_dirs(src_dir: Path,
#                  ref_dir: Path,
#                  use_isclose: bool = False,
#                  rtol: float = 1e-5,
#                  atol: float = 1e-8,
#                  treat_nan_equal: bool = True,
#                  verbose: bool = True) -> Tuple[List[Dict], Dict]:
#     """
#     遍历 src_dir 下的 .npy 文件，与 ref_dir 中的同名文件比较。
#     返回（逐文件结果列表，总结字典）。
#     """
#     results: List[Dict] = []
#     missing_ref: List[str] = []
#     bad_shape_files: List[str] = []

#     src_files = sorted(src_dir.rglob("*.npy"))

#     for f in src_files:
#         rel_name = f.name
#         ref_f = ref_dir / rel_name
#         rec = {
#             "file": str(f),
#             "ref_file": str(ref_f),
#             "exists_in_ref": ref_f.exists(),
#             "src_shape_ok": False,
#             "ref_shape_ok": False,
#             "src_shape": None,
#             "ref_shape": None,
#             "mismatch_count": None,
#         }

#         if not ref_f.exists():
#             missing_ref.append(rel_name)
#             results.append(rec)
#             if verbose:
#                 print(f"[MISSING] {rel_name} -> {ref_f}")
#             continue

#         try:
#             a = np.load(f, allow_pickle=False).reshape(1, 1536)
#         except Exception as e:
#             if verbose:
#                 print(f"[ERROR] Load src failed: {f} ({e})")
#             results.append(rec)
#             continue

#         try:
#             b = np.load(ref_f, allow_pickle=False).reshape(1, 1536)
#         except Exception as e:
#             if verbose:
#                 print(f"[ERROR] Load ref failed: {ref_f} ({e})")
#             results.append(rec)
#             continue

#         rec["src_shape"] = tuple(a.shape)
#         rec["ref_shape"] = tuple(b.shape)
#         rec["src_shape_ok"] = (a.shape == (1, 1536))
#         rec["ref_shape_ok"] = (b.shape == (1, 1536))

#         if not (rec["src_shape_ok"] and rec["ref_shape_ok"]):
#             bad_shape_files.append(rel_name)
#             # 形状不合要求也仍可计算不一致元素（若形状不同，函数会按最大元素数计）
#         rec["mismatch_count"] = count_mismatches(
#             a, b,
#             use_isclose=use_isclose,
#             rtol=rtol,
#             atol=atol,
#             treat_nan_equal=treat_nan_equal,
#         )
#         results.append(rec)

#     # 统计
#     compared = [r for r in results if r["exists_in_ref"] and r["mismatch_count"] is not None]
#     total_compared = len(compared)
#     mismatched_files = [r for r in compared if r["mismatch_count"] > 0]
#     total_mismatch_elems = sum(r["mismatch_count"] for r in compared)
#     max_mismatch = max((r["mismatch_count"] for r in compared), default=0)
#     max_mismatch_files = [r["file"] for r in compared if r["mismatch_count"] == max_mismatch]

#     summary = {
#         "total_src_files": len(src_files),
#         "total_found_in_ref": sum(1 for r in results if r["exists_in_ref"]),
#         "total_missing_in_ref": len(missing_ref),
#         "total_compared_files": total_compared,
#         "files_with_bad_shape": len(bad_shape_files),
#         "bad_shape_examples": bad_shape_files[:10],
#         "num_files_with_mismatches": len(mismatched_files),
#         "total_mismatched_elements": int(total_mismatch_elems),
#         "max_mismatch_in_a_file": int(max_mismatch),
#         "files_with_max_mismatch": max_mismatch_files[:10],
#         "comparison_mode": "isclose" if use_isclose else "exact",
#         "rtol": rtol,
#         "atol": atol,
#         "treat_nan_equal": treat_nan_equal,
#     }

#     return results, summary

# def main():
#     parser = argparse.ArgumentParser(description="逐文件比较 .npy 内容一致性并统计不一致元素数量")
#     parser.add_argument("--src_dir", type=Path,
#                         default=Path("/home/jovyan/zfd/SelfTok-o-main/outputs_imagenet1k_overfit_10000_ema"),
#                         help="源目录（将被遍历的 .npy 文件所在目录）")
#     parser.add_argument("--ref_dir", type=Path,
#                         default=Path("/home/jovyan/zfd/SelfTok-o-main/dataset/overfitting/image_token"),
#                         help="参考目录（查找同名 .npy 文件）")
#     parser.add_argument("--isclose", action="store_true",
#                         help="使用 np.isclose 进行近似比较（默认关闭，使用精确比较）")
#     parser.add_argument("--rtol", type=float, default=1e-5, help="np.isclose 的 rtol")
#     parser.add_argument("--atol", type=float, default=1e-8, help="np.isclose 的 atol")
#     parser.add_argument("--no-equal-nan", action="store_true",
#                         help="不将 NaN 视作相等（默认将同位置的 NaN 视为相等）")
#     parser.add_argument("--print_all", action="store_true",
#                         help="逐文件全部打印（默认只打印不一致>0或异常的文件）")
#     args = parser.parse_args()

#     results, summary = compare_dirs(
#         args.src_dir,
#         args.ref_dir,
#         use_isclose=args.isclose,
#         rtol=args.rtol,
#         atol=args.atol,
#         treat_nan_equal=not args.no_equal_nan,
#         verbose=True
#     )

#     print("\n=== Per-file mismatch counts ===")
#     for r in results:
#         show = args.print_all or (not r["exists_in_ref"]) or (not r["src_shape_ok"]) or (not r["ref_shape_ok"]) or ((r["mismatch_count"] or 0) > 0)
#         if show:
#             print(f"{Path(r['file']).name:>40} | exists_in_ref={r['exists_in_ref']} | "
#                   f"src_shape={r['src_shape']} ok={r['src_shape_ok']} | "
#                   f"ref_shape={r['ref_shape']} ok={r['ref_shape_ok']} | "
#                   f"mismatch_count={r['mismatch_count']}")

#     print("\n=== Summary ===")
#     for k, v in summary.items():
#         print(f"{k}: {v}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import csv
from typing import List, Dict, Tuple

def _equality_mask(a: np.ndarray, b: np.ndarray,
                   use_isclose: bool,
                   rtol: float, atol: float,
                   treat_nan_equal: bool) -> np.ndarray:
    if a.shape != b.shape:
        # 形状不同，无法产生逐元素比较的掩码
        return None
    if use_isclose:
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=treat_nan_equal)
    else:
        if treat_nan_equal:
            return (a == b) | (np.isnan(a) & np.isnan(b))
        else:
            return (a == b)

def count_mismatches(a: np.ndarray, b: np.ndarray, *,
                     use_isclose: bool = False,
                     rtol: float = 1e-5,
                     atol: float = 1e-8,
                     treat_nan_equal: bool = True) -> int:
    """
    返回 a 与 b 逐元素不一致的数量。
    - use_isclose=False：精确匹配；NaN 视作相等（可配置）。
    - use_isclose=True ：使用 np.isclose（equal_nan 受 treat_nan_equal 控制）。
    """
    if a.shape != b.shape:
        return max(a.size, b.size)

    eq = _equality_mask(a, b, use_isclose, rtol, atol, treat_nan_equal)
    mismatches = int(a.size - np.count_nonzero(eq))
    return mismatches

def mismatch_positions(a: np.ndarray, b: np.ndarray, *,
                       use_isclose: bool = False,
                       rtol: float = 1e-5,
                       atol: float = 1e-8,
                       treat_nan_equal: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回（pos, src_vals, ref_vals），其中：
      - pos：不一致 token 的位置索引（优先按 (1, N) 的第二维；否则退化为扁平化索引）
      - src_vals/ref_vals：对应位置的数值
    若形状不同，返回 (None, None, None)。
    """
    if a.shape != b.shape:
        return None, None, None

    eq = _equality_mask(a, b, use_isclose, rtol, atol, treat_nan_equal)
    # 对齐到 token 维度：
    if a.ndim == 2 and a.shape[0] == 1:
        # 形状 (1, N) —— token 维是第 1 维
        bad_mask = ~eq[0]
        positions = np.nonzero(bad_mask)[0]
        src_vals = a[0, positions]
        ref_vals = b[0, positions]
    elif a.ndim == 1:
        bad_mask = ~eq
        positions = np.nonzero(bad_mask)[0]
        src_vals = a[positions]
        ref_vals = b[positions]
    else:
        # 其它形状：退化为扁平索引
        bad_mask = ~eq
        positions = np.flatnonzero(bad_mask.ravel())
        src_vals = a.ravel()[positions]
        ref_vals = b.ravel()[positions]

    return positions.astype(int), src_vals, ref_vals

def compare_dirs(src_dir: Path,
                 ref_dir: Path,
                 use_isclose: bool = False,
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 treat_nan_equal: bool = True,
                 verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    遍历 src_dir 下的 .npy 文件，与 ref_dir 中的同名文件比较。
    返回（逐文件结果列表，总结字典）。
    """
    results: List[Dict] = []
    missing_ref: List[str] = []
    bad_shape_files: List[str] = []

    src_files = sorted(src_dir.rglob("*.npy"))

    for f in src_files:
        rel_name = f.name
        ref_f = ref_dir / rel_name
        rec = {
            "file": str(f),
            "ref_file": str(ref_f),
            "exists_in_ref": ref_f.exists(),
            "src_shape_ok": False,
            "ref_shape_ok": False,
            "src_shape": None,
            "ref_shape": None,
            "mismatch_count": None,
            # 新增：不一致位置（token idx）与数值
            "mismatch_positions": None,   # np.ndarray[int]
            "mismatch_src_vals": None,    # np.ndarray[float]
            "mismatch_ref_vals": None,    # np.ndarray[float]
        }

        if not ref_f.exists():
            missing_ref.append(rel_name)
            results.append(rec)
            if verbose:
                print(f"[MISSING] {rel_name} -> {ref_f}")
            continue

        try:
            a = np.load(f, allow_pickle=False).reshape(1, 1536)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Load src failed: {f} ({e})")
            results.append(rec)
            continue

        try:
            b = np.load(ref_f, allow_pickle=False).reshape(1, 1536)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Load ref failed: {ref_f} ({e})")
            results.append(rec)
            continue

        rec["src_shape"] = tuple(a.shape)
        rec["ref_shape"] = tuple(b.shape)
        rec["src_shape_ok"] = (a.shape == (1, 1536))
        rec["ref_shape_ok"] = (b.shape == (1, 1536))
        if not (rec["src_shape_ok"] and rec["ref_shape_ok"]):
            bad_shape_files.append(rel_name)

        # 统计数量
        rec["mismatch_count"] = count_mismatches(
            a, b,
            use_isclose=use_isclose,
            rtol=rtol,
            atol=atol,
            treat_nan_equal=treat_nan_equal,
        )

        # 统计位置（若形状不同则 None）
        pos, src_vals, ref_vals = mismatch_positions(
            a, b,
            use_isclose=use_isclose,
            rtol=rtol,
            atol=atol,
            treat_nan_equal=treat_nan_equal,
        )
        rec["mismatch_positions"] = pos
        rec["mismatch_src_vals"] = src_vals
        rec["mismatch_ref_vals"] = ref_vals

        results.append(rec)

    # 统计
    compared = [r for r in results if r["exists_in_ref"] and r["mismatch_count"] is not None]
    total_compared = len(compared)
    mismatched_files = [r for r in compared if (r["mismatch_count"] or 0) > 0]
    total_mismatch_elems = sum(int(r["mismatch_count"] or 0) for r in compared)
    max_mismatch = max((int(r["mismatch_count"] or 0) for r in compared), default=0)
    max_mismatch_files = [r["file"] for r in compared if int(r["mismatch_count"] or 0) == max_mismatch]

    summary = {
        "total_src_files": len(src_files),
        "total_found_in_ref": sum(1 for r in results if r["exists_in_ref"]),
        "total_missing_in_ref": len(missing_ref),
        "total_compared_files": total_compared,
        "files_with_bad_shape": len(bad_shape_files),
        "bad_shape_examples": bad_shape_files[:10],
        "num_files_with_mismatches": len(mismatched_files),
        "total_mismatched_elements": int(total_mismatch_elems),
        "max_mismatch_in_a_file": int(max_mismatch),
        "files_with_max_mismatch": max_mismatch_files[:10],
        "comparison_mode": "isclose" if use_isclose else "exact",
        "rtol": rtol,
        "atol": atol,
        "treat_nan_equal": treat_nan_equal,
    }
    return results, summary

def save_positions_csv(results: List[Dict], csv_path: Path):
    """
    把所有文件的不一致位置及数值写入 CSV：
    columns: file, pos, src_value, ref_value, abs_diff
    仅写入 mismatch_count > 0 且有位置数据的记录。
    """
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "pos", "src_value", "ref_value", "abs_diff"])
        for r in results:
            pos = r["mismatch_positions"]
            svals = r["mismatch_src_vals"]
            rvals = r["mismatch_ref_vals"]
            if pos is None or svals is None or rvals is None:
                continue
            if len(pos) == 0:
                continue
            diffs = np.abs(svals - rvals)
            for p, sv, rv, dv in zip(pos.tolist(),
                                     svals.astype(float).tolist(),
                                     rvals.astype(float).tolist(),
                                     diffs.astype(float).tolist()):
                writer.writerow([r["file"], p, sv, rv, dv])

def main():
    parser = argparse.ArgumentParser(description="比较 .npy 内容一致性，统计不一致数量与错误 token 位置")
    parser.add_argument("--src_dir", type=Path,
                        default=Path("/home/jovyan/zfd/SelfTok-o-main/outputs_imagenet1k_overfit_10000_ema"),
                        help="源目录（将被遍历的 .npy 文件所在目录）")
    parser.add_argument("--ref_dir", type=Path,
                        default=Path("/home/jovyan/zfd/SelfTok-o-main/dataset/overfitting/image_token"),
                        help="参考目录（查找同名 .npy 文件）")
    parser.add_argument("--isclose", action="store_true",
                        help="使用 np.isclose 进行近似比较（默认关闭，使用精确比较）")
    parser.add_argument("--rtol", type=float, default=1e-5, help="np.isclose 的 rtol")
    parser.add_argument("--atol", type=float, default=1e-8, help="np.isclose 的 atol")
    parser.add_argument("--no-equal-nan", action="store_true",
                        help="不将 NaN 视作相等（默认将同位置的 NaN 视为相等）")
    parser.add_argument("--print_all", action="store_true",
                        help="逐文件全部打印（默认只打印不一致>0或异常的文件）")
    parser.add_argument("--max_positions_print", type=int, default=50,
                        help="每个文件终端最多打印的不一致位置数量")
    parser.add_argument("--save_positions_csv", type=Path, default=None,
                        help="可选：把所有不一致位置写入 CSV 路径（例如 mismatches.csv）")
    args = parser.parse_args()

    results, summary = compare_dirs(
        args.src_dir,
        args.ref_dir,
        use_isclose=args.isclose,
        rtol=args.rtol,
        atol=args.atol,
        treat_nan_equal=not args.no_equal_nan,
        verbose=True
    )

    print("\n=== Per-file mismatch counts & positions ===")
    for r in results:
        show = args.print_all or (not r["exists_in_ref"]) or (not r["src_shape_ok"]) or (not r["ref_shape_ok"]) or ((r["mismatch_count"] or 0) > 0)
        if not show:
            continue

        fname = Path(r["file"]).name
        print(f"{fname:>40} | exists_in_ref={r['exists_in_ref']} | "
              f"src_shape={r['src_shape']} ok={r['src_shape_ok']} | "
              f"ref_shape={r['ref_shape']} ok={r['ref_shape_ok']} | "
              f"mismatch_count={r['mismatch_count']}")

        # 打印不一致位置（若可用）
        pos = r["mismatch_positions"]
        if pos is None:
            print("  mismatched positions: N/A (shape differs)")
        else:
            if len(pos) == 0:
                print("  mismatched positions: []")
            else:
                to_show = pos[:args.max_positions_print].tolist()
                suffix = "" if len(pos) <= args.max_positions_print else f" ... (+{len(pos)-args.max_positions_print} more)"
                print(f"  mismatched positions (token idx): {to_show}{suffix}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.save_positions_csv is not None:
        save_positions_csv(results, args.save_positions_csv)
        print(f"\n[Saved] mismatch positions CSV -> {args.save_positions_csv}")

if __name__ == "__main__":
    main()
