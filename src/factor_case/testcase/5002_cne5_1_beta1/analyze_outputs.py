#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 5002_cne5_1_beta1 测试中 C++ 与 Python 输出的耗时和值差异。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_result(path: Path) -> pd.DataFrame:
    """Load result csv (skipping metadata header)."""

    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")

    return pd.read_csv(path, comment="#")


def _beta_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col.startswith("stock_") and col.endswith("_beta")]


def main() -> None:
    base_dir = Path(__file__).parent
    cpp_path = base_dir / "output.csv"
    py_path = base_dir / "output_py.csv"

    cpp_df = _load_result(cpp_path)
    py_df = _load_result(py_path)

    key_cols = ["time_index", "step", "operation"]
    for col in key_cols:
        if col not in cpp_df.columns or col not in py_df.columns:
            raise ValueError(f"缺少关键列: {col}")

    merged = cpp_df.merge(
        py_df,
        on=key_cols,
        suffixes=("_cpp", "_py"),
        how="outer",
        indicator=True,
        sort=True,
    )

    if not (merged["_merge"] == "both").all():
        missing_cpp = merged.loc[merged["_merge"] == "right_only", key_cols]
        missing_py = merged.loc[merged["_merge"] == "left_only", key_cols]
        raise ValueError(
            "C++ 与 Python 结果行不对齐:\n"
            f"  仅在 Python 中存在的行:\n{missing_cpp}\n"
            f"  仅在 C++ 中存在的行:\n{missing_py}"
        )

    beta_cols_cpp = _beta_columns(col for col in cpp_df.columns if col not in key_cols)
    beta_cols_py = _beta_columns(col for col in py_df.columns if col not in key_cols)
    common_beta = sorted(set(beta_cols_cpp).intersection(beta_cols_py))
    if not common_beta:
        raise ValueError("未找到重叠的 beta 列，无法对比。")

    elapsed_cpp = merged["elapsed_ms_cpp"].to_numpy()
    elapsed_py = merged["elapsed_ms_py"].to_numpy()

    # 构造 beta 矩阵
    cpp_beta = merged[[f"{col}_cpp" for col in common_beta]].to_numpy(dtype=float)
    py_beta = merged[[f"{col}_py" for col in common_beta]].to_numpy(dtype=float)

    diff = cpp_beta - py_beta
    abs_diff = np.abs(diff)
    isnan_mask = np.isnan(cpp_beta) | np.isnan(py_beta)
    effective_counts = (~isnan_mask).sum(axis=1)

    # 记录每个 beta 的列名，方便定位
    beta_array_cols = np.array(common_beta)

    with np.errstate(invalid="ignore"):
        mean_abs_diff = np.where(
            effective_counts > 0,
            np.nanmean(abs_diff, axis=1),
            np.nan,
        )
        max_abs_diff = np.nanmax(np.where(isnan_mask, np.nan, abs_diff), axis=1)

    tolerance = 1e-6
    mismatch_mask = (abs_diff > tolerance) & ~isnan_mask
    mismatch_counts = mismatch_mask.sum(axis=1)

    print("=== 5002_cne5_1_beta1 C++ vs Python 对比 ===")
    print(f"比较文件:\n  C++ : {cpp_path}\n  Python : {py_path}")
    print(f"共有 {len(common_beta)} 条 beta 列，{len(merged)} 个时间点")
    print()

    for idx, row in merged[key_cols].iterrows():
        ti, step, op = row
        cpp_time = elapsed_cpp[idx]
        py_time = elapsed_py[idx]
        time_diff = cpp_time - py_time
        ratio = cpp_time / py_time if py_time and not math.isnan(py_time) else math.nan
        print(
            f"[time={ti}, step={step}, op={op}]"
            f"  C++耗时={cpp_time:.3f} ms, Python耗时={py_time:.3f} ms, "
            f"差值={time_diff:+.3f} ms, 速度比(C++/Py)={ratio:.3f}"
        )

        mean_diff = mean_abs_diff[idx]
        max_diff = max_abs_diff[idx]
        mismatches = mismatch_counts[idx]
        valid = effective_counts[idx]
        print(
            f"  Beta差异: 有效比对{valid}个, 均值差={mean_diff:.3e},"
            f" 最大差={max_diff:.3e}, 超过阈值(>{tolerance:g})={mismatches}"
        )
        if mismatches > 0:
            top_indices = np.argsort(-abs_diff[idx])[: min(10, mismatches)]
            details = []
            for col_idx in top_indices:
                col_name = beta_array_cols[col_idx]
                cpp_val = cpp_beta[idx, col_idx]
                py_val = py_beta[idx, col_idx]
                delta = diff[idx, col_idx]
                details.append(
                    f"    {col_name}: cpp={cpp_val:.6g}, py={py_val:.6g}, diff={delta:+.6g}"
                )
            print("  超阈值TOP差值:")
            print("\n".join(details))
        print()

    overall_mean = np.nanmean(mean_abs_diff)
    overall_max = np.nanmax(max_abs_diff)
    total_mismatch = mismatch_counts.sum()
    total_valid = effective_counts.sum()
    print("=== 汇总 ===")
    print(f"平均耗时(C++)={elapsed_cpp.mean():.3f} ms, 平均耗时(Python)={elapsed_py.mean():.3f} ms")
    print(f"平均绝对差={overall_mean:.3e}, 最大绝对差={overall_max:.3e}")
    print(f"超出阈值的 beta 数量={total_mismatch} / {total_valid}")


if __name__ == "__main__":
    main()

