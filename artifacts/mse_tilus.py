# /home/hohyeon/private/tilus-artifacts/artifacts/figure8_mutis_mse.py

import os
import math
import traceback

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from hidet.ir import data_type
from mutis.kernels.baselines import MatmulLayer
from mutis.kernels.vm.matmul_mma_decode import matmul_mma_decode
from mutis.kernels.vm.matmul_mma import reference_matmul_mma

SEED = 0
DEVICE = "cuda"

OUT_DIR = os.environ.get("TILUS_ARTIFACT_RESULTS_DIR", "./results")
OUT_CSV = os.path.join(OUT_DIR, "figure8_mutis_mse.csv")
OUT_TXT = os.path.join(OUT_DIR, "figure8_mutis_mse.txt")
OUT_PDF = os.path.join(OUT_DIR, "figure8_mutis_mse.pdf")

K = 8192
N = 8192
GROUP_SIZE = 128

SEQLENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

DTYPE_ORDER = [
    "uint8",
    "float6_e3m2",
    "uint4b",
    "int4b",
    "uint2b",
    "uint1b",
]

DTYPE_LABEL = {
    "uint8": "u8",
    "float6_e3m2": "f6",
    "uint4b": "u4",
    "int4b": "i4",
    "uint2b": "u2",
    "uint1b": "u1",
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()) ** 2).mean().item()


def make_reference_output(layer, a: torch.Tensor):
    from mutis.kernels.vm.matmul_mma import reference_matmul_mma
    from mutis.kernels.vm.matmul_mma_decode import matmul_mma_decode

    group_size = layer.group_size
    if group_size == -1:
        group_size = layer.k

    b_decoded = matmul_mma_decode(
        k=layer.k,
        n=layer.n,
        dtype=layer.b_dtype,
        output_dtype=layer.a_dtype,
        x=layer.b,
    )

    if layer.m > 256:
        y_ref = torch.matmul(a, b_decoded)
        path = "decode_plus_torch_matmul"
    else:
        y_ref = reference_matmul_mma(
            m=layer.m,
            n=layer.n,
            k=layer.k,
            group_size=group_size,
            a=a,
            b=b_decoded,
            scales=layer.scales,
            zeros=layer.zeros,
            a_dtype=layer.a_dtype,
            b_dtype=layer.b_dtype,
            c_dtype=layer.a_dtype,
        )
        path = "official_reference_matmul_mma"

    return y_ref, path


def run_one(seqlen: int, b_dtype: str):
    group_size = GROUP_SIZE if data_type(b_dtype).is_integer() else -1

    set_seed(SEED + seqlen + K + N + hash(b_dtype) % 1000)

    layer = MatmulLayer.create(
        "mutis",
        a_dtype=data_type("float16"),
        b_dtype=data_type(b_dtype),
        group_size=group_size,
        m=seqlen,
        k=K,
        n=N,
    )

    a = torch.randn((seqlen, K), device=DEVICE, dtype=torch.float16)

    with torch.no_grad():
        y_real = layer.run(a)
        torch.cuda.synchronize()

        y_ref, reference_path = make_reference_output(layer, a)
        torch.cuda.synchronize()

    err = y_real.float() - y_ref.float()

    return {
        "device": torch.cuda.get_device_name(0),
        "runner": "mutis",
        "runner_label": "Tilus",
        "b_dtype": b_dtype,
        "dtype_tick": DTYPE_LABEL[b_dtype],
        "group_size": group_size,
        "seqlen": seqlen,
        "m": seqlen,
        "k": K,
        "n": N,
        "mse_real_vs_ref": mse(y_real, y_ref),
        "max_abs_err": err.abs().max().item(),
        "mean_abs_err": err.abs().mean().item(),
        "reference_path": reference_path,
        "status": "ok",
        "error": "",
    }


def run_experiments():
    rows = []

    for b_dtype in DTYPE_ORDER:
        for seqlen in SEQLENS:
            print(f"[RUN] mutis dtype={b_dtype}, seqlen={seqlen}", flush=True)

            try:
                row = run_one(seqlen, b_dtype)
            except Exception:
                row = {
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                    "runner": "mutis",
                    "runner_label": "Tilus",
                    "b_dtype": b_dtype,
                    "dtype_tick": DTYPE_LABEL[b_dtype],
                    "group_size": GROUP_SIZE if data_type(b_dtype).is_integer() else -1,
                    "seqlen": seqlen,
                    "m": seqlen,
                    "k": K,
                    "n": N,
                    "mse_real_vs_ref": float("nan"),
                    "max_abs_err": float("nan"),
                    "mean_abs_err": float("nan"),
                    "reference_path": "",
                    "status": "error",
                    "error": traceback.format_exc().splitlines()[-1],
                }

            rows.append(row)
            print(
                f"  status={row['status']} "
                f"mse={row['mse_real_vs_ref']} "
                f"path={row['reference_path']} "
                f"error={row['error']}",
                flush=True,
            )

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame):
    ok_df = df[df["status"] == "ok"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, b_dtype in zip(axes, DTYPE_ORDER):
        sub = ok_df[ok_df["b_dtype"] == b_dtype].sort_values("seqlen")

        if len(sub) > 0:
            ax.plot(
                sub["seqlen"],
                sub["mse_real_vs_ref"],
                marker="o",
                linewidth=1.8,
                label="Tilus / Mutis",
            )

        ax.axvline(256, linestyle="--", linewidth=1)
        ax.text(
            256,
            0.95,
            "m=256 boundary",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            fontsize=8,
        )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(SEQLENS)
        ax.set_xticklabels([str(x) for x in SEQLENS], rotation=45, fontsize=8)
        ax.set_title(DTYPE_LABEL[b_dtype])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("MSE(real, ref)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"Figure 8: Tilus/Mutis Kernel MSE vs Sequence Length, K={K}, N={N}", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = run_experiments()

    df.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w") as f:
        f.write(df.to_string(index=False))

    plot_results(df)

    print(f"\nSaved csv: {OUT_CSV}")
    print(f"Saved txt: {OUT_TXT}")
    print(f"Saved pdf: {OUT_PDF}")


if __name__ == "__main__":
    main()
