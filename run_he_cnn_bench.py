# run_he_cnn_bench.py

import time
import platform
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tenseal as ts


# -----------------------------
# CKKS context (TenSEAL)
# -----------------------------
def make_ckks_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=(40, 21, 21, 40), global_scale=2**40):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree,
        list(coeff_mod_bit_sizes),
    )
    ctx.global_scale = global_scale
    # Needed for dot() in your TenSEAL
    ctx.generate_galois_keys()
    return ctx


def bytes_of_ciphertext_ckks(ct: ts.CKKSVector) -> int:
    return len(ct.serialize())


# -----------------------------
# Tiny CNN-style feature extractor
# conv (3x3) -> square -> avgpool 2x2
# Input is 8x8 (digits)
# Output: K filters * (3x3 pooled map) = K*9 features
# -----------------------------
def conv3x3_valid(img8x8: np.ndarray, kernel3x3: np.ndarray) -> np.ndarray:
    # valid conv: output 6x6
    out = np.zeros((6, 6), dtype=np.float64)
    for r in range(6):
        for c in range(6):
            patch = img8x8[r:r+3, c:c+3]
            out[r, c] = np.sum(patch * kernel3x3)
    return out


def avgpool2x2_stride2(x6x6: np.ndarray) -> np.ndarray:
    # output 3x3
    out = np.zeros((3, 3), dtype=np.float64)
    for r in range(0, 6, 2):
        for c in range(0, 6, 2):
            out[r//2, c//2] = np.mean(x6x6[r:r+2, c:c+2])
    return out


def extract_features_plain(x64: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    img = x64.reshape(8, 8)
    feats = []
    for k in kernels:
        z = conv3x3_valid(img, k)
        a = z * z  # square activation (HE-friendly)
        p = avgpool2x2_stride2(a)  # 3x3
        feats.append(p.reshape(-1))
    return np.concatenate(feats, axis=0)  # shape K*9


# -----------------------------
# Encrypted feature extraction
# We represent input as CKKSVector length 64.
# For each conv position, we do dot(mask, enc_x) where mask has the 3x3 kernel weights placed at proper pixel indices.
# We keep conv outputs encrypted scalars, apply square under HE, avgpool under HE,
# then decrypt final pooled features.
# -----------------------------
def build_conv_masks_for_kernel(kernel3x3: np.ndarray):
    # For each output position (r,c) in 6x6, build a length-64 mask
    masks = []
    for r in range(6):
        for c in range(6):
            mask = np.zeros((64,), dtype=np.float64)
            for kr in range(3):
                for kc in range(3):
                    rr = r + kr
                    cc = c + kc
                    idx = rr * 8 + cc
                    mask[idx] = float(kernel3x3[kr, kc])
            masks.append(mask)
    return masks  # len 36, each length 64


def enc_dot(enc_x: ts.CKKSVector, w: np.ndarray) -> ts.CKKSVector:
    # returns encrypted scalar (CKKSVector size 1)
    return enc_x.dot(w.tolist())


def extract_features_encrypted(enc_x: ts.CKKSVector, kernels: np.ndarray):
    # returns numpy array features (decrypted), shape K*9
    all_feats = []

    for k in kernels:
        masks = build_conv_masks_for_kernel(k)  # 36 masks
        # conv outputs encrypted scalars (36)
        conv_enc = [enc_dot(enc_x, m) for m in masks]

        # square activation under HE: a = z^2
        act_enc = [z * z for z in conv_enc]

        # avgpool 2x2 stride2 over 6x6 -> 3x3
        # mapping indices: (r,c) in 6x6 -> idx = r*6 + c
        pooled_enc = []
        for pr in range(3):
            for pc in range(3):
                r0 = pr * 2
                c0 = pc * 2
                idx00 = (r0)*6 + (c0)
                idx01 = (r0)*6 + (c0+1)
                idx10 = (r0+1)*6 + (c0)
                idx11 = (r0+1)*6 + (c0+1)

                s = act_enc[idx00] + act_enc[idx01] + act_enc[idx10] + act_enc[idx11]
                avg = s * 0.25
                pooled_enc.append(avg)

        # decrypt pooled features (9 scalars)
        pooled_plain = np.array([pe.decrypt()[0] for pe in pooled_enc], dtype=np.float64)
        all_feats.append(pooled_plain)

    return np.concatenate(all_feats, axis=0)  # K*9


def run():
    # ---------------------------
    # Dataset: sklearn digits -> binary 0 vs 1
    # ---------------------------
    digits = load_digits()
    X = digits.data.astype(np.float64)  # 64 feats
    y = digits.target.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Standardize input (helps conv scale + CKKS numeric stability)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float64)
    X_test_s = scaler.transform(X_test).astype(np.float64)

    # ---------------------------
    # Define K random conv kernels (small, fixed)
    # ---------------------------
    rng = np.random.default_rng(123)
    K = 4
    kernels = rng.normal(loc=0.0, scale=0.5, size=(K, 3, 3)).astype(np.float64)

    # ---------------------------
    # Train classifier on extracted features (plaintext)
    # ---------------------------
    F_train = np.stack([extract_features_plain(X_train_s[i], kernels) for i in range(len(X_train_s))], axis=0)
    F_test_plain = np.stack([extract_features_plain(X_test_s[i], kernels) for i in range(len(X_test_s))], axis=0)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(F_train, y_train)

    y_pred_plain = clf.predict(F_test_plain)
    acc_plain = accuracy_score(y_test, y_pred_plain)

    # Plaintext latency: feature extraction + predict
    n_runs = min(108, len(X_test_s))  # keep comparable to your LR runs
    t0 = time.perf_counter()
    for i in range(n_runs):
        f = extract_features_plain(X_test_s[i], kernels).reshape(1, -1)
        _ = clf.predict(f)
    t1 = time.perf_counter()
    plain_latency_ms = (t1 - t0) / n_runs * 1000.0
    plain_throughput = n_runs / (t1 - t0)

    # ---------------------------
    # CKKS encrypted feature extraction
    # ---------------------------
    ctx = make_ckks_context()
    # ciphertext size for one encrypted input
    enc0 = ts.ckks_vector(ctx, X_test_s[0])
    ct_size = bytes_of_ciphertext_ckks(enc0)

    t0 = time.perf_counter()
    y_pred_encfeat = []
    for i in range(n_runs):
        enc_x = ts.ckks_vector(ctx, X_test_s[i])
        feats = extract_features_encrypted(enc_x, kernels).reshape(1, -1)
        y_hat = clf.predict(feats)[0]
        y_pred_encfeat.append(y_hat)
    t1 = time.perf_counter()

    enc_latency_ms = (t1 - t0) / n_runs * 1000.0
    enc_throughput = n_runs / (t1 - t0)
    acc_encfeat = accuracy_score(y_test[:n_runs], np.array(y_pred_encfeat))

    # ---------------------------
    # Save results for paper
    # ---------------------------
    system_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }

    results = [
        {
            "scheme": "PLAINTEXT",
            "model": "TinyCNN-features (conv+square+avgpool) + LogReg",
            "dataset": "sklearn digits (8x8, 0 vs 1)",
            "n_test_samples": n_runs,
            "accuracy": acc_plain,
            "avg_ciphertext_size_bytes": 0,
            "latency_ms_per_sample": plain_latency_ms,
            "throughput_samples_per_sec": plain_throughput,
        },
        {
            "scheme": "CKKS",
            "model": "Encrypted feature extraction + LogReg",
            "dataset": "sklearn digits (8x8, 0 vs 1)",
            "n_test_samples": n_runs,
            "accuracy": acc_encfeat,
            "avg_ciphertext_size_bytes": ct_size,
            "latency_ms_per_sample": enc_latency_ms,
            "throughput_samples_per_sec": enc_throughput,
        },
    ]

    df = pd.DataFrame(results)
    df.to_csv("results_section5_cnn.csv", index=False)
    pd.DataFrame([system_info]).to_csv("system_info_cnn.csv", index=False)

    print("\n=== System Info ===")
    print(system_info)

    print("\n=== Section 5 Results (Tiny CNN-style) ===")
    print(df.to_string(index=False))

    print("\nSaved: results_section5_cnn.csv and system_info_cnn.csv")


if __name__ == "__main__":
    run()
