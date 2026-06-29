# run_he_lr_bench.py

import time
import math
import platform
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tenseal as ts


def approx_sigmoid_poly1(x: float) -> float:
    # Very simple approximation: sigma(x) ≈ 0.5 + 0.125x
    # Clip to [0,1] to behave like a probability
    y = 0.5 + 0.125 * x
    return float(min(1.0, max(0.0, y)))


def bytes_of_ciphertext_ckks(ct: ts.CKKSVector) -> int:
    # TenSEAL serializes to bytes; size reflects ciphertext footprint.
    return len(ct.serialize())


def make_ckks_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=(40, 21, 21, 40), global_scale=2**40):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree,
        list(coeff_mod_bit_sizes),
    )
    ctx.global_scale = global_scale
    ctx.generate_galois_keys() 
    return ctx



def try_make_bfv_context(poly_modulus_degree=8192, plain_modulus=1032193):
    ctx = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree,
        plain_modulus=plain_modulus,
    )
    ctx.generate_galois_keys()
    return ctx


def ckks_encrypted_dot(enc_x: ts.CKKSVector, w: np.ndarray) -> float:
    if hasattr(enc_x, "dot"):
        enc_dot = enc_x.dot(w.tolist() if isinstance(w, np.ndarray) else w)
        return float(enc_dot.decrypt()[0])

    enc_prod = enc_x * w
    return float(np.sum(enc_prod.decrypt()))



def run():
    # ---------------------------
    # Dataset (offline): sklearn digits (8x8 images -> 64 features)
    # We'll do binary classification: digit 0 vs 1 (privacy-friendly toy)
    # ---------------------------
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(int)

    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Standardize features (important for LR stability; CKKS works with floats)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ---------------------------
    # Plaintext Logistic Regression baseline
    # ---------------------------
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X_train_s, y_train)

    w = lr.coef_.reshape(-1).astype(np.float64)  # shape (d,)
    b = float(lr.intercept_[0])

    y_pred_plain = lr.predict(X_test_s)
    acc_plain = accuracy_score(y_test, y_pred_plain)

    # Plaintext latency/throughput micro-benchmark
    n_runs = min(1000, len(X_test_s))
    t0 = time.perf_counter()
    for i in range(n_runs):
        _ = lr.predict(X_test_s[i:i+1])
    t1 = time.perf_counter()
    plain_latency_ms = (t1 - t0) / n_runs * 1000.0
    plain_throughput = n_runs / (t1 - t0)

    # ---------------------------
    # CKKS encrypted inference
    # ---------------------------
    ctx_ckks = make_ckks_context()
    # Encrypt some samples and benchmark
    n_enc = min(200, len(X_test_s))  # keep it reasonable
    enc_inputs = []
    ct_sizes = []
    for i in range(n_enc):
        enc_x = ts.ckks_vector(ctx_ckks, X_test_s[i].astype(np.float64))
        enc_inputs.append(enc_x)
        ct_sizes.append(bytes_of_ciphertext_ckks(enc_x))

    avg_ct_size_bytes = float(np.mean(ct_sizes))

    # Encrypted inference: compute z = w^T x + b under HE, then approx sigmoid in plaintext
    t0 = time.perf_counter()
    y_pred_ckks = []
    for i in range(n_enc):
        z = ckks_encrypted_dot(enc_inputs[i], w) + b
        p = approx_sigmoid_poly1(z)
        y_hat = 1 if p >= 0.5 else 0
        y_pred_ckks.append(y_hat)
    t1 = time.perf_counter()

    ckks_latency_ms = (t1 - t0) / n_enc * 1000.0
    ckks_throughput = n_enc / (t1 - t0)
    acc_ckks = accuracy_score(y_test[:n_enc], np.array(y_pred_ckks))

    # ---------------------------
    # Optional BFV (integer arithmetic) – best effort
    # Many HE setups need careful modulus selection and integer encoding.
    # We'll try a simple quantization approach.
    # ---------------------------
    bfv_available = True
    try:
        ctx_bfv = try_make_bfv_context()
    except Exception as e:
        bfv_available = False
        ctx_bfv = None
        bfv_error = str(e)

    bfv_metrics = None
    if bfv_available:
        # Quantize standardized features to small ints
        # (This is a toy approach; for publishable BFV, document quantization carefully.)
        scale_q = 50.0
        Xq = np.round(X_test_s[:n_enc] * scale_q).astype(np.int64)
        wq = np.round(w * scale_q).astype(np.int64)

        # Encrypt inputs as BFV vectors
        enc_bfv = []
        ct_sizes_bfv = []
        for i in range(n_enc):
            v = Xq[i].tolist()
            encv = ts.bfv_vector(ctx_bfv, v)
            enc_bfv.append(encv)
            ct_sizes_bfv.append(len(encv.serialize()))
        avg_ct_size_bfv = float(np.mean(ct_sizes_bfv))

        # Encrypted dot with BFV: elementwise multiply then sum slots
        # TenSEAL BFVVector supports rotate/add similarly.
        def bfv_dot_fallback_decrypt_sum(enc_vec, w_list):
            # Multiply homomorphically, then decrypt and sum in plaintext
            prod = enc_vec * w_list
            dec = prod.decrypt()
            return int(np.sum(dec))

        t0 = time.perf_counter()
        y_pred_bfv = []
        for i in range(n_enc):
            s = bfv_dot_fallback_decrypt_sum(enc_bfv[i], wq.tolist())
            # Recover approximate z: because of scaling, z approx = s/(scale_q^2) + b
            z = float(s) / (scale_q * scale_q) + b
            p = approx_sigmoid_poly1(z)
            y_hat = 1 if p >= 0.5 else 0
            y_pred_bfv.append(y_hat)
        t1 = time.perf_counter()

        bfv_latency_ms = (t1 - t0) / n_enc * 1000.0
        bfv_throughput = n_enc / (t1 - t0)
        acc_bfv = accuracy_score(y_test[:n_enc], np.array(y_pred_bfv))

        bfv_metrics = {
            "scheme": "BFV",
            "model": "LogReg (0 vs 1)",
            "dataset": "sklearn digits (64 feats)",
            "n_test_samples": n_enc,
            "accuracy": acc_bfv,
            "avg_ciphertext_size_bytes": avg_ct_size_bfv,
            "latency_ms_per_sample": bfv_latency_ms,
            "throughput_samples_per_sec": bfv_throughput,
        }

    # ---------------------------
    # Collect results
    # ---------------------------
    system_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }

    results = [
        {
            "scheme": "PLAINTEXT",
            "model": "LogReg (0 vs 1)",
            "dataset": "sklearn digits (64 feats)",
            "n_test_samples": n_runs,
            "accuracy": acc_plain,
            "avg_ciphertext_size_bytes": 0,
            "latency_ms_per_sample": plain_latency_ms,
            "throughput_samples_per_sec": plain_throughput,
        },
        {
            "scheme": "CKKS",
            "model": "LogReg (0 vs 1) + poly sigmoid",
            "dataset": "sklearn digits (64 feats)",
            "n_test_samples": n_enc,
            "accuracy": acc_ckks,
            "avg_ciphertext_size_bytes": avg_ct_size_bytes,
            "latency_ms_per_sample": ckks_latency_ms,
            "throughput_samples_per_sec": ckks_throughput,
        },
    ]
    if bfv_metrics is not None:
        results.append(bfv_metrics)

    df = pd.DataFrame(results)

    # Save outputs for paper tables
    df.to_csv("results_section5_lr.csv", index=False)

    # Also save system info for reproducibility
    pd.DataFrame([system_info]).to_csv("system_info.csv", index=False)

    print("\n=== System Info ===")
    print(system_info)

    print("\n=== Section 5 Results (LogReg) ===")
    print(df.to_string(index=False))

    if not bfv_available:
        print("\n[INFO] BFV benchmark was skipped due to context init error:")
        print(bfv_error)

    print("\nSaved: results_section5_lr.csv and system_info.csv")


if __name__ == "__main__":
    run()
