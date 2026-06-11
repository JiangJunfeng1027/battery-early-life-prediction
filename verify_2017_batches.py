"""
2017 原始批次基线验证（论文补充实验 · strictly additive）
=========================================================
目的：在 Severson 原研究的 Batch 1 (2017-05-12) + Batch 2 (2017-06-30) 上
     复跑 discharge-feature ElasticNet 基线，验证 workflow 在原始批次上的表现。

用法：
1. 从 https://data.matr.io/1/  (项目: "Data-driven prediction of battery cycle life
   before capacity degradation") 下载以下两个文件，放进 ./data/MIT/ ：
     2017-05-12_batchdata_updated_struct_errorcorrect.mat
     2017-06-30_batchdata_updated_struct_errorcorrect.mat
2. 在项目根目录运行:  python verify_2017_batches.py
3. Use the final printed table for dissertation reporting.

输出：样本账目（raw→exclusion→continuation→eligible）+ 四个基线模型的
     repeated 3-fold ×10 MAPE/RMSE（与论文 Table 4.2 同协议）。
转换结果缓存到 ./output_2017/，二次运行免重转。
"""
import os, pickle, warnings
import numpy as np
import h5py
warnings.filterwarnings("ignore")

DATA_DIR = "./data/MIT/"
OUT_DIR = "./output_2017/"
os.makedirs(OUT_DIR, exist_ok=True)

MAT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
}

# ---------- 1) 加载（与 step1_load_and_explore_final.py 同逻辑） ----------
def load_batch(path, prefix):
    print(f"Loading {os.path.basename(path)} ...")
    f = h5py.File(path, "r"); batch = f["batch"]
    n = batch["summary"].shape[0]
    out = {}
    for i in range(n):
        print(f"  cell {i+1}/{n}", end="\r")
        try:
            cl = f[batch["cycle_life"][i, 0]][()].flatten()
        except Exception:
            cl = np.array([np.nan])
        try:
            policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()
        except Exception:
            policy = "unknown"
        sg = f[batch["summary"][i, 0]]
        summary = {}
        for mf, of in {"IR":"IR","QCharge":"QC","QDischarge":"QD","Tavg":"Tavg",
                        "Tmin":"Tmin","Tmax":"Tmax","chargetime":"chargetime","cycle":"cycle"}.items():
            try: summary[of] = sg[mf][0, :]
            except Exception: summary[of] = np.array([])
        cg = f[batch["cycles"][i, 0]]
        cyc = {}
        ncy = cg["I"].shape[0]
        for j in range(ncy):
            cd = {}
            for mf, of in {"Qdlin":"Qdlin","Qd":"Qd","V":"V","T":"T","t":"t","I":"I"}.items():
                try:
                    v = cg[mf][j, 0]
                    if isinstance(v, h5py.h5r.Reference): v = f[v][()].flatten()
                    cd[of] = np.atleast_1d(v)
                except Exception: pass
            cyc[str(j)] = cd
        out[f"{prefix}{i}"] = {"cycle_life": cl, "charge_policy": policy,
                               "summary": summary, "cycles": cyc}
    f.close(); print(f"\n  loaded {n} cells")
    return out, n

# ---------- 2) 剔除 + continuation 合并（官方逻辑，在2017批次上真正生效） ----------
def merge(all_b, raw_counts):
    b1 = all_b["batch1"].copy(); b2 = all_b["batch2"].copy()
    bad1 = ["b1c8","b1c10","b1c12","b1c13","b1c22"]
    n_bad1 = sum(1 for k in bad1 if k in b1)
    for k in bad1: b1.pop(k, None)
    b2k = ["b2c7","b2c8","b2c9","b2c15","b2c16"]
    b1k = ["b1c0","b1c1","b1c2","b1c3","b1c4"]
    add_len = [662, 981, 1060, 208, 482]
    merged = 0
    for i, bk in enumerate(b1k):
        if bk in b1 and b2k[i] in b2:
            b1[bk]["cycle_life"] = b1[bk]["cycle_life"] + add_len[i]
            for j in b1[bk]["summary"]:
                if j == "cycle":
                    b1[bk]["summary"][j] = np.hstack((b1[bk]["summary"][j],
                        b2[b2k[i]]["summary"][j] + len(b1[bk]["summary"][j])))
                else:
                    b1[bk]["summary"][j] = np.hstack((b1[bk]["summary"][j], b2[b2k[i]]["summary"][j]))
            last = len(b1[bk]["cycles"])
            for ji, jk in enumerate(b2[b2k[i]]["cycles"]):
                b1[bk]["cycles"][str(last + ji)] = b2[b2k[i]]["cycles"][jk]
            b2.pop(b2k[i], None); merged += 1
    bat = {**b1, **b2}
    print(f"\n=== 样本账目 ===")
    print(f"raw: batch1={raw_counts['batch1']}  batch2={raw_counts['batch2']}  合计={sum(raw_counts.values())}")
    print(f"batch1 剔除清单生效: -{n_bad1}   continuation 合并(b2→b1): {merged} 对")
    print(f"合并后 cells: {len(bat)}")
    return bat

# ---------- 3) 22特征提取（与 step2 同逻辑；这里只需基线所用子集+全集） ----------
def features(bat):
    rows = {}
    drop_nan, drop_short = 0, 0
    for k, c in bat.items():
        cl = c["cycle_life"].flatten()[0]
        if np.isnan(cl): drop_nan += 1; continue
        if len(c["cycles"]) < 101: drop_short += 1; continue
        try:
            q100 = np.asarray(c["cycles"]["99"]["Qdlin"]).flatten()
            q10  = np.asarray(c["cycles"]["9"]["Qdlin"]).flatten()
            m = min(len(q100), len(q10))
            if m == 0: drop_short += 1; continue
            dq = q100[:m] - q10[:m]
            f = {}
            f["dq_variance"] = np.log10(np.abs(np.var(dq)) + 1e-12)
            f["dq_minimum"] = np.min(dq); f["dq_mean"] = np.mean(dq)
            from scipy.stats import skew, kurtosis
            f["dq_skewness"] = skew(dq); f["dq_kurtosis"] = kurtosis(dq)
            f["dq_max_minus_min"] = np.max(dq) - np.min(dq)
            qd = np.asarray(c["summary"]["QD"]).flatten()
            if len(qd) < 100: drop_short += 1; continue
            f["qd_cycle2"] = qd[1]; f["qd_cycle100"] = qd[99]
            f["qd_ratio_100_2"] = qd[99]/qd[1] if qd[1] else np.nan
            x = np.arange(91, 101); s, b = np.polyfit(x, qd[90:100], 1)
            f["qd_slope_91_100"] = s; f["qd_intercept_91_100"] = b
            f["qd_slope_2_100"] = np.polyfit(np.arange(2, 101), qd[1:100], 1)[0]
            for src, nm in [("chargetime","chargetime"), ("IR","ir")]:
                a = np.asarray(c["summary"][src]).flatten()
                f[f"{nm}_cycle2"] = a[1] if len(a) > 1 else np.nan
                f[f"{nm}_cycle100"] = a[99] if len(a) > 99 else np.nan
            f["ir_diff_100_2"] = f["ir_cycle100"] - f["ir_cycle2"]
            for src, nm in [("Tavg","tavg"), ("Tmin","tmin"), ("Tmax","tmax")]:
                a = np.asarray(c["summary"][src]).flatten()[:100]
                f[f"{nm}_first100"] = (np.mean(a) if nm=="tavg" else (np.min(a) if nm=="tmin" else np.max(a))) if len(a) else np.nan
            ta = np.asarray(c["summary"]["Tavg"]).flatten()[:100]
            f["tavg_slope"] = np.polyfit(np.arange(len(ta)), ta, 1)[0] if len(ta) > 2 else np.nan
            try:
                q5 = np.asarray(c["cycles"]["4"]["Qdlin"]).flatten()
                q4 = np.asarray(c["cycles"]["3"]["Qdlin"]).flatten()
                mm = min(len(q5), len(q4))
                f["dq_5_4_variance"] = np.log10(np.abs(np.var(q5[:mm]-q4[:mm])) + 1e-12)
            except Exception:
                f["dq_5_4_variance"] = np.nan
            f["cycle_life"] = float(cl)
            rows[k] = f
        except Exception:
            drop_short += 1
    print(f"NaN寿命剔除: -{drop_nan}   不足101周期/数据缺失剔除: -{drop_short}")
    print(f"合格特征行: {len(rows)}")
    import pandas as pd
    return pd.DataFrame(rows).T

# ---------- 4) 基线（repeated 3-fold ×10，log目标，fold内标准化；同论文协议） ----------
def baseline(df):
    import pandas as pd
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedKFold
    y = np.log10(df["cycle_life"].astype(float).values)
    ycl = df["cycle_life"].astype(float).values
    sets = {
        "Naive mean": None,
        "ENet (variance only)": ["dq_variance"],
        "ENet (discharge feats)": ["dq_variance","dq_minimum","dq_mean","dq_skewness","dq_kurtosis",
                                   "dq_max_minus_min","qd_cycle2","qd_cycle100","qd_ratio_100_2",
                                   "qd_slope_91_100","qd_intercept_91_100","qd_slope_2_100"],
        "ENet (full 22)": [c for c in df.columns if c != "cycle_life"],
    }
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=42)
    print(f"\n=== 基线结果（n={len(df)}，repeated 3-fold ×10）===")
    print(f"{'Model':26s} {'MAPE%':>8s} {'RMSE(cyc)':>10s}")
    for name, cols in sets.items():
        mape, rmse = [], []
        for tr, te in rkf.split(df):
            if cols is None:
                pred = np.full(len(te), ycl[tr].mean())
            else:
                X = df[cols].astype(float).fillna(0).values
                sc = StandardScaler().fit(X[tr])
                m = ElasticNetCV(l1_ratio=[.1,.5,.7,.9], max_iter=10000, cv=3, random_state=42)
                m.fit(sc.transform(X[tr]), y[tr])
                pred = 10 ** m.predict(sc.transform(X[te]))
            mape.append(np.mean(np.abs(pred - ycl[te]) / ycl[te]) * 100)
            rmse.append(np.sqrt(np.mean((pred - ycl[te]) ** 2)))
        print(f"{name:26s} {np.mean(mape):8.1f} {np.mean(rmse):10.0f}")
    print("\n→ Use the printed block above for dissertation supplementary reporting and presentation updates.")

if __name__ == "__main__":
    pkl = os.path.join(OUT_DIR, "bat_2017.pkl")
    if os.path.exists(pkl):
        print("发现缓存，跳过转换"); bat = pickle.load(open(pkl, "rb"))
    else:
        all_b, raw = {}, {}
        for bn, mf in MAT_FILES.items():
            p = os.path.join(DATA_DIR, mf)
            assert os.path.exists(p), f"缺文件: {p}（请先下载，见文件头说明）"
            all_b[bn], raw[bn] = load_batch(p, bn.replace("batch", "b") + "c")
        bat = merge(all_b, raw)
        pickle.dump(bat, open(pkl, "wb"), protocol=4)
        print(f"已缓存 → {pkl}")
    df = features(bat)
    df.to_csv(os.path.join(OUT_DIR, "feature_matrix_2017.csv"))
    baseline(df)
