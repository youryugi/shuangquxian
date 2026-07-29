# -*- coding: utf-8 -*-
"""
雷达降雨残差校正原型 (Residual correction of radar QPE with sparse gauges)
=========================================================================
流程: 雷达基准场 -> 站点残差 (gauge - radar) -> 用空间协变量训练残差模型
      -> 全域残差场 -> 叠加回基准场

三个模型对比:
  1. OK   : Ordinary Kriging of residuals (经典 conditional merging 的核心)
  2. RF   : Random Forest on covariates
  3. MLP  : 神经网络 + k 近邻残差特征 (学习非平稳、地形感知的插值)

验证协议: 留一站交叉验证 (LOSO) x 按事件划分, 杜绝空间/时间泄漏。
指标: RMSE / MAE / Bias 整体 + 强降雨分位 (top 10%) 单独报告。

换真实数据时只需替换 make_synthetic_dataset() -> 返回同样结构的 DataFrame。
依赖: numpy pandas scikit-learn scipy torch
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import torch
import torch.nn as nn
torch.set_num_threads(2)

RNG = np.random.default_rng(42)
torch.manual_seed(42)


# ----------------------------------------------------------------------
# 1. 合成数据: 山区地形 + 依赖高程/波束遮挡的雷达偏差
#    真实数据到手后, 把这一节替换成 XRAIN + AMeDAS 的加载即可
# ----------------------------------------------------------------------

def make_terrain(n=64, extent_km=30.0):
    """生成合成 DEM (几座山脊叠加), 返回格点坐标与高程/坡度/坡向."""
    xs = np.linspace(0, extent_km, n)
    X, Y = np.meshgrid(xs, xs)
    Z = np.zeros_like(X)
    ridges = [(8, 20, 900, 5), (20, 10, 1200, 6), (24, 24, 700, 4)]
    for cx, cy, h, w in ridges:
        Z += h * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)))
    Z += 120 * RNG.standard_normal(Z.shape) * 0.05  # 小尺度粗糙度
    gy, gx = np.gradient(Z, xs, xs)
    slope = np.degrees(np.arctan(np.hypot(gx, gy) / 1000.0))
    aspect = np.arctan2(-gx, gy)  # 弧度
    return X, Y, Z, slope, aspect


def beam_blockage(X, Y, Z, radar_xy=(0.0, 0.0), beam_h0=300.0):
    """极简波束遮挡代理: 沿雷达->格点视线, 地形超过波束高度的比例.
    真实研究中用 DEM + 波束几何精确计算 (e.g., wradlib.beamblockage)."""
    rx, ry = radar_xy
    n = X.shape[0]
    bb = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            steps = 24
            ts = np.linspace(0.05, 1.0, steps)
            px, py = rx + ts * (X[i, j] - rx), ry + ts * (Y[i, j] - ry)
            dist = np.hypot(px - rx, py - ry)
            beam_h = beam_h0 + dist * 18.0  # 波束随距离抬升(近似)
            terr = _bilinear(Z, X, Y, px, py)
            bb[i, j] = np.mean(terr > beam_h)
    return bb


def _bilinear(Z, X, Y, px, py):
    xs, ys = X[0, :], Y[:, 0]
    ix = np.clip(np.searchsorted(xs, px) - 1, 0, len(xs) - 2)
    iy = np.clip(np.searchsorted(ys, py) - 1, 0, len(ys) - 2)
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = (px - x0) / (x1 - x0)
    ty = (py - y0) / (y1 - y0)
    return ((1 - tx) * (1 - ty) * Z[iy, ix] + tx * (1 - ty) * Z[iy, ix + 1]
            + (1 - tx) * ty * Z[iy + 1, ix] + tx * ty * Z[iy + 1, ix + 1])


def make_synthetic_dataset(n_stations=30, n_events=30, n_steps_per_event=8):
    """
    返回 long-format DataFrame, 每行 = (station, time) 一个样本:
      station_id, event_id, x, y, elev, slope, aspect_sin, aspect_cos,
      blockage, dist_radar, radar (基准场值), gauge (真值), residual
    以及格点协变量表 grid_df (用于全域出图).
    """
    X, Y, Z, slope, aspect = make_terrain()
    bb = beam_blockage(X, Y, Z)
    dist_r = np.hypot(X, Y)

    # 站点: 刻意模仿现实 —— 大多沿河谷(低海拔)布设, 高海拔稀疏
    flat = np.argsort(Z.ravel())
    low_pool = flat[: int(0.5 * flat.size)]
    high_pool = flat[int(0.7 * flat.size):]
    idx = np.concatenate([
        RNG.choice(low_pool, int(n_stations * 0.8), replace=False),
        RNG.choice(high_pool, n_stations - int(n_stations * 0.8), replace=False),
    ])
    si, sj = np.unravel_index(idx, Z.shape)

    rows = []
    for ev in range(n_events):
        etype = RNG.choice(["stratiform", "convective", "typhoon"],
                           p=[0.45, 0.35, 0.20])
        base = {"stratiform": 4.0, "convective": 10.0, "typhoon": 18.0}[etype]
        # 事件内的真实降雨场: 平滑场 + 地形增雨 (orographic enhancement)
        for t in range(n_steps_per_event):
            cx, cy = RNG.uniform(5, 25, 2)
            w = RNG.uniform(6, 14)
            field = base * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)))
            field *= (1.0 + 0.0006 * Z)                    # 高程增雨
            field += RNG.gamma(1.2, 0.6, size=field.shape)  # 小尺度噪声
            truth = np.clip(field, 0, None)

            # 雷达观测 = 真值 x 系统性偏差 x 随机误差
            #   遮挡越强低估越多; 高海拔波束过冲也低估; 乘性噪声
            bias = (1 - 0.55 * bb) * (1 - 0.00025 * Z)
            radar = truth * bias * RNG.lognormal(0, 0.15, size=truth.shape)

            for k in range(len(si)):
                i, j = si[k], sj[k]
                g = truth[i, j] * RNG.lognormal(0, 0.05)  # 雨量计自身小误差
                rows.append(dict(
                    station_id=k, event_id=ev, event_type=etype, t=t,
                    x=X[i, j], y=Y[i, j], elev=Z[i, j], slope=slope[i, j],
                    aspect_sin=np.sin(aspect[i, j]),
                    aspect_cos=np.cos(aspect[i, j]),
                    blockage=bb[i, j], dist_radar=dist_r[i, j],
                    radar=radar[i, j], gauge=g, residual=g - radar[i, j],
                ))
    df = pd.DataFrame(rows)

    grid_df = pd.DataFrame(dict(
        x=X.ravel(), y=Y.ravel(), elev=Z.ravel(), slope=slope.ravel(),
        aspect_sin=np.sin(aspect).ravel(), aspect_cos=np.cos(aspect).ravel(),
        blockage=bb.ravel(), dist_radar=dist_r.ravel(),
    ))
    return df, grid_df


# ----------------------------------------------------------------------
# 2. 模型
# ----------------------------------------------------------------------

COVARS = ["x", "y", "elev", "slope", "aspect_sin", "aspect_cos",
          "blockage", "dist_radar", "radar"]


def ok_predict(train_xy, train_res, test_xy, rng_km=12.0, nugget=0.1):
    """Ordinary Kriging (指数变差函数, 参数固定的简化版)."""
    d_tt = cdist(train_xy, train_xy)
    d_pt = cdist(test_xy, train_xy)
    sill = np.var(train_res) + 1e-6

    def cov(d):
        return sill * np.exp(-d / rng_km)

    n = len(train_xy)
    K = np.empty((n + 1, n + 1))
    K[:n, :n] = cov(d_tt) + nugget * sill * np.eye(n)
    K[n, :], K[:, n], K[n, n] = 1.0, 1.0, 0.0
    preds = np.empty(len(test_xy))
    for m in range(len(test_xy)):
        k = np.append(cov(d_pt[m]), 1.0)
        w = solve(K, k)
        preds[m] = w[:n] @ train_res
    return preds


def add_neighbor_features(df_target, df_obs, k=5):
    """给每个目标样本拼上同一时刻 k 个最近观测站的 (残差, 距离, 高程差)."""
    out = []
    for (ev, t), grp_t in df_target.groupby(["event_id", "t"], sort=False):
        obs = df_obs[(df_obs.event_id == ev) & (df_obs.t == t)]
        oxy = obs[["x", "y"]].values
        txy = grp_t[["x", "y"]].values
        d = cdist(txy, oxy)
        nn = np.argsort(d, axis=1)[:, :k]
        feats = {}
        for q in range(k):
            cols_idx = nn[:, q]
            feats[f"nb{q}_res"] = obs["residual"].values[cols_idx]
            feats[f"nb{q}_dist"] = d[np.arange(len(txy)), cols_idx]
            feats[f"nb{q}_delev"] = (grp_t["elev"].values
                                     - obs["elev"].values[cols_idx])
        out.append(grp_t.assign(**feats))
    return pd.concat(out).sort_index()


class ResidualMLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, epochs=30, lr=2e-3, batch=1024):
    model = ResidualMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    loss_fn = nn.SmoothL1Loss()  # 对残差长尾更稳
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


# ----------------------------------------------------------------------
# 3. 评价: 留一站 LOSO x 事件划分
# ----------------------------------------------------------------------

@dataclass
class Scores:
    rmse: float
    mae: float
    bias: float
    rmse_heavy: float  # gauge 前 10% 强降雨样本

    @staticmethod
    def compute(y_true, y_pred, gauge):
        e = y_pred - y_true
        thr = np.quantile(gauge, 0.9)
        heavy = gauge >= thr
        return Scores(
            rmse=float(np.sqrt(np.mean(e ** 2))),
            mae=float(np.mean(np.abs(e))),
            bias=float(np.mean(e)),
            rmse_heavy=float(np.sqrt(np.mean(e[heavy] ** 2))),
        )


def loso_evaluate(df, k_nb=5, test_event_frac=0.3):
    """对每个站: 其余站为训练, 该站为测试; 事件也切成 train/test 两半.
    评价对象是 '校正后的降雨' = radar + predicted_residual."""
    events = df.event_id.unique()
    RNG.shuffle(events)
    test_events = set(events[: int(len(events) * test_event_frac)])
    is_test_ev = df.event_id.isin(test_events)

    results = {m: {"y": [], "p": [], "g": []}
               for m in ["radar_raw", "OK", "RF", "MLP"]}

    for sid in sorted(df.station_id.unique()):
        print(f"fold {sid}", flush=True)
        tr = df[(df.station_id != sid) & (~is_test_ev)]
        te = df[(df.station_id == sid) & (is_test_ev)]
        if len(te) == 0:
            continue

        # --- baseline 0: 不校正 ---
        results["radar_raw"]["y"].append(te.gauge.values)
        results["radar_raw"]["p"].append(te.radar.values)
        results["radar_raw"]["g"].append(te.gauge.values)

        # --- OK: 用测试时刻其他站的残差做克里金 (真实可用信息) ---
        ok_preds = []
        te_sorted = te.sort_values(["event_id", "t"])
        for (ev, t), grp in te_sorted.groupby(["event_id", "t"], sort=False):
            obs = df[(df.event_id == ev) & (df.t == t)
                     & (df.station_id != sid)]
            ok_preds.append(ok_predict(obs[["x", "y"]].values,
                                       obs.residual.values,
                                       grp[["x", "y"]].values))
        ok_res = np.concatenate(ok_preds)
        results["OK"]["y"].append(te_sorted.gauge.values)
        results["OK"]["p"].append(te_sorted.radar.values + ok_res)
        results["OK"]["g"].append(te_sorted.gauge.values)

        # --- 拼邻居特征 (MLP / RF 共用输入协议) ---
        obs_pool_tr = df[(df.station_id != sid) & (~is_test_ev)]
        obs_pool_te = df[(df.station_id != sid) & (is_test_ev)]
        tr_f = add_neighbor_features(tr, obs_pool_tr, k=k_nb)
        te_f = add_neighbor_features(te, obs_pool_te, k=k_nb)
        nb_cols = [c for c in tr_f.columns if c.startswith("nb")]
        feat_cols = COVARS + nb_cols

        scaler = StandardScaler().fit(tr_f[feat_cols].values)
        Xtr = scaler.transform(tr_f[feat_cols].values)
        Xte = scaler.transform(te_f[feat_cols].values)
        ytr = tr_f.residual.values

        # --- RF ---
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                   n_jobs=-1, random_state=0).fit(Xtr, ytr)
        results["RF"]["y"].append(te_f.gauge.values)
        results["RF"]["p"].append(te_f.radar.values + rf.predict(Xte))
        results["RF"]["g"].append(te_f.gauge.values)

        # --- MLP ---
        mlp = train_mlp(Xtr, ytr)
        with torch.no_grad():
            mres = mlp(torch.tensor(Xte, dtype=torch.float32)).numpy()
        results["MLP"]["y"].append(te_f.gauge.values)
        results["MLP"]["p"].append(te_f.radar.values + mres)
        results["MLP"]["g"].append(te_f.gauge.values)

    print(f"\n{'model':<10}{'RMSE':>8}{'MAE':>8}{'Bias':>8}{'RMSE(暴雨top10%)':>20}")
    print("-" * 56)
    table = {}
    for m, d in results.items():
        y = np.concatenate(d["y"]); p = np.concatenate(d["p"])
        g = np.concatenate(d["g"])
        s = Scores.compute(y, p, g)
        table[m] = s
        print(f"{m:<10}{s.rmse:>8.3f}{s.mae:>8.3f}{s.bias:>8.3f}"
              f"{s.rmse_heavy:>20.3f}")
    return table


# ----------------------------------------------------------------------
# 4. 全域校正场生成 (拿最终模型对格点出图用)
# ----------------------------------------------------------------------

def correct_full_field(df, grid_df, k_nb=5):
    """用全部站点训练 MLP, 对格点生成残差场 (radar 格点值需真实数据提供;
    合成演示中略去 radar 列, 这里仅示范接口)."""
    obs = df
    tr_f = add_neighbor_features(df, obs, k=k_nb)
    nb_cols = [c for c in tr_f.columns if c.startswith("nb")]
    feat_cols = COVARS + nb_cols
    scaler = StandardScaler().fit(tr_f[feat_cols].values)
    mlp = train_mlp(scaler.transform(tr_f[feat_cols].values),
                    tr_f.residual.values)
    return mlp, scaler, feat_cols


if __name__ == "__main__":
    print("生成合成数据 (含地形、波束遮挡、高程依赖的雷达偏差)...")
    df, grid_df = make_synthetic_dataset()
    print(f"样本量: {len(df)}  站点: {df.station_id.nunique()}  "
          f"事件: {df.event_id.nunique()}")
    print(f"残差统计: mean={df.residual.mean():.3f}  "
          f"std={df.residual.std():.3f}  (系统性低估 => mean > 0)")
    loso_evaluate(df)
