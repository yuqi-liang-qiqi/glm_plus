import pandas as pd
import numpy as np
import os as _os, sys as _sys
from typing import Dict, Tuple
import warnings
import hashlib
try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
except Exception:
    raise RuntimeError("需要安装 statsmodels>=0.13 才能运行 QuantReg（pip install statsmodels）。")

# 确保可以以绝对方式导入 `glm_plus`（把项目根目录加入 sys.path）
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
from glm_plus.frequentist.torque import FrequentistOQR
from matplotlib import pyplot as plt

# ==================== 常量与配置 ====================
# 分位点集合：细化到 0.1..0.9（含 9 个点），便于 granularity 分析
# QUANTILES_MAIN = (0.2, 0.5, 0.8)
QUANTILES_MAIN = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ORDER_LABELS = ["Assistant/Junior", "Regular", "Leader", "Chief/Founder"]

# Granularity 带区与对比对设置
BANDS_DEF = {
    'low': (0.1, 0.2),
    'low_mid': (0.3, 0.4),
    'mid': (0.5,),
    'mid_high': (0.6, 0.7),
    'high': (0.8, 0.9),
}
COMPARE_PAIRS = (
    ('mid', 'low'),
    ('mid', 'high'),
    ('low', 'high'),
)

# 估计对象（estimand）与二阶段开关
# - 'single_index': 关闭 two-index，只在 h(Y) 尺子上抽系数（解释最直观）
# - 'final_combo': 保持 two-index，一律抽“最终组合”u_tau = Xβ1 + g^{-1}(Xβ2,τ) 的线性近似系数
ESTIMAND_MODE = 'single_index'  # 可改为 'final_combo'
# Bootstrap 次数（200 已足够稳健；如需更稳可改 500）
N_BOOTSTRAP_DEFAULT = 200
# QuantReg 容差（提高收敛稳定性）
QR_P_TOL = 1e-6

# —— 效应量阈值（避免“显著但无意义”）——
# 只有当 |Δ| ≥ ε 且 CI 不跨 0 才将 Δ 判为 sticky/glass
# ε 可按 h(Y) 的 SD 或 IQR 的比例设定
EFFECT_SIZE_MODE = 'sd_pct'   # 可选: 'sd_pct' 或 'iqr_pct'
EPSILON_SD_FRAC = 0.02        # ε = 2% * sd(h(Y))
EPSILON_IQR_FRAC = 0.01       # ε = 1% * IQR(h(Y))

# —— 可选：置换检验（默认关闭，设为 >0 开启，例如 500）——
N_PERMUTATION_DEFAULT = 0

# —— 概率差阈值（可解释量纲）：至少 2 个百分点 ——
PROB_DIFF_THRESHOLD = 0.02

# 是否在保存后显示图窗（在终端运行时会阻塞），默认不显示
SHOW_FIG = False

# 简单进度日志函数（按步骤打印，便于初学者理解）
def _log(msg: str) -> None:
    print(f"[Progress] {msg}")

# subsample 规则：大样本下加速拟合；不改变估计含义
SUBSAMPLE_RULE = lambda n: (5000 if n > 20000 else (3000 if n > 3000 else None))

# 输出目录
import os
OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 统一拟合函数（保持与你现有 pipeline 一致的配置）
def fit_model(X: np.ndarray, y: np.ndarray, taus=QUANTILES_MAIN, random_state: int = 0) -> FrequentistOQR:
    # 最小改法：每国只拟合一次；固定单指数；轻量网格以提速
    subsample_dynamic = SUBSAMPLE_RULE(X.shape[0]) or 2000
    m = FrequentistOQR(
        quantiles=taus,
        use_two_index=False,
        auto_select_k=True,
        subsample_n=int(subsample_dynamic),
        rank_grid_n=25,
        t_grid_n=31,
        random_state=random_state,
        qr_p_tol=QR_P_TOL,
    )
    m.fit(X, y)
    return m

# ==================== Step 1: 读取数据 ====================
_log("[1/3] 读取数据并清洗类别/哑元...")
_DATA_DIR = "/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_data/real_data_my_paper/"
_CSV_PATH = _os.path.join(_DATA_DIR, "df_seniority.csv")
assert _os.path.exists(_CSV_PATH), f"找不到数据文件: {_CSV_PATH}"
df_seniority = pd.read_csv(_CSV_PATH)
assert 'country' in df_seniority.columns, "df_seniority 需要包含 'country' 列"

# —— 统一清洗 Y10 标签，避免大小写/空格导致的丢行 ——
_CANON_MAP = {
    "assistant/junior": "Assistant/Junior",
    "regular": "Regular",
    "leader": "Leader",
    "chief/founder": "Chief/Founder",
}

def _canonize_y10(val: str) -> str:
    s = str(val).strip().lower()
    if not s:
        return s
    # 归一化分隔符与空格
    for ch in ["&", "and", "-", "_"]:
        s = s.replace(ch, "/")
    s = "/".join([t.strip() for t in s.split("/") if t.strip()])
    # 关键词映射
    if "assistant" in s or "junior" in s:
        key = "assistant/junior"
    elif "regular" in s:
        key = "regular"
    elif "leader" in s:
        key = "leader"
    elif "chief" in s or "founder" in s:
        key = "chief/founder"
    else:
        # 未识别则原样返回（后续将被丢弃，且打印提示）
        return str(val).strip()
    return _CANON_MAP[key]

df_seniority['Y10'] = df_seniority['Y10'].apply(_canonize_y10)
unknown_labels = sorted(set(df_seniority['Y10']) - set(ORDER_LABELS))
if unknown_labels:
    print(f"[warn] Y10 中存在未识别标签，将被丢弃: {unknown_labels}")
    vc = df_seniority['Y10'].value_counts()
    unk_counts = {lab: int(vc.get(lab, 0)) for lab in unknown_labels}
    print(f"[warn] 未识别标签的样本量: {unk_counts}")
print("[info] 已知标签样本量:")
print(df_seniority['Y10'].value_counts().reindex(ORDER_LABELS).fillna(0).astype(int))

# —— 预建立全局哑元列集合，确保各国设计矩阵列一致 ——
_CAT_COLS = ['highest_educational_degree', 'internationalization', 'company_size']
global_cat_dummies = pd.get_dummies(df_seniority[_CAT_COLS], columns=_CAT_COLS, drop_first=True, dtype=int)
GLOBAL_DUMMY_COLUMNS = list(global_cat_dummies.columns)

_log("[1/3] 数据加载与清洗完成")

# 设计矩阵构造（小白友好：把 y 编码、female 指示变量、类别哑元都在这里做）
def build_design_for_subset(df_sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    cols_needed = [
        'gender', 'highest_educational_degree',
        'whether_bachelor_university_prestigious',
        'internationalization', 'work_years', 'company_size', 'Y10'
    ]
    d = df_sub[cols_needed].dropna().copy()
    # y: 把职位等级转为有序类别，再编码为 1..J
    cat_type = pd.CategoricalDtype(categories=ORDER_LABELS, ordered=True)
    d['y_cat'] = d['Y10'].astype(cat_type)
    d = d[~d['y_cat'].isna()].copy()
    d['y'] = d['y_cat'].cat.codes + 1
    # female: 文本转小写后匹配
    gg = d['gender'].astype(str).str.strip().str.lower()
    d['female'] = (gg == 'female').astype(int)
    # prestigious_bachelor: 统一成 0/1
    col_p = 'whether_bachelor_university_prestigious'
    if str(d[col_p].dtype) == 'bool':
        d['prestigious_bachelor'] = d[col_p].astype(int)
    else:
        d['prestigious_bachelor'] = (
            d[col_p].astype(str).str.strip().str.lower().isin(['true','1','yes','y','t'])
        ).astype(int)
    # work_years: 转数值
    d['work_years'] = pd.to_numeric(d['work_years'], errors='coerce')
    d = d.dropna(subset=['work_years']).copy()
    # one-hot: 类别变量做哑元（drop_first=True 以避免完全共线）
    cat_cols = _CAT_COLS
    X_cat = pd.get_dummies(d[cat_cols], columns=cat_cols, drop_first=True, dtype=int)
    # 统一到全局哑元列
    X_cat = X_cat.reindex(columns=GLOBAL_DUMMY_COLUMNS, fill_value=0)
    X_df = pd.concat([
        d[['female', 'prestigious_bachelor', 'work_years']].reset_index(drop=True),
        X_cat.reset_index(drop=True)
    ], axis=1)
    X = X_df.to_numpy(dtype=float)
    y = d['y'].to_numpy(dtype=int)
    if d['female'].nunique() < 2:
        try:
            _ctry = str(df_sub.get('country', '').iloc[0]) if 'country' in df_sub.columns else ''
        except Exception:
            _ctry = ''
        print(f"[warn] 该子集 female 只有一个取值，分位系数可能不可识别或不稳。country={_ctry}")
    return d, X_df, X, y

# ==================== Step 2: 工具函数（抽系数 + CI + p） ====================
def _extract_female_coef_by_tau(model: FrequentistOQR, X_df: pd.DataFrame, X: np.ndarray, taus, mode: str = ESTIMAND_MODE, warn_counter: Dict[str, int] | None = None) -> Dict[float, float]:
    """
    返回 {tau: female 系数}（在模型当前拟合下）。
    优先使用模型内存储；否则用 h(Y) 的连续响应重新做 QuantReg 兜底。
    """
    coef_by_tau: Dict[float, float] = {}
    female_idx = int(X_df.columns.get_loc('female'))

    # 全局已导入 sm 与 IterationLimitWarning

    # Single-index：直接在 h(Y) 上抽系数
    if mode == 'single_index':
        # 1) 优先尝试：模型内部是否已暴露 per-τ 的 QR 结果
        qr_store = getattr(model, 'qr_models_', None) or getattr(model, 'qr_results_', None)
        if isinstance(qr_store, dict) and len(qr_store) > 0:
            for tau in taus:
                res = qr_store.get(tau, None)
                if res is not None and hasattr(res, 'params'):
                    coef_by_tau[float(tau)] = float(np.asarray(res.params, dtype=float)[female_idx])
        # 2) 兜底：使用 h_iso_ 和 _y_jit 重建连续响应，再对 (hy, X) 做 QuantReg
        missing = [t for t in taus if float(t) not in coef_by_tau]
        if missing:
            hy = None
            try:
                hy = model.h_iso_.transform(getattr(model, '_y_jit'))  # type: ignore[attr-defined]
            except Exception:
                hy = None
            if hy is None:
                for name in ('y_tilde_', 'y_cont_', 'y_transformed_', 'y_star_', 'y_continuous_'):
                    hy = getattr(model, name, None)
                    if hy is not None:
                        break
            if hy is None:
                raise RuntimeError("无法获取模型的连续响应（h(Y)）。可用 dir(model) 查看实际字段名后加入候选列表。")
            hy = np.asarray(hy, dtype=float).reshape(-1)
            X_used = np.asarray(X, dtype=float)
            if X_used.shape[0] != hy.shape[0]:
                X_last = getattr(model, '_last_X_', None)
                if X_last is None or np.asarray(X_last).shape[0] != hy.shape[0]:
                    raise RuntimeError("X 与连续响应长度不匹配，且 model._last_X_ 不可用。")
                X_used = np.asarray(X_last, dtype=float)
            for tau in missing:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", IterationLimitWarning)
                    try:
                        res = sm.QuantReg(hy, X_used).fit(q=float(tau), max_iter=5000, p_tol=QR_P_TOL)
                    except TypeError:
                        res = sm.QuantReg(hy, X_used).fit(q=float(tau), max_iter=5000)
                if warn_counter is not None:
                    warn_counter['fit_count'] = warn_counter.get('fit_count', 0) + 1
                    warn_counter['warn_count'] = warn_counter.get('warn_count', 0) + sum(1 for wi in w if issubclass(wi.category, IterationLimitWarning))
                coef_by_tau[float(tau)] = float(np.asarray(res.params, dtype=float)[female_idx])
        return coef_by_tau

    # final_combo：对 u_tau = Xβ1 + g^{-1}(Xβ2,τ) 做一遍 τ-QuantReg，抽 female 系数
    X_used = np.asarray(X, dtype=float)
    X_last = getattr(model, '_last_X_', None)
    if X_last is not None and X_used.shape[0] != np.asarray(X_last).shape[0]:
        X_used = np.asarray(X_last, dtype=float)
    xb1 = X_used @ (model.beta1_.reshape(-1) if model.beta1_ is not None else np.zeros(X_used.shape[1]))
    for tau in taus:
        u_tau = xb1.copy()
        if getattr(model, 'use_two_index', False) and getattr(model, 'beta2_tau_', None) is not None:
            b2t = model.beta2_tau_.get(float(tau), model.beta2_)
            if b2t is None and isinstance(model.beta2_tau_, dict) and len(model.beta2_tau_) > 0:
                b2t = next(iter(model.beta2_tau_.values()))
            if b2t is not None:
                u2 = (X_used @ b2t.reshape(-1))
                try:
                    r_tau = model._ginv(u2)
                except Exception:
                    r_tau = u2
                u_tau = u_tau + r_tau
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", IterationLimitWarning)
            try:
                res = sm.QuantReg(u_tau, X_used).fit(q=float(tau), max_iter=5000, p_tol=QR_P_TOL)
            except TypeError:
                res = sm.QuantReg(u_tau, X_used).fit(q=float(tau), max_iter=5000)
        if warn_counter is not None:
            warn_counter['fit_count'] = warn_counter.get('fit_count', 0) + 1
            warn_counter['warn_count'] = warn_counter.get('warn_count', 0) + sum(1 for wi in w if issubclass(wi.category, IterationLimitWarning))
        coef_by_tau[float(tau)] = float(np.asarray(res.params, dtype=float)[female_idx])

    return coef_by_tau

def _ci_from_samples(samples, level=0.95):
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, False
    lo = float(np.nanpercentile(arr, (1.0 - level) / 2.0 * 100.0))
    hi = float(np.nanpercentile(arr, (1.0 + level) / 2.0 * 100.0))
    sig = not (lo <= 0.0 <= hi)
    return lo, hi, sig

def _p_from_sign_two_sided(samples):
    arr = np.asarray(samples, dtype=float)
    return float(2.0 * min(np.mean(arr >= 0.0), np.mean(arr <= 0.0)))

# ==================== Step 3: 按国家提取“female 分位系数 + CI” ====================
def analyze_by_country_detailed(min_n: int = 500, n_bootstrap: int = N_BOOTSTRAP_DEFAULT, random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    female_coef_rows = []            # 点估（逐 τ）
    female_coef_boot_rows = []       # Bootstrap CI（逐 τ）
    band_rows = []                   # 带区聚合（low/low_mid/mid/mid_high/high）
    band_compare_rows = []           # 带区对比（mid-vs-low、mid-vs-high、low-vs-high）
    delta_rows = []                  # 国家级 Δ 兜底汇总（底 vs 顶，用于与旧口径对比）

    print("=== 按国家抽取 being female 的分位系数（含95% bootstrap CI）===")
    print("[note] 本分析矩阵不含常数列（拦截项），female 系数为相对无截距线性项的边际效应。")
    if ESTIMAND_MODE == 'final_combo':
        print("[note] final_combo 模式：对最终组合 u_τ 进行线性投影（QuantReg），系数为线性近似，不作结构性解释。")
    for ctry, df_g in df_seniority.groupby('country'):
        d, Xdf, Xg, yg = build_design_for_subset(df_g)
        n = len(yg)
        if n < min_n:
            print(f"{ctry}: 跳过 (n={n} < {min_n})")
            continue
        print(f"{ctry}: 拟合模型并记录点估 (n={n})…")
        # 若 female 列不存在或无变异，直接跳过（避免无意义的 bootstrap 与 Δ 判定）
        if ('female' not in Xdf.columns) or (Xdf['female'].nunique() < 2):
            print(f"[warn] 跳过：{ctry} 的 female 列不存在或无变异")
            continue

        # 1) 拟合主模型并记录点估曲线
        m = fit_model(Xg, yg, taus=QUANTILES_MAIN, random_state=int(random_state))
        warn_counter = {'fit_count': 0, 'warn_count': 0}
        try:
            coef_point = _extract_female_coef_by_tau(m, Xdf, Xg, QUANTILES_MAIN, mode=ESTIMAND_MODE, warn_counter=warn_counter)
            for tau, val in coef_point.items():
                female_coef_rows.append({
                    'country': str(ctry), 'n': n, 'tau': float(tau),
                    'female_coef_point': float(val)
                })
        except Exception as e:
            print(f"[warn] 无法抽取系数点估（{ctry}）：{e}")

        # 2) Bootstrap：固定 h/g，仅对 QR 抽样系数（一次拟合 + 抽样）
        print(f"{ctry}: 进行 bootstrap（{int(n_bootstrap)} 次，固定 h/g，仅 QR 重估）以构造 CI…")
        seed = int(hashlib.md5(str(ctry).encode()).hexdigest()[:8], 16) + int(random_state)
        TAUS_FOR_BOOT = tuple(QUANTILES_MAIN)
        q_low, q_high = (min(QUANTILES_MAIN), max(QUANTILES_MAIN))
        female_idx = int(Xdf.columns.get_loc('female'))
        boot = m.bootstrap_inference(n_boot=int(n_bootstrap), random_state=seed, return_coefs=True)
        # 常见返回：beta_tau[tau]['draws'] -> (n_boot, n_params)
        coef_boot = {float(tau): [] for tau in TAUS_FOR_BOOT}
        delta_boot = []
        for tau in TAUS_FOR_BOOT:
            draws = boot.get('beta_tau', {}).get(float(tau), {}).get('draws', None)
            if draws is None:
                continue
            coef_boot[float(tau)] = list(np.asarray(draws)[:, female_idx].reshape(-1))
        if (q_low in coef_boot) and (q_high in coef_boot) and len(coef_boot[q_low]) > 0 and len(coef_boot[q_high]) > 0:
            a = np.asarray(coef_boot[q_high], dtype=float)
            b = np.asarray(coef_boot[q_low], dtype=float)
            n_success = int(min(a.shape[0], b.shape[0]))
            delta_boot = list((a[:n_success] - b[:n_success]).reshape(-1))
        else:
            n_success = 0

        # 2.1) 构造连续响应 h(Y)，用于效应量阈值计算与可选置换检验
        hy = None
        try:
            hy = m.h_iso_.transform(getattr(m, '_y_jit'))  # type: ignore[attr-defined]
        except Exception:
            hy = None
        if hy is None:
            for name in ('y_tilde_', 'y_cont_', 'y_transformed_', 'y_star_', 'y_continuous_'):
                hy = getattr(m, name, None)
                if hy is not None:
                    break
        if hy is None:
            # 若无法直接获取 h(Y)，用 X @ beta1_ 近似（仅用于定义尺度，不作推断）
            try:
                hy = (np.asarray(Xg, dtype=float) @ (m.beta1_.reshape(-1) if m.beta1_ is not None else np.zeros(Xg.shape[1])))
            except Exception:
                hy = np.asarray([np.nan] * Xg.shape[0])
        hy = np.asarray(hy, dtype=float).reshape(-1)
        hy_sd = float(np.nanstd(hy, ddof=1)) if np.isfinite(hy).all() else np.nan
        if np.isfinite(hy).all():
            hy_q25 = float(np.nanpercentile(hy, 25.0))
            hy_q75 = float(np.nanpercentile(hy, 75.0))
            hy_iqr = float(hy_q75 - hy_q25)
        else:
            hy_q25, hy_q75, hy_iqr = np.nan, np.nan, np.nan
        if EFFECT_SIZE_MODE == 'sd_pct' and np.isfinite(hy_sd):
            epsilon = float(EPSILON_SD_FRAC * hy_sd)
            epsilon_mode = 'sd_pct'
        elif EFFECT_SIZE_MODE == 'iqr_pct' and np.isfinite(hy_iqr):
            epsilon = float(EPSILON_IQR_FRAC * hy_iqr)
            epsilon_mode = 'iqr_pct'
        else:
            epsilon = np.nan
            epsilon_mode = 'undefined'

        # 3) CI 与显著性
        bottom_sig_neg = False
        top_sig_neg = False
        bottom_sig_pos = False
        top_sig_pos = False
        for tau in QUANTILES_MAIN:
            samples = coef_boot.get(tau, [])
            if samples:
                lo, hi, sig = _ci_from_samples(samples, level=0.95)
                p_boot = _p_from_sign_two_sided(samples)
                # 找对应点估
                pt = np.nan
                try:
                    pt = float([r['female_coef_point'] for r in female_coef_rows if (r['country'] == str(ctry) and abs(r['tau'] - float(tau)) < 1e-9)][-1])
                except Exception:
                    pass
                female_coef_boot_rows.append({
                    'country': str(ctry), 'n': n, 'tau': float(tau),
                    'female_coef_point': pt,
                    'female_coef_ci_low': lo, 'female_coef_ci_high': hi,
                    'female_coef_sig': sig, 'female_coef_p_boot': float(p_boot),
                    'ci_level': 0.95
                })
                if abs(float(tau) - float(q_low)) < 1e-9:
                    bottom_sig_neg = (hi < 0.0)
                    bottom_sig_pos = (lo > 0.0)
                if abs(float(tau) - float(q_high)) < 1e-9:
                    top_sig_neg = (hi < 0.0)
                    top_sig_pos = (lo > 0.0)

        # 若 female 列不存在或全常数，跳过后续 Δ 判定
        if 'female' not in Xdf.columns or Xdf['female'].nunique() < 2:
            print(f"[warn] 跳过 Δ 判定：{ctry} 的 female 列不存在或无变异")
            continue

        # 3.1) 直接检验 Δ_top-bottom 的显著性（叠加效应量阈值 ε）
        delta_row = None
        if delta_boot:
            d_lo, d_hi, d_sig = _ci_from_samples(delta_boot, level=0.95)
            d_p = _p_from_sign_two_sided(delta_boot)
            d_point = np.nan
            try:
                pt_top = float([r['female_coef_point'] for r in female_coef_rows if (r['country'] == str(ctry) and abs(r['tau'] - float(q_high)) < 1e-9)][-1])
                pt_bot = float([r['female_coef_point'] for r in female_coef_rows if (r['country'] == str(ctry) and abs(r['tau'] - float(q_low)) < 1e-9)][-1])
                d_point = pt_top - pt_bot
            except Exception:
                pass
            delta_abs = float(abs(d_point)) if np.isfinite(d_point) else np.nan
            delta_std_sd = float(d_point / hy_sd) if (np.isfinite(d_point) and np.isfinite(hy_sd) and hy_sd > 0) else np.nan
            # 概率尺度 Δ_prob（基于代表性样本、τ 网格近似）：
            # Δ_prob = [P_bottom(f=1)-P_bottom(f=0)] - [P_top(f=1)-P_top(f=0)]
            delta_prob = np.nan
            prob_bottom_f0 = np.nan
            prob_bottom_f1 = np.nan
            prob_top_f0 = np.nan
            prob_top_f1 = np.nan
            try:
                # 代表性样本：使用各特征的样本均值；仅切换 female=0/1
                x_rep = np.asarray(Xdf.mean(), dtype=float).reshape(-1)
                x0 = x_rep.copy(); x1 = x_rep.copy()
                x0[female_idx] = 0.0
                x1[female_idx] = 1.0
                # τ 网格
                tau_grid = np.linspace(0.02, 0.98, 41)
                y0_list = []
                y1_list = []
                rq_model = sm.QuantReg(hy, Xg)
                for t in tau_grid:
                    try:
                        rq = rq_model.fit(q=float(t), max_iter=5000, p_tol=QR_P_TOL)
                    except TypeError:
                        rq = rq_model.fit(q=float(t), max_iter=5000)
                    b = np.asarray(rq.params, dtype=float).reshape(-1)
                    hy0 = float(x0 @ b); hy1 = float(x1 @ b)
                    y0 = float(m._hinv(np.array([hy0]))[0])
                    y1 = float(m._hinv(np.array([hy1]))[0])
                    y0_list.append(y0)
                    y1_list.append(y1)
                y0_arr = np.asarray(y0_list, dtype=float)
                y1_arr = np.asarray(y1_list, dtype=float)
                # 类别边界
                j_min = int(m._y_min_cat) if m._y_min_cat is not None else int(np.nanmin(yg))
                j_max = int(m._y_max_cat) if m._y_max_cat is not None else int(np.nanmax(yg))
                # 底部概率: P(Y<=j_min)
                mask0_bottom = (y0_arr <= float(j_min) + 1e-9)
                mask1_bottom = (y1_arr <= float(j_min) + 1e-9)
                prob_bottom_f0 = float(np.max(tau_grid[mask0_bottom])) if np.any(mask0_bottom) else 0.0
                prob_bottom_f1 = float(np.max(tau_grid[mask1_bottom])) if np.any(mask1_bottom) else 0.0
                # 顶部概率: P(Y=j_max) = 1 - P(Y<=j_max-1)
                thresh_top = float(j_max - 1)
                mask0_topcdf = (y0_arr <= thresh_top + 1e-9)
                mask1_topcdf = (y1_arr <= thresh_top + 1e-9)
                F_topminus1_f0 = float(np.max(tau_grid[mask0_topcdf])) if np.any(mask0_topcdf) else 0.0
                F_topminus1_f1 = float(np.max(tau_grid[mask1_topcdf])) if np.any(mask1_topcdf) else 0.0
                prob_top_f0 = float(1.0 - F_topminus1_f0)
                prob_top_f1 = float(1.0 - F_topminus1_f1)
                delta_prob = (prob_bottom_f1 - prob_bottom_f0) - (prob_top_f1 - prob_top_f0)
            except Exception:
                delta_prob = np.nan

            # 分类规则（含边角）
            cls = 'no_heterogeneity'
            reason = '规则未触发显著差异'
            above_eps = (np.isfinite(delta_abs) and np.isfinite(epsilon) and delta_abs >= epsilon)
            meets_prob = (np.isfinite(delta_prob) and abs(float(delta_prob)) >= float(PROB_DIFF_THRESHOLD))
            if d_sig and above_eps and meets_prob and d_point > 0 and (delta_prob > 0):
                cls = 'sticky_floor'; reason = f'Δ>0、显著，|Δ|≥ε（{epsilon_mode}），且 Δ_prob≥{PROB_DIFF_THRESHOLD:.2f}'
            elif d_sig and above_eps and meets_prob and d_point < 0 and (delta_prob < 0):
                cls = 'glass_ceiling'; reason = f'Δ<0、显著，|Δ|≥ε（{epsilon_mode}），且 Δ_prob≥{PROB_DIFF_THRESHOLD:.2f}'
            elif d_sig and (not above_eps):
                cls = 'no_heterogeneity_small_effect'; reason = 'Δ 显著但效应量低于阈值 ε'
            elif d_sig and above_eps and (not meets_prob):
                cls = 'no_heterogeneity_small_prob_effect'; reason = f'Δ 显著且 |Δ|≥ε，但 Δ_prob<{PROB_DIFF_THRESHOLD:.2f}'
            elif bottom_sig_neg and top_sig_neg and not d_sig:
                cls = 'double_disadvantage'; reason = '两端均显著<0，但分位差Δ不显著'
            elif bottom_sig_pos and not (top_sig_neg or top_sig_pos):
                cls = 'female_advantage_bottom'; reason = '底部显著>0，顶部不显著'
            elif top_sig_pos and not (bottom_sig_neg or bottom_sig_pos):
                cls = 'female_advantage_top'; reason = '顶部显著>0，底部不显著'
            elif (not bottom_sig_neg and not bottom_sig_pos) and (not top_sig_neg and not top_sig_pos):
                cls = 'both_non_sig'; reason = '两端均不显著'

            unstable_flag = (n_success / max(int(n_bootstrap), 1) < 0.6)
            warn_rate = (float(warn_counter.get('warn_count', 0)) / float(warn_counter.get('fit_count', 1)))
            high_warn = bool(warn_rate > 0.2)
            # 可选：置换检验（在 h(Y) 上快速近似），默认关闭
            p_perm = np.nan
            if int(N_PERMUTATION_DEFAULT) > 0 and np.isfinite(d_point):
                try:
                    rng = np.random.default_rng(seed)
                    def _coef_at_tau(hy_arr, X_arr, tau):
                        try:
                            res = sm.QuantReg(hy_arr, X_arr).fit(q=float(tau), max_iter=5000, p_tol=QR_P_TOL)
                        except TypeError:
                            res = sm.QuantReg(hy_arr, X_arr).fit(q=float(tau), max_iter=5000)
                        return float(np.asarray(res.params, dtype=float)[female_idx])
                    def _delta_once(X_arr):
                        c_low = _coef_at_tau(hy, X_arr, q_low)
                        c_high = _coef_at_tau(hy, X_arr, q_high)
                        return float(c_high - c_low)
                    obs_abs = abs(float(d_point))
                    extreme = 0
                    for _ in range(int(N_PERMUTATION_DEFAULT)):
                        X_perm = np.asarray(Xg, dtype=float).copy()
                        X_perm[:, female_idx] = rng.permutation(X_perm[:, female_idx])
                        d_perm = _delta_once(X_perm)
                        if abs(d_perm) >= obs_abs:
                            extreme += 1
                    p_perm = float((extreme + 1) / (int(N_PERMUTATION_DEFAULT) + 1))
                except Exception:
                    p_perm = np.nan

            delta_row = {
                'country': str(ctry), 'n': n,
                'delta_top_bottom_point': d_point,
                'delta_top_bottom_ci_low': d_lo,
                'delta_top_bottom_ci_high': d_hi,
                'delta_top_bottom_sig': d_sig,
                'delta_top_bottom_p_boot': float(d_p),
                'delta_top_bottom_p_perm': p_perm,
                'classification': cls,
                'reason': reason,
                'delta_abs': delta_abs,
                'delta_std_sd': delta_std_sd,
                'hy_sd': hy_sd,
                'hy_iqr': hy_iqr,
                'epsilon': epsilon,
                'epsilon_mode': epsilon_mode,
                'prob_bottom_female0': prob_bottom_f0,
                'prob_bottom_female1': prob_bottom_f1,
                'prob_top_female0': prob_top_f0,
                'prob_top_female1': prob_top_f1,
                'delta_prob': delta_prob,
                'bootstrap_success_n': int(n_success),
                'bootstrap_total_n': int(n_bootstrap),
                'unstable_CI': bool(unstable_flag),
                'qr_warn_count': int(warn_counter.get('warn_count', 0)),
                'qr_fit_count': int(warn_counter.get('fit_count', 0)),
                'qr_warn_rate': warn_rate,
                'high_warn': high_warn,
                'estimand_mode': ESTIMAND_MODE,
            }
            # 将 Δ 信息重复附加到每个 τ 的行，便于统一导出与筛选
            for r in [row for row in female_coef_boot_rows if row['country'] == str(ctry) and row['n'] == n]:
                r.update({k: v for k, v in delta_row.items() if k not in r})
            # 国家级 Δ 行收集
            delta_rows.append(delta_row)

            # 3.2) Granularity: 带区聚合（均值）与三对对比
            # 带区聚合点估
            # 先构建 τ->点估 的映射
            point_map: Dict[float, float] = {
                float(r['tau']): float(r['female_coef_point'])
                for r in female_coef_rows if r['country'] == str(ctry) and r['n'] == n and np.isfinite(r.get('female_coef_point', np.nan))
            }
            # 构建 τ->bootstrap 抽样（每个 τ 的 female 系数）
            boot_map: Dict[float, np.ndarray] = {}
            for t in TAUS_FOR_BOOT:
                samples = coef_boot.get(float(t), [])
                if samples:
                    boot_map[float(t)] = np.asarray(samples, dtype=float).reshape(-1)
            # 预先计算 band 内的概率尺度 Δ_prob（使用 tau_grid 中落在 band 的点）
            band_to_delta_prob: Dict[str, float] = {}
            band_to_probs: Dict[str, Dict[str, float]] = {}
            # 预先计算基于 hy 的 rq_model（避免重复构建）
            try:
                rq_model_global = sm.QuantReg(hy, Xg)
            except Exception:
                rq_model_global = None
            # 先计算一次 y0_arr, y1_arr（用 41 点），后续按 band 过滤
            tau_grid_all = np.linspace(0.02, 0.98, 41)
            y0_arr = None; y1_arr = None
            try:
                x_rep = np.asarray(Xdf.mean(), dtype=float).reshape(-1)
                x0 = x_rep.copy(); x1 = x_rep.copy()
                x0[female_idx] = 0.0; x1[female_idx] = 1.0
                y0_list = []; y1_list = []
                if rq_model_global is not None:
                    for t in tau_grid_all:
                        try:
                            rq = rq_model_global.fit(q=float(t), max_iter=5000, p_tol=QR_P_TOL)
                        except TypeError:
                            rq = rq_model_global.fit(q=float(t), max_iter=5000)
                        b = np.asarray(rq.params, dtype=float).reshape(-1)
                        hy0 = float(x0 @ b); hy1 = float(x1 @ b)
                        y0_list.append(float(m._hinv(np.array([hy0]))[0]))
                        y1_list.append(float(m._hinv(np.array([hy1]))[0]))
                    y0_arr = np.asarray(y0_list, dtype=float)
                    y1_arr = np.asarray(y1_list, dtype=float)
            except Exception:
                y0_arr = None; y1_arr = None
            j_min = int(m._y_min_cat) if m._y_min_cat is not None else int(np.nanmin(yg))
            j_max = int(m._y_max_cat) if m._y_max_cat is not None else int(np.nanmax(yg))

            def _band_prob_metrics(low_tau: float, high_tau: float) -> Dict[str, float]:
                out = {'prob_bottom_f0': np.nan, 'prob_bottom_f1': np.nan, 'prob_top_f0': np.nan, 'prob_top_f1': np.nan, 'delta_prob': np.nan}
                if (y0_arr is None) or (y1_arr is None):
                    return out
                mask = (tau_grid_all >= float(low_tau) - 1e-12) & (tau_grid_all <= float(high_tau) + 1e-12)
                if not np.any(mask):
                    return out
                y0b = y0_arr[mask]; y1b = y1_arr[mask]; taus = tau_grid_all[mask]
                m0b = (y0b <= float(j_min) + 1e-9)
                m1b = (y1b <= float(j_min) + 1e-9)
                prob_bottom_f0 = float(np.max(taus[m0b])) if np.any(m0b) else 0.0
                prob_bottom_f1 = float(np.max(taus[m1b])) if np.any(m1b) else 0.0
                m0tcdf = (y0b <= float(j_max - 1) + 1e-9)
                m1tcdf = (y1b <= float(j_max - 1) + 1e-9)
                F0_topm1 = float(np.max(taus[m0tcdf])) if np.any(m0tcdf) else 0.0
                F1_topm1 = float(np.max(taus[m1tcdf])) if np.any(m1tcdf) else 0.0
                prob_top_f0 = float(1.0 - F0_topm1)
                prob_top_f1 = float(1.0 - F1_topm1)
                delta_prob_b = (prob_bottom_f1 - prob_bottom_f0) - (prob_top_f1 - prob_top_f0)
                out.update({'prob_bottom_f0': prob_bottom_f0, 'prob_bottom_f1': prob_bottom_f1,
                            'prob_top_f0': prob_top_f0, 'prob_top_f1': prob_top_f1,
                            'delta_prob': float(delta_prob_b)})
                return out

            # 逐 band 聚合
            band_points: Dict[str, float] = {}
            band_draws: Dict[str, np.ndarray] = {}
            for band_name, band_taus in BANDS_DEF.items():
                taus_in_band = [float(t) for t in band_taus]
                # 点估：同带 τ 的平均
                vals = [point_map.get(float(t), np.nan) for t in taus_in_band]
                vals = [v for v in vals if np.isfinite(v)]
                point_band = float(np.mean(vals)) if len(vals) > 0 else np.nan
                band_points[band_name] = point_band
                # 抽样：对每个 τ 的抽样先堆叠，再按列平均
                sample_mats = []
                for t in taus_in_band:
                    s = boot_map.get(float(t), None)
                    if s is not None and s.size > 0:
                        sample_mats.append(s.reshape(1, -1))
                if len(sample_mats) > 0:
                    S = np.vstack(sample_mats)  # shape: (#taus_in_band, n_boot)
                    draws_band = np.nanmean(S, axis=0).reshape(-1)  # n_boot,
                else:
                    draws_band = np.array([], dtype=float)
                band_draws[band_name] = draws_band
                lo_b, hi_b, sig_b = _ci_from_samples(draws_band, level=0.95)
                p_b = _p_from_sign_two_sided(draws_band)
                # 概率尺度（该 band 的 Δ_prob）
                if len(band_taus) == 1:
                    low_t, high_t = float(band_taus[0]), float(band_taus[0])
                else:
                    low_t, high_t = float(min(band_taus)), float(max(band_taus))
                prob_metrics = _band_prob_metrics(low_t, high_t)
                band_to_delta_prob[band_name] = prob_metrics['delta_prob']
                band_to_probs[band_name] = prob_metrics
                band_rows.append({
                    'country': str(ctry), 'n': n, 'band': band_name,
                    'taus': ','.join([str(t) for t in taus_in_band]),
                    'female_coef_point_band': point_band,
                    'female_coef_ci_low_band': lo_b, 'female_coef_ci_high_band': hi_b,
                    'female_coef_sig_band': bool(sig_b), 'female_coef_p_boot_band': float(p_b),
                    'delta_prob_band': float(prob_metrics['delta_prob']),
                    'prob_bottom_female0_band': float(prob_metrics['prob_bottom_f0']),
                    'prob_bottom_female1_band': float(prob_metrics['prob_bottom_f1']),
                    'prob_top_female0_band': float(prob_metrics['prob_top_f0']),
                    'prob_top_female1_band': float(prob_metrics['prob_top_f1']),
                    'hy_sd': hy_sd, 'epsilon': epsilon, 'epsilon_mode': epsilon_mode,
                })

            # 三对对比 + Holm 校正
            pair_rows_tmp = []
            for a, b in COMPARE_PAIRS:
                da = band_draws.get(a, np.array([], dtype=float))
                db = band_draws.get(b, np.array([], dtype=float))
                if da.size == 0 or db.size == 0:
                    diff_draws = np.array([], dtype=float)
                else:
                    diff_draws = (da - db).reshape(-1)
                lo_d, hi_d, sig_d = _ci_from_samples(diff_draws, level=0.95)
                p_d = _p_from_sign_two_sided(diff_draws) if diff_draws.size > 0 else np.nan
                point_d = float(band_points.get(a, np.nan) - band_points.get(b, np.nan))
                delta_prob_diff = float(band_to_delta_prob.get(a, np.nan) - band_to_delta_prob.get(b, np.nan))
                # 阈值判定
                above_eps_pair = (np.isfinite(point_d) and np.isfinite(epsilon) and abs(point_d) >= float(epsilon))
                meets_prob_pair = (np.isfinite(delta_prob_diff) and abs(delta_prob_diff) >= float(PROB_DIFF_THRESHOLD))
                cls_pair = 'no_heterogeneity'
                reason_pair = '规则未触发显著差异'
                if sig_d and above_eps_pair and meets_prob_pair and point_d > 0 and (delta_prob_diff > 0):
                    cls_pair = f'{a}_more_sticky_vs_{b}'; reason_pair = f'{a} 相对 {b} 更“底部不利”（均满足阈值）'
                elif sig_d and above_eps_pair and meets_prob_pair and point_d < 0 and (delta_prob_diff < 0):
                    cls_pair = f'{a}_more_glass_vs_{b}'; reason_pair = f'{a} 相对 {b} 更“顶部不利”（均满足阈值）'
                elif sig_d and not above_eps_pair:
                    cls_pair = 'diff_sig_but_small_effect'; reason_pair = '|差值| 低于 ε'
                elif sig_d and above_eps_pair and not meets_prob_pair:
                    cls_pair = 'diff_sig_but_small_prob_effect'; reason_pair = f'Δ_prob 差值 < {PROB_DIFF_THRESHOLD:.2f}'

                rowp = {
                    'country': str(ctry), 'n': n,
                    'band_A': a, 'band_B': b,
                    'diff_point': point_d,
                    'diff_ci_low': lo_d, 'diff_ci_high': hi_d,
                    'diff_sig': bool(sig_d), 'diff_p_boot': float(p_d) if np.isfinite(p_d) else np.nan,
                    'delta_prob_diff': delta_prob_diff,
                    'classification_pair': cls_pair,
                    'reason_pair': reason_pair,
                    'epsilon': epsilon, 'epsilon_mode': epsilon_mode,
                }
                pair_rows_tmp.append(rowp)
            # Holm 校正（每国 3 次对比）
            p_vals = [r['diff_p_boot'] for r in pair_rows_tmp if np.isfinite(r['diff_p_boot'])]
            if len(p_vals) > 0:
                m = len(pair_rows_tmp)
                # sort by p ascending
                order = sorted(range(m), key=lambda i: (pair_rows_tmp[i]['diff_p_boot'] if np.isfinite(pair_rows_tmp[i]['diff_p_boot']) else 1.0))
                p_sorted = [pair_rows_tmp[i]['diff_p_boot'] if np.isfinite(pair_rows_tmp[i]['diff_p_boot']) else 1.0 for i in order]
                p_adj_sorted = []
                for j, pj in enumerate(p_sorted, start=1):
                    p_adj_sorted.append(min((m - j + 1) * pj, 1.0))
                # enforce monotonicity
                for j in range(1, len(p_adj_sorted)):
                    p_adj_sorted[j] = max(p_adj_sorted[j], p_adj_sorted[j - 1])
                # assign back
                for idx_pos, i in enumerate(order):
                    pair_rows_tmp[i]['diff_p_boot_holm'] = float(p_adj_sorted[idx_pos])
            # 基于 Holm 的更保守显著性标记
            for r in pair_rows_tmp:
                r['diff_sig_holm'] = (np.isfinite(r.get('diff_p_boot_holm', np.nan)) and float(r['diff_p_boot_holm']) < 0.05)
                # 按“未校正CI显著 AND Holm显著”共同为真进入强结论
                if r['diff_sig'] and r['diff_sig_holm']:
                    pass  # 分类在上方已依据未校正CI；保留 reason_pair，不强改
                else:
                    # 若任一不显著，则若原分类为强结论则降级为 no_heterogeneity
                    if r['classification_pair'] in (f'{a}_more_sticky_vs_{b}', f'{a}_more_glass_vs_{b}'):
                        r['classification_pair'] = 'no_heterogeneity'
                        r['reason_pair'] = 'Holm 未通过或 CI 未显著'
            for r in pair_rows_tmp:
                band_compare_rows.append(r)
        else:
            warn_rate = (float(warn_counter.get('warn_count', 0)) / float(warn_counter.get('fit_count', 1)))
            delta_row = {
                'country': str(ctry), 'n': n,
                'delta_top_bottom_point': np.nan,
                'delta_top_bottom_ci_low': np.nan,
                'delta_top_bottom_ci_high': np.nan,
                'delta_top_bottom_sig': False,
                'delta_top_bottom_p_boot': np.nan,
                'classification': 'insufficient_bootstrap',
                'reason': 'Δ 无有效自助样本',
                'bootstrap_success_n': int(n_success),
                'bootstrap_total_n': int(n_bootstrap),
                'unstable_CI': True,
                'qr_warn_count': int(warn_counter.get('warn_count', 0)),
                'qr_fit_count': int(warn_counter.get('fit_count', 0)),
                'qr_warn_rate': warn_rate,
                'high_warn': True,
                'estimand_mode': ESTIMAND_MODE,
            }
            for r in [row for row in female_coef_boot_rows if row['country'] == str(ctry) and row['n'] == n]:
                r.update({k: v for k, v in delta_row.items() if k not in r})
            # 国家级 Δ 行收集
            delta_rows.append(delta_row)

    female_coef_df = pd.DataFrame(female_coef_rows)
    female_coef_ci_df = pd.DataFrame(female_coef_boot_rows)
    female_coef_all = (
        female_coef_df.merge(
            female_coef_ci_df,
            on=['country', 'n', 'tau'],
            how='outer'
        ).sort_values(['country', 'tau'])
    )
    # 统一列名：合并后可能出现 female_coef_point_x / female_coef_point_y
    if 'female_coef_point_x' in female_coef_all.columns or 'female_coef_point_y' in female_coef_all.columns:
        # 优先使用点估来源（_x），缺失时用 CI 表中的回填（_y）
        female_coef_all['female_coef_point'] = female_coef_all.get('female_coef_point_x')
        if 'female_coef_point_y' in female_coef_all.columns:
            female_coef_all['female_coef_point'] = female_coef_all['female_coef_point'].fillna(
                female_coef_all['female_coef_point_y']
            )
        drop_cols = [c for c in ['female_coef_point_x', 'female_coef_point_y'] if c in female_coef_all.columns]
        female_coef_all = female_coef_all.drop(columns=drop_cols)
    return female_coef_all, pd.DataFrame(delta_rows), pd.DataFrame(band_rows), pd.DataFrame(band_compare_rows)

_log(f"[2/3] 开始分国家系数分析 (min_n=500, bootstrap={N_BOOTSTRAP_DEFAULT})…")
female_coef_all, delta_df, band_df, band_cmp_df = analyze_by_country_detailed(min_n=500, n_bootstrap=N_BOOTSTRAP_DEFAULT)

print(f"\n完成！涵盖 {female_coef_all['country'].nunique() if len(female_coef_all) > 0 else 0} 个国家")
print("\n=== being female 的分位系数（前10行）===")
print(female_coef_all.head(10))

# ==================== Step 4: 导出与可视化 ====================
_log("[3/3] 导出 CSV 并绘制系数—分位点曲线图…")
if len(female_coef_all) > 0:
    out_csv = f'{OUTPUT_DIR}/female_coef_by_quantile.csv'
    female_coef_all.to_csv(out_csv, index=False)
    print(f"✓ 分位系数表（female + CI）已保存: {out_csv}")
    # 输出国家级汇总（Δ 与分类 + 诊断），直接用 delta_df，确保无 CI 国家也有分类
    keep_cols = ['country', 'n',
                 'delta_top_bottom_point', 'delta_abs', 'delta_std_sd', 'delta_prob',
                 'prob_bottom_female0', 'prob_bottom_female1', 'prob_top_female0', 'prob_top_female1',
                 'delta_top_bottom_ci_low', 'delta_top_bottom_ci_high',
                 'delta_top_bottom_sig', 'delta_top_bottom_p_boot', 'delta_top_bottom_p_perm',
                 'classification', 'reason',
                 'hy_sd', 'hy_iqr', 'epsilon', 'epsilon_mode',
                 'bootstrap_success_n', 'bootstrap_total_n', 'unstable_CI',
                 'qr_warn_count', 'qr_fit_count', 'qr_warn_rate', 'high_warn',
                 'estimand_mode']
    summary_df = delta_df.reindex(columns=[c for c in keep_cols if c in delta_df.columns])
    out_csv2 = f'{OUTPUT_DIR}/female_coef_summary_by_country.csv'
    summary_df.to_csv(out_csv2, index=False)
    print(f"✓ 国家级汇总（Δ 与分类）已保存: {out_csv2}")
    # 输出带区聚合与对比结果
    if len(band_df) > 0:
        out_csv3 = f'{OUTPUT_DIR}/female_coef_bands.csv'
        band_df.to_csv(out_csv3, index=False)
        print(f"✓ 带区聚合结果已保存: {out_csv3}")
    if len(band_cmp_df) > 0:
        out_csv4 = f'{OUTPUT_DIR}/female_coef_band_comparisons.csv'
        band_cmp_df.to_csv(out_csv4, index=False)
        print(f"✓ 带区对比结果已保存: {out_csv4}")
        # 生成写作友好的“带区对比概览”宽表（附录用）
        try:
            cols_view = ['country','band_A','band_B','diff_point','diff_ci_low','diff_ci_high','diff_p_boot_holm','delta_prob_diff','classification_pair']
            view_df = band_cmp_df.reindex(columns=[c for c in cols_view if c in band_cmp_df.columns]).copy()
            # 可按国家透视为多列（可在后处理中继续美化），此处先直接导出长表视图
            out_csv5 = f'{OUTPUT_DIR}/female_coef_band_comparisons_view.csv'
            view_df.to_csv(out_csv5, index=False)
            print(f"✓ 带区对比概览（长表）已保存: {out_csv5}")
        except Exception:
            pass
    print("[note] 附注：qr_warn_rate > 0.2 视为不稳定（high_warn=True）。")

    # 画每国一条曲线，并用阴影标出 95% CI
    plt.figure(figsize=(7, 5))
    for ctry, g in female_coef_all.groupby('country'):
        g = g.sort_values('tau')
        plt.plot(g['tau'], g['female_coef_point'], label=str(ctry))
        if {'female_coef_ci_low', 'female_coef_ci_high'}.issubset(g.columns):
            mask = g[['female_coef_ci_low','female_coef_ci_high']].notna().all(axis=1)
            if mask.any():
                plt.fill_between(g.loc[mask, 'tau'], g.loc[mask, 'female_coef_ci_low'], g.loc[mask, 'female_coef_ci_high'], alpha=0.15)
    plt.axhline(0.0, linestyle='--', linewidth=1)
    plt.xlabel('Quantile (τ)')
    ylabel = 'Coef of being female (on h(Y) scale)' if ESTIMAND_MODE=='single_index' else 'Linearized coef of being female on final combo u_τ'
    plt.ylabel(ylabel)
    # 取与 bootstrap 匹配的底/顶分位用于标题说明
    q_low, q_high = (min(QUANTILES_MAIN), max(QUANTILES_MAIN))
    plt.title(f'Female effect across quantiles (95% CI at τ={q_low}, {q_high})')
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = f'{OUTPUT_DIR}/female_coef_curve.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()
    print(f"✓ 系数曲线图已保存: {fig_path}")
else:
    print("无可导出的系数结果。")

# 简明阅读口径（用于论文/报告）
print("\n—— 如何判读：")
print("- 底部 τ=0.2 显著<0、顶部 τ=0.8 ≈0 → sticky floor")
print("- 顶部 τ=0.8 显著<0、底部 τ=0.2 ≈0 → glass ceiling")
print("- 两端均显著<0 → double disadvantage；区间覆盖0 → 无显著分位异质性")

print("\n🎯 核心方法提示：本脚本直接用 “being female 的分位系数 + bootstrap CI” 验证 sticky floor / glass ceiling。" )
print("- 设计矩阵无截距：系数为相对无截距基准的边际效应；Δ 判定不受影响，仅需说明。")
print("- 概率尺度为近似：通过 τ→h(Y)→h^{-1}(·)→类别阈值 的离散近似得到底/顶概率差 Δ_prob，便于可解释性。")

