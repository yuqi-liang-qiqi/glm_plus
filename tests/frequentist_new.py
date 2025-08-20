import pandas as pd
import numpy as np
import os as _os, sys as _sys, math
# 确保可以以绝对方式导入 `glm_plus`（把项目根目录加入 sys.path）
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
from glm_plus.frequentist_version.torque import FrequentistOQR
from matplotlib import pyplot as plt
from typing import Dict, Tuple

# ==================== 常量与配置（全局唯一） ====================
# 分位点集合
QUANTILES_MAIN = (0.2, 0.5, 0.8)
ALL_QUANTILES = QUANTILES_MAIN
BOTTOM_QUANTILES = {0.2}
TOP_QUANTILES = {0.8}
ORDER_LABELS = ["Assistant/Junior", "Regular", "Leader", "Chief/Founder"]

def _pos_of_tau(t: float) -> str:
    return 'bottom' if t <= 0.3 else ('top' if t >= 0.7 else 'middle')

# 简单进度日志函数
def _log(msg: str) -> None:
    print(f"[Progress] {msg}")

# subsample 规则
SUBSAMPLE_RULE = lambda n: (min(n, 5000) if n > 20000 else (min(n, 3000) if n > 2000 else None))

# 输出目录
import os
OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 统一拟合函数：量化配置在一处管理
def fit_model(X: np.ndarray, y: np.ndarray, taus=QUANTILES_MAIN, random_state: int = 0, use_two_index: bool = False) -> FrequentistOQR:
    # 最小改法：每国只拟合一次 + 降低网格/子样本提升速度；固定单指数
    m = FrequentistOQR(
        quantiles=taus,
        use_two_index=False,
        auto_select_k=True,
        subsample_n=2000,
        rank_grid_n=31,
        t_grid_n=41,
        random_state=random_state,
    )
    m.fit(X, y)
    return m

# ==================== Step 1: 读取数据（仅用于后续分国家分析） ====================
_log("[1/4] 读取数据...")
_DATA_DIR = _os.path.dirname(__file__)
_CSV_PATH = _os.path.join(_DATA_DIR, "df_seniority.csv")
assert _os.path.exists(_CSV_PATH), f"找不到数据文件: {_CSV_PATH}"
df_seniority = pd.read_csv(_CSV_PATH)
_log("[1/4] 数据加载完成")

# 最基本的数据列检查
assert 'country' in df_seniority.columns, "df_seniority 需要包含 'country' 列"

# 按国家分组评估“粘地板/玻璃天花板”
# 仅保留用于构建设计矩阵的函数，其它旧评估函数已由新版覆盖

def build_design_for_subset(df_sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    cols_needed = [
        'gender', 'highest_educational_degree',
        'whether_bachelor_university_prestigious',
        'internationalization', 'work_years', 'company_size', 'Y10'
    ]
    d = df_sub[cols_needed].dropna().copy()
    # y
    cat_type = pd.CategoricalDtype(categories=ORDER_LABELS, ordered=True)
    d['y_cat'] = d['Y10'].astype(cat_type)
    # 过滤不在已知标签内的记录，避免 y 编码为 0
    d = d[~d['y_cat'].isna()].copy()
    d['y'] = d['y_cat'].cat.codes + 1
    # female
    gg = d['gender'].astype(str).str.strip().str.lower()
    d['female'] = (gg == 'female').astype(int)
    # prestige
    col_p = 'whether_bachelor_university_prestigious'
    if str(d[col_p].dtype) == 'bool':
        d['prestigious_bachelor'] = d[col_p].astype(int)
    else:
        d['prestigious_bachelor'] = (
            d[col_p].astype(str).str.strip().str.lower().isin(['true','1','yes','y','t'])
        ).astype(int)
    # numeric
    d['work_years'] = pd.to_numeric(d['work_years'], errors='coerce')
    d = d.dropna(subset=['work_years']).copy()
    # one-hot
    cat_cols = ['highest_educational_degree', 'internationalization', 'company_size']
    X_cat = pd.get_dummies(d[cat_cols], columns=cat_cols, drop_first=True, dtype=int)
    X_df = pd.concat([
        d[['female', 'prestigious_bachelor', 'work_years']].reset_index(drop=True),
        X_cat.reset_index(drop=True)
    ], axis=1)
    X = X_df.to_numpy(dtype=float)
    y = d['y'].to_numpy(dtype=int)
    return d, X_df, X, y

 

# ==================== 函数提前准备好：写好函数的性别差（底/中/顶）====================

def compute_gender_gaps_at_key_quantiles(model: FrequentistOQR, X_df_in: pd.DataFrame, 
                                        quantiles=QUANTILES_MAIN) -> Dict:
    """
    固定协变量在典型值，只切换性别，计算各分位点的female-male差异
    返回详细结果字典
    """
    # 构造代表性个体：连续变量取中位数，类别变量取基准类(全0)
    ref = {c: 0.0 for c in X_df_in.columns}
    ref['female'] = 0.0  # 男性基准
    if 'prestigious_bachelor' in X_df_in:
        ref['prestigious_bachelor'] = float(X_df_in['prestigious_bachelor'].median())
    if 'work_years' in X_df_in:
        ref['work_years'] = float(X_df_in['work_years'].median())
    
    # 男性和女性的预测
    X_male = pd.DataFrame([ref])
    X_female = pd.DataFrame([ref]); X_female['female'] = 1.0
    
    pred_male = model.predict_quantiles_continuous(X_male.to_numpy(dtype=float), quantiles=quantiles)
    pred_female = model.predict_quantiles_continuous(X_female.to_numpy(dtype=float), quantiles=quantiles)
    
    # 计算差异：female - male（负值表示女性预测更低，即处境更差）
    gaps = {}
    for tau in quantiles:
        male_val = float(np.asarray(pred_male[tau], dtype=float).reshape(-1)[0])
        female_val = float(np.asarray(pred_female[tau], dtype=float).reshape(-1)[0])
        gap = female_val - male_val
        gaps[tau] = {
            'quantile': tau,
            'male_pred': male_val,
            'female_pred': female_val, 
            'gap_female_minus_male': gap,
            'position': _pos_of_tau(tau)
        }
    
    # 判别粘地板vs玻璃天花板
    bottom_gap = float(gaps[0.2]['gap_female_minus_male']) if 0.2 in gaps else np.nan
    top_gap = float(gaps[0.8]['gap_female_minus_male']) if 0.8 in gaps else np.nan
    
    conclusion = 'glass ceiling' if abs(top_gap) > abs(bottom_gap) else 'sticky floor'
    gap_diff = bottom_gap - top_gap  # 正值倾向sticky floor
    
    return {
        'gaps_by_quantile': gaps,
        'bottom_gap_avg': bottom_gap,
        'top_gap_avg': top_gap, 
        'gap_diff_bottom_minus_top': gap_diff,
        'conclusion': conclusion,
        'reference_profile': dict(ref)
    }

# ==================== 函数提前准备好：离下一门槛的距离（中点法） ====================

def compute_threshold_distances(model: FrequentistOQR, X: np.ndarray, y: np.ndarray, 
                               X_df: pd.DataFrame, quantiles=ALL_QUANTILES, non_negative: bool = False) -> Dict:
    """
    计算每个个体在各分位点离下一门槛的距离，并分析性别差异
    - 当前级别使用观测 y_true 而非预测floor
    - non_negative=True 则对距离做非负截断
    """
    pred_cont = model.predict_quantiles_continuous(X, quantiles=quantiles)
    results = {}
    for tau in quantiles:
        y_pred_cont = pred_cont[tau]
        distances, current_levels, target_levels = [], [], []
        for y_pred, y_true_i in zip(y_pred_cont, y):
            current_level = int(np.clip(y_true_i, 1, 4))
            if current_level >= 4:
                distances.append(np.nan); current_levels.append(current_level); target_levels.append(np.nan)
                continue
            # 中点法：把相邻类别 j 与 j+1 的边界定义为 j+0.5
            boundary = current_level + 0.5
            d_raw = float(boundary - y_pred)
            d_use = max(0.0, d_raw) if non_negative else d_raw
            distances.append(d_use)
            current_levels.append(current_level)
            target_levels.append(current_level + 1)
        df_dist = pd.DataFrame({
            'y_pred_cont': y_pred_cont,
            'y_true': y,
            'current_level': current_levels,
            'target_level': target_levels,
            'distance_to_next': distances,
            'female': X_df['female'].values if 'female' in X_df.columns else 0
        })
        df_valid = df_dist.dropna(subset=['distance_to_next'])
        if len(df_valid) > 0:
            gender_distances = df_valid.groupby('female')['distance_to_next'].agg(['mean', 'std', 'count']).reset_index()
            gender_distances['gender'] = gender_distances['female'].map({0: 'male', 1: 'female'})
            if len(gender_distances) == 2:
                male_dist = gender_distances.loc[gender_distances['gender']=='male','mean'].iloc[0]
                female_dist = gender_distances.loc[gender_distances['gender']=='female','mean'].iloc[0]
                gap = float(female_dist - male_dist)
            else:
                male_dist = female_dist = gap = np.nan
        else:
            gender_distances = pd.DataFrame()
            male_dist = female_dist = gap = np.nan
        results[tau] = {
            'quantile': tau,
            'position': _pos_of_tau(tau),
            'data': df_dist,
            'gender_summary': gender_distances,
            'male_avg_distance': male_dist,
            'female_avg_distance': female_dist,
            'gap_female_minus_male': gap
        }
    return results


# 随机代表人平均差：从真实协变量分布中随机抽样，计算平均性别差
def compute_average_gaps_over_random_references(
    model: FrequentistOQR,
    X_df_in: pd.DataFrame,
    n_samples: int = 1000,
    random_state: int = 0,
    quantiles = ALL_QUANTILES,
) -> Dict[float, float]:
    rng = np.random.RandomState(random_state)
    if len(X_df_in) == 0:
        return {tau: np.nan for tau in quantiles}
    sample_size = int(min(n_samples, len(X_df_in)))
    idx = rng.choice(len(X_df_in), size=sample_size, replace=True)
    X_sample = X_df_in.iloc[idx].copy()
    X_male = X_sample.copy(); X_male['female'] = 0.0
    X_female = X_sample.copy(); X_female['female'] = 1.0
    pred_male = model.predict_quantiles_continuous(X_male.to_numpy(dtype=float), quantiles=quantiles)
    pred_female = model.predict_quantiles_continuous(X_female.to_numpy(dtype=float), quantiles=quantiles)
    avg_gaps = {}
    for tau in quantiles:
        male_vals = np.asarray(pred_male[tau], dtype=float)
        female_vals = np.asarray(pred_female[tau], dtype=float)
        avg_gaps[tau] = float(np.nanmean(female_vals - male_vals))
    return avg_gaps


# ==================== Step 2: 按国家分组分析 ====================

def analyze_by_country_detailed(min_n: int = 500, n_bootstrap: int = 500, random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按国家进行完整的性别差和门槛距离分析
    返回两个汇总表：gender_gap_summary, threshold_distance_summary
    """
    gap_rows = []
    threshold_rows = []
    avg_refs_rows = []
    boot_summary_rows = []
    
    print("=== 按国家分析进度 ===")
    for ctry, df_g in df_seniority.groupby('country'):
        d, Xdf, Xg, yg = build_design_for_subset(df_g)
        n = len(yg)
        if n < min_n:
            print(f"{ctry}: 跳过 (n={n} < {min_n})")
            continue
        
        print(f"{ctry}: 分析中 (n={n})...")
        
        # 拟合模型（一次拟合）
        m = fit_model(Xg, yg, taus=QUANTILES_MAIN, random_state=0, use_two_index=False)
        # Bootstrap 推断：固定 h/g，仅对 QR 重估（返回系数抽样）
        try:
            boot = m.bootstrap_inference(n_boot=n_bootstrap, return_coefs=True)
            female_idx = int(Xdf.columns.get_loc('female')) if 'female' in Xdf.columns else None
            if female_idx is not None:
                b1_ci_lo = float(np.asarray(boot['beta1']['ci_low'])[female_idx])
                b1_ci_hi = float(np.asarray(boot['beta1']['ci_high'])[female_idx])
                b1_ci_level = float(boot['beta1'].get('ci_level', 0.95))
                b1_sig = not (b1_ci_lo <= 0.0 <= b1_ci_hi)
                # 近似 p 值（正态近似）：z = coef / se
                female_coef = float(m.beta1_[female_idx]) if getattr(m, 'beta1_', None) is not None else np.nan
                female_se = float(np.asarray(boot['beta1']['se'])[female_idx])
                if np.isfinite(female_coef) and np.isfinite(female_se) and female_se > 0:
                    z = abs(female_coef / female_se)
                    b1_p = float(math.erfc(z / math.sqrt(2.0)))
                else:
                    b1_p = np.nan
            else:
                b1_ci_lo = b1_ci_hi = np.nan
                b1_ci_level = 0.95
                b1_sig = False
                b1_p = np.nan
        except Exception as _e:
            b1_ci_lo = b1_ci_hi = np.nan
            b1_ci_level = 0.95
            b1_sig = False
            b1_p = np.nan
        
        # 性别差分析（基准代表人）
        gaps = compute_gender_gaps_at_key_quantiles(m, Xdf)
        for tau in ALL_QUANTILES:
            gap_info = gaps['gaps_by_quantile'][tau]
            gap_rows.append({
                'country': str(ctry),
                'n': n,
                'quantile': tau,
                'position': gap_info['position'],
                'male_pred': gap_info['male_pred'],
                'female_pred': gap_info['female_pred'],
                'gap_female_minus_male': gap_info['gap_female_minus_male'],
                'female_ci_low_b1_q50': b1_ci_lo,
                'female_ci_high_b1_q50': b1_ci_hi,
                'female_sig_b1_q50': b1_sig,
                'female_se_b1_q50': female_se if 'female_se' in locals() else np.nan,
                'female_p_b1_q50': b1_p,
                'ci_level': b1_ci_level
            })

        # 顶部差/底部差（点估，按代表人）
        bottom_gap_point = float(gaps['gaps_by_quantile'][0.2]['gap_female_minus_male']) if 0.2 in gaps['gaps_by_quantile'] else np.nan
        top_gap_point = float(gaps['gaps_by_quantile'][0.8]['gap_female_minus_male']) if 0.8 in gaps['gaps_by_quantile'] else np.nan
        diff_abs_point = float(abs(top_gap_point) - abs(bottom_gap_point)) if np.isfinite(bottom_gap_point) and np.isfinite(top_gap_point) else np.nan
        
        # 门槛距离分析
        thresh_results = compute_threshold_distances(m, Xg, yg, Xdf)
        for tau in ALL_QUANTILES:
            thresh_info = thresh_results[tau]
            threshold_rows.append({
                'country': str(ctry),
                'n': n,
                'quantile': tau,
                'position': thresh_info['position'],
                'male_avg_distance': thresh_info['male_avg_distance'],
                'female_avg_distance': thresh_info['female_avg_distance'],
                'gap_female_minus_male': thresh_info['gap_female_minus_male']
            })

        # 随机代表人平均差（额外稳健性）
        avg_gaps = compute_average_gaps_over_random_references(m, Xdf, n_samples=1000, random_state=random_state)
        avg_bottom = float(avg_gaps.get(0.2, np.nan))
        avg_top = float(avg_gaps.get(0.8, np.nan))
        avg_diff_abs_point = float(abs(avg_top) - abs(avg_bottom)) if np.isfinite(avg_bottom) and np.isfinite(avg_top) else np.nan
        avg_refs_rows.append({
            'country': str(ctry),
            'n': n,
            'avg_bottom_gap_random_refs': avg_bottom,
            'avg_top_gap_random_refs': avg_top,
            'conclusion_random_refs': ('glass ceiling' if abs(avg_top) > abs(avg_bottom) else 'sticky floor')
        })

        # Bootstrap 顶部差−底部差：使用系数抽样（female 系数在 h(Y) 的 QR 中对所有 X 恒等于系数本身）
        boot_diff_abs = []
        boot_avg_diff_abs = []
        try:
            female_idx2 = int(Xdf.columns.get_loc('female')) if 'female' in Xdf.columns else None
            if female_idx2 is not None:
                draws_low = boot.get('beta_tau', {}).get(0.2, {}).get('draws', None)
                draws_top = boot.get('beta_tau', {}).get(0.8, {}).get('draws', None)
                if draws_low is not None and draws_top is not None:
                    low_f = np.asarray(draws_low)[:, female_idx2]
                    top_f = np.asarray(draws_top)[:, female_idx2]
                    boot_diff_abs = list(np.abs(top_f) - np.abs(low_f))
                    # 平均代表人差同样等于 female 系数，因此与上者相同
                    boot_avg_diff_abs = list(np.abs(top_f) - np.abs(low_f))
        except Exception:
            boot_diff_abs = []
            boot_avg_diff_abs = []

        def _ci_and_sig(samples: list[float], level: float = 0.95) -> tuple[float, float, bool]:
            if not samples:
                return (np.nan, np.nan, False)
            arr = np.asarray(samples, dtype=float)
            lo = float(np.nanpercentile(arr, (1.0 - level) / 2.0 * 100.0))
            hi = float(np.nanpercentile(arr, (1.0 + level) / 2.0 * 100.0))
            sig = not (lo <= 0.0 <= hi)
            return (lo, hi, sig)

        def _p_from_sign(samples: list[float]) -> float:
            if not samples:
                return float('nan')
            arr = np.asarray(samples, dtype=float)
            p = 2.0 * float(min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
            return float(min(max(p, 0.0), 1.0))

        diff_abs_lo, diff_abs_hi, diff_abs_sig = _ci_and_sig(boot_diff_abs, level=0.95)
        avg_diff_abs_lo, avg_diff_abs_hi, avg_diff_abs_sig = _ci_and_sig(boot_avg_diff_abs, level=0.95)
        diff_abs_p = _p_from_sign(boot_diff_abs)
        avg_diff_abs_p = _p_from_sign(boot_avg_diff_abs)

        boot_summary_rows.append({
            'country': str(ctry),
            'n': n,
            'diff_abs_top_minus_bottom_point': diff_abs_point,
            'diff_abs_top_minus_bottom_ci_low': diff_abs_lo,
            'diff_abs_top_minus_bottom_ci_high': diff_abs_hi,
            'diff_abs_top_minus_bottom_sig': diff_abs_sig,
            'diff_abs_top_minus_bottom_p': diff_abs_p,
            'avg_diff_abs_top_minus_bottom_point': avg_diff_abs_point,
            'avg_diff_abs_top_minus_bottom_ci_low': avg_diff_abs_lo,
            'avg_diff_abs_top_minus_bottom_ci_high': avg_diff_abs_hi,
            'avg_diff_abs_top_minus_bottom_sig': avg_diff_abs_sig,
            'avg_diff_abs_top_minus_bottom_p': avg_diff_abs_p,
            'ci_level': 0.95
        })
    
    gap_df = pd.DataFrame(gap_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    avg_refs_df = pd.DataFrame(avg_refs_rows)
    boot_summary_df = pd.DataFrame(boot_summary_rows)
    
    return gap_df, threshold_df, avg_refs_df, boot_summary_df

_log("[2/4] 开始分国家分析 (min_n=500, bootstrap=500)...")
country_gaps_df, country_thresholds_df, avg_refs_df, boot_summary_df = analyze_by_country_detailed(min_n=500, n_bootstrap=500)

print(f"\n完成！涵盖 {country_gaps_df['country'].nunique()} 个国家")
print("\n=== 性别差汇总（前10行）===")
print(country_gaps_df.head(10))

print("\n=== 门槛距离汇总（前10行）===")
print(country_thresholds_df.head(10))
print("\n=== 随机代表人平均差（前10行）===")
print(avg_refs_df.head(10))
print("\n=== 顶部差−底部差（含CI，前10行）===")
print(boot_summary_df.head(10))

_log("[3/4] 生成图表...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Gender Gap Analysis by Country (Two Heatmaps)', fontsize=16)

# 左图：按国家的性别差热图
ax_left = axes[0]
if len(country_gaps_df) > 0:
    pivot_gaps = country_gaps_df.pivot(index='country', columns='quantile', values='gap_female_minus_male')
    im1 = ax_left.imshow(pivot_gaps.values, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax_left.set_xticks(range(len(pivot_gaps.columns)))
    ax_left.set_xticklabels([f'τ={tau:.1f}' for tau in pivot_gaps.columns])
    ax_left.set_yticks(range(len(pivot_gaps.index)))
    ax_left.set_yticklabels(pivot_gaps.index, fontsize=8)
    ax_left.set_title('Gender Gap by Country (Female - Male)')
    ax_left.set_xlabel('Quantile')
    plt.colorbar(im1, ax=ax_left, label='Gap')
else:
    ax_left.text(0.5, 0.5, 'No country data available', ha='center', va='center')
    ax_left.set_title('Gender Gap by Country')

# 右图：按国家的门槛距离热图
ax_right = axes[1]
if len(country_thresholds_df) > 0:
    pivot_thresholds = country_thresholds_df.pivot(index='country', columns='quantile', values='gap_female_minus_male')
    im2 = ax_right.imshow(pivot_thresholds.values, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax_right.set_xticks(range(len(pivot_thresholds.columns)))
    ax_right.set_xticklabels([f'τ={tau:.1f}' for tau in pivot_thresholds.columns])
    ax_right.set_yticks(range(len(pivot_thresholds.index)))
    ax_right.set_yticklabels(pivot_thresholds.index, fontsize=8)
    ax_right.set_title('Threshold Distance Gap by Country (Female - Male)')
    ax_right.set_xlabel('Quantile')
    plt.colorbar(im2, ax=ax_right, label='Gap')
else:
    ax_right.text(0.5, 0.5, 'No threshold data available', ha='center', va='center')
    ax_right.set_title('Threshold Distance Gap by Country')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/gender_gap_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()
_log(f"[3/4] 图表已保存到 {OUTPUT_DIR}/gender_gap_analysis.pdf (双热图)")

# 打印关键结论（分国家维度）
print("=" * 60)
print("关键研究结论（分国家）")
print("=" * 60)

if len(country_gaps_df) > 0:
    # 按国家统计结论分布
    country_conclusions = []
    for country in country_gaps_df['country'].unique():
        country_data = country_gaps_df[country_gaps_df['country'] == country]
        bottom_avg = country_data[country_data['position'] == 'bottom']['gap_female_minus_male'].mean()
        top_avg = country_data[country_data['position'] == 'top']['gap_female_minus_male'].mean()
        conclusion = 'glass ceiling' if abs(top_avg) > abs(bottom_avg) else 'sticky floor'
        country_conclusions.append({'country': country, 'conclusion': conclusion, 
                                   'bottom_gap': bottom_avg, 'top_gap': top_avg})
    country_summary = pd.DataFrame(country_conclusions)
    
    
# ==================== Step 4: 导出结果 ====================
_log("[4/4] 写出 CSV 与配置...")

# 1. 性别差分位表（仅国家级）
if len(country_gaps_df) > 0:
    gap_combined = country_gaps_df.copy()
    gap_combined['analysis_level'] = 'country'
    gap_combined = gap_combined[['analysis_level', 'country', 'quantile', 'position', 
                                 'male_pred', 'female_pred', 'gap_female_minus_male',
                                 'female_ci_low_b1_q50', 'female_ci_high_b1_q50', 'female_sig_b1_q50', 'ci_level']]
else:
    gap_combined = pd.DataFrame(columns=['analysis_level','country','quantile','position','male_pred','female_pred','gap_female_minus_male'])

gap_combined.to_csv(f'{OUTPUT_DIR}/gender_gap_by_quantile.csv', index=False)
print(f"✓ 性别差分位表已保存: {OUTPUT_DIR}/gender_gap_by_quantile.csv")

# 2. 门槛距离表（仅国家级）
if len(country_thresholds_df) > 0:
    thresh_combined = country_thresholds_df.copy()
    thresh_combined['analysis_level'] = 'country'
    thresh_combined = thresh_combined[['analysis_level', 'country', 'quantile', 'position',
                                       'male_avg_distance', 'female_avg_distance', 'gap_female_minus_male']]
else:
    thresh_combined = pd.DataFrame(columns=['analysis_level','country','quantile','position','male_avg_distance','female_avg_distance','gap_female_minus_male'])

# 2.1 随机代表人平均差（附加表）
if len(avg_refs_df) > 0:
    avg_refs_export = avg_refs_df.copy()
    avg_refs_export.to_csv(f'{OUTPUT_DIR}/avg_gap_random_refs.csv', index=False)
    print(f"✓ 随机代表人平均差表已保存: {OUTPUT_DIR}/avg_gap_random_refs.csv")

# 2.2 顶部差−底部差（含显著性）
if len(boot_summary_df) > 0:
    boot_summary_export = boot_summary_df.copy()
    boot_summary_export.to_csv(f'{OUTPUT_DIR}/top_minus_bottom_diff_with_ci.csv', index=False)
    print(f"✓ 顶部差−底部差（含CI）已保存: {OUTPUT_DIR}/top_minus_bottom_diff_with_ci.csv")

thresh_combined.to_csv(f'{OUTPUT_DIR}/threshold_distance.csv', index=False)
print(f"✓ 门槛距离表已保存: {OUTPUT_DIR}/threshold_distance.csv")

# 3. 汇总结论表（仅国家级）
summary_export = []
if len(country_gaps_df) > 0 and 'country_summary' in locals():
    for _, row in country_summary.iterrows():
        summary_export.append({
            'analysis_level': 'country',
            'country': row['country'],
            'bottom_gap_avg': row['bottom_gap'],
            'top_gap_avg': row['top_gap'],
            'gap_diff_bottom_minus_top': row['bottom_gap'] - row['top_gap'],
            'conclusion': row['conclusion'],
            'n_observations': int(country_gaps_df[country_gaps_df['country'] == row['country']].iloc[0]['n'])
        })

summary_df = pd.DataFrame(summary_export)
summary_df.to_csv(f'{OUTPUT_DIR}/analysis_summary.csv', index=False)
print(f"✓ 汇总结论表已保存: {OUTPUT_DIR}/analysis_summary.csv")

# 保存模型配置信息
config_info = {
    'model_config': 'per-country models via fit_model()',
    'quantiles_analyzed': ALL_QUANTILES,
    'bottom_quantiles': BOTTOM_QUANTILES,
    'top_quantiles': TOP_QUANTILES,
    'countries_analyzed': int(country_gaps_df['country'].nunique()) if len(country_gaps_df) > 0 else 0,
    'analysis_timestamp': pd.Timestamp.now().isoformat()
}

import json
with open(f'{OUTPUT_DIR}/analysis_config.json', 'w') as f:
    json.dump(config_info, f, indent=2, default=str)
print(f"✓ 分析配置已保存: {OUTPUT_DIR}/analysis_config.json")

print(f"\n🎉 分析完成！所有结果已保存到 '{OUTPUT_DIR}/' 目录")
print("\n核心文件清单:")
print(f"  📊 gender_gap_by_quantile.csv - 国家级性别差分位详表")
print(f"  📏 threshold_distance.csv - 国家级门槛距离分析详表") 
print(f"  📋 analysis_summary.csv - 粘地板/玻璃天花板结论汇总")
print(f"  📈 gender_gap_analysis.pdf - 双热图可视化")
print(f"  ⚙️  analysis_config.json - 模型配置与元数据")

# 最终研究结论摘要
print("\n" + "=" * 80)
print("🔍 最终研究结论摘要")
print("=" * 80)
print(f"研究问题: 分析 Y10 职级中的性别差异模式（分国家）")
if len(country_gaps_df) > 0:
    print(f"国家覆盖: {int(country_gaps_df['country'].nunique())} 个国家")

print(f"\n可直接用于报告的关键数据:")
print("建议阅读 CSV 热图而非单一全样本曲线；结论以国家级别为准。")
