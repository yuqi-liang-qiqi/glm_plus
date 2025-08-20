# 时间滞后序数回归模型：t年的rarity_score预测t+1年的seniority
# 因变量: t+1年的seniority (序数变量：Junior < Assistant < Regular < Senior < Leader < Chief or founder)
# 解释变量: t年的rarity_score
# 控制变量: t年的其他变量
# 注意：由于需要下一年的seniority，最后一年的观测值会丢失

import pandas as pd
import numpy as np
import os
import argparse
from scipy import stats

# 解析参数并设置默认 CSV 路径（脚本同目录）
script_dir = "/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_data/real_data_my_paper/250808_tree/"
default_csv = os.path.join(script_dir, "final_df.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--final-csv", dest="final_csv", default=default_csv, help="Path to final_df.csv")
args = parser.parse_args()

final_df = pd.read_csv(args.final_csv)

# 导入自定义gologit2库
import sys
import os
# 添加glm_plus包的路径
glm_plus_path = os.path.join(os.path.dirname(__file__), '..', '..', 'glm_plus')
sys.path.insert(0, glm_plus_path)

try:
    from gologit2.gologit2 import GeneralizedOrderedModel
    HAS_ORDERED = True
    print("✓ 使用自定义gologit2库进行序数回归")
except ImportError as e:
    HAS_ORDERED = False
    print(f"⚠️ 自定义gologit2库不可用: {e}")
    print("请确保glm_plus/gologit2/gologit2.py文件存在")

import statsmodels.api as sm
import patsy
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

assert 'final_df' in globals(), "请确保已有 final_df 变量（包含 worker_id, country, year, seniority, rarity_score 等列）"

# 确保必要列存在
required_cols = {'worker_id','country','year','seniority','rarity_score'}
missing_req = required_cols - set(final_df.columns)
assert not missing_req, f"final_df 缺少必要列: {missing_req}"

print("创建时间滞后数据集...")
print(f"原始数据: {len(final_df)} 行")

# 将 seniority 编码为序数
seniority_order = ['Junior', 'Assistant', 'Regular', 'Senior', 'Leader', 'Chief or founder']
seniority_mapping = {level: i for i, level in enumerate(seniority_order)}

print("序数变量 seniority 的编码映射:")
for level, code in seniority_mapping.items():
    print(f"  {level} -> {code}")

# 检查是否有未知的 seniority 值
unknown_seniority = set(final_df['seniority'].dropna().unique()) - set(seniority_order)
if unknown_seniority:
    print(f"⚠️ 发现未知的 seniority 值: {unknown_seniority}")

# 创建序数编码的 seniority
final_df['seniority_ordinal'] = final_df['seniority'].map(seniority_mapping)

# 预处理：去除极端缺失
work = final_df.dropna(subset=['worker_id','year','seniority','rarity_score', 'seniority_ordinal']).copy()
print(f"去除缺失后: {len(work)} 行")

# 确保 year 是整数
work['year'] = work['year'].astype(int)
work['worker_id'] = work['worker_id'].astype(str).str.strip()

# 创建时间滞后数据
print("创建时间滞后变量...")

# 按 worker_id 分组，为每个人创建滞后变量
lag_data = []

for worker_id, group in work.groupby('worker_id'):
    # 按年份排序
    group_sorted = group.sort_values('year')
    
    # 检查年份是否连续（至少有两个连续年份）
    years = group_sorted['year'].values
    
    for i in range(len(group_sorted) - 1):
        current_row = group_sorted.iloc[i]
        next_row = group_sorted.iloc[i + 1]
        
        # 检查是否是连续年份
        if next_row['year'] == current_row['year'] + 1:
            # 创建滞后观测：t年的解释变量 + t+1年的因变量
            lag_row = current_row.copy()  # t年的所有变量作为解释变量
            lag_row['seniority_next'] = next_row['seniority']  # t+1年的seniority作为因变量
            lag_row['seniority_ordinal_next'] = next_row['seniority_ordinal']  # 编码后的
            lag_row['year_current'] = current_row['year']  # 当前年份
            lag_row['year_next'] = next_row['year']  # 下一年份
            
            lag_data.append(lag_row)

# 转换为DataFrame
lag_df = pd.DataFrame(lag_data)
print(f"时间滞后数据集: {len(lag_df)} 行 (从 {len(work)} 行创建)")

if len(lag_df) == 0:
    print("❌ 没有找到连续年份的数据，无法创建滞后数据集")
    exit()

# 对关键解释变量 rarity_score 做 z-score 标准化（按国家分组），以便跨国比较（每 +1 SD）
def _zscore_series(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std(ddof=0)
    if pd.notna(sd) and sd > 0:
        return (s - m) / sd
    # 若标准差为0或NaN，则返回0，避免产生NaN/Inf
    return pd.Series(0.0, index=s.index)

lag_df['rarity_score_z'] = lag_df.groupby('country', group_keys=False)['rarity_score'].apply(_zscore_series)
print("已对 rarity_score 做按国家的 z-score 标准化（1 单位 = 1 个国家内标准差）")

# 显示数据统计
print(f"\n数据分布:")
print(f"- 年份范围: {lag_df['year_current'].min()}-{lag_df['year_current'].max()} -> {lag_df['year_next'].min()}-{lag_df['year_next'].max()}")
print(f"- 独特个体数: {lag_df['worker_id'].nunique()}")
print(f"- 国家分布: {lag_df['country'].value_counts().to_dict()}")

# 处理性别变量
if 'gender' in lag_df.columns:
    lag_df['gender'] = lag_df['gender'].astype(str).str.strip().str.lower()
    lag_df['gender_female'] = lag_df['gender'].map({'female': 1.0, 'male': 0.0})
else:
    lag_df['gender_female'] = np.nan

# 控制变量集合（排除主键与因变量/主解释变量）
exclude = set([
    'worker_id','country','year','seniority','gender',
    'seniority_next', 'seniority_ordinal_next', 'year_current', 'year_next',
    'seniority_ordinal',
    'rarity_score', 'rarity_score_z'  # 显式排除主解释变量，避免重复加入
])
control_cols_all = [c for c in lag_df.columns if c not in exclude]

# 分类变量（object 或 category）与数值变量
cat_controls = [c for c in control_cols_all if str(lag_df[c].dtype) in ['object','category']]
num_controls = [c for c in control_cols_all if c not in cat_controls]

print(f"\n控制变量:")
print(f"- 数值控制变量: {num_controls}")
print(f"- 分类控制变量: {cat_controls}")

countries = list(pd.unique(lag_df['country']))
print(f"- 分析国家: {countries}")

results = []
models = {}

# === 工具函数：计算有序Logit的边际效应（APE） ===
def compute_ordered_marginal_effects(model_obj, X_df, var_name, seniority_levels):
    """
    计算有序Logit模型中某个变量对每个类别概率的平均偏效应（Average Partial Effects）。
    使用数值近似方法，兼容自定义gologit2 API。

    参数:
        model_obj: 已拟合的GeneralizedOrderedModel对象
        X_df: 设计矩阵DataFrame
        var_name: 变量名
        seniority_levels: 序数级别名称列表

    返回：
      {
        'levels': [level names...],
        'effects': np.ndarray shape (J,),  # 各类别的APE
        'se': np.ndarray shape (J,) or None,  # 标准误（目前为None）
        'summary': str  # 可读性摘要
      }
    """
    try:
        # 使用数值近似计算边际效应
        dx = 1e-4
        # 采样以加速计算
        max_sample = 5000
        if len(X_df) > max_sample:
            X_s = X_df.sample(n=max_sample, random_state=123).copy()
        else:
            X_s = X_df.copy()

        if var_name not in X_s.columns:
            raise ValueError(f"变量 {var_name} 不在设计矩阵中")

        # 判断是否二元变量（0/1）
        col = pd.Series(pd.unique(X_s[var_name].dropna()))
        is_binary = set(np.sort(col.values)) <= {0, 1}

        if is_binary:
            # 离散变化：将该列整体从0切换到1，计算概率差的样本平均
            X0 = X_s.copy() 
            X0[var_name] = 0.0
            X1 = X_s.copy()
            X1[var_name] = 1.0
            
            # 使用自定义gologit2的predict_proba方法
            P0 = model_obj.predict_proba(X0)  # shape: (n_samples, n_categories)
            P1 = model_obj.predict_proba(X1)
            
            dP = (P1 - P0)
            ape = dP.mean(axis=0)  # 各类别的平均边际效应
            
            J = len(seniority_levels)
            ape = np.asarray(ape).reshape(-1)[:J]
            
            return {
                'levels': seniority_levels,
                'effects': ape,
                'se': None,  # 暂不提供标准误
                'summary': 'Discrete APE (0→1) computed via gologit2 predict_proba'
            }

        else:
            # 连续变量：数值导数近似
            # 对标准化变量（如rarity_score_z）使用dx=1.0来对齐"+1 SD"解释
            if var_name.endswith('_z'):
                dx = 1.0  # 标准化变量，1单位 = 1 SD
            else:
                dx = 1e-4  # 其他连续变量使用小步长数值导数
            
            X_plus = X_s.copy()
            X_plus[var_name] = X_plus[var_name].astype(float) + dx

            # 使用自定义gologit2的predict_proba方法预测概率矩阵
            P = model_obj.predict_proba(X_s)      # shape: (n_samples, n_categories) 
            P_plus = model_obj.predict_proba(X_plus)

            # 平均偏效应
            dP = (P_plus - P) / dx
            ape = dP.mean(axis=0)

            J = len(seniority_levels)
            ape = np.asarray(ape).reshape(-1)[:J]

            dx_desc = 'per+1SD' if dx == 1.0 else f'dx={dx}'
            return {
                'levels': seniority_levels,
                'effects': ape,
                'se': None,  # 暂不提供标准误
                'summary': f"Numerical APE ({dx_desc}) computed via gologit2 over {len(X_s)} samples"
            }
            
    except Exception as e:
        return {
            'levels': seniority_levels,
            'effects': None,
            'se': None,
            'summary': f"边际效应计算失败: {e}"
        }

# === 工具函数：系数的平均边际效应（基于期望职级 E[Y]）===
def compute_ame_expected(model_obj, X_df, var_name):
    """
    返回单个数值：对期望职级 E[Y] 的平均边际效应。
    - 连续变量：数值导数 (E[Y|x+dx]-E[Y|x])/dx 的样本平均
    - 二元变量（0/1）：离散变化 E[Y|x=1]-E[Y|x=0] 的样本平均
    
    注意：对于gologit2非平行变量，AME可能与单个系数方向不同。
    这是正常现象，因为：
    1. gologit2系数是"各cutpoint的差异"，不是"全局斜率"
    2. 非平行变量在不同阈值的效应可能方向不同
    3. AME是所有阈值效应的加权平均，更贴近实际平均效应
    4. 解释时应优先使用AME，而非单个系数值
    
    参数:
        model_obj: 已拟合的GeneralizedOrderedModel对象
        X_df: 设计矩阵DataFrame
        var_name: 变量名
    """
    # 采样以加速
    max_sample = 5000
    if len(X_df) > max_sample:
        X_s = X_df.sample(n=max_sample, random_state=123).copy()
    else:
        X_s = X_df.copy()

    if var_name not in X_s.columns:
        return np.nan, 'missing'

    # 预测各类别概率并计算期望职级
    def expected_y(X):
        P = model_obj.predict_proba(X)  # shape: (n_samples, n_categories)
        levels = np.arange(P.shape[1])  # 0..J-1（序数编码）
        return (P * levels.reshape(1, -1)).sum(axis=1)

    col = X_s[var_name]
    # 判断是否二元
    unique_vals = pd.Series(pd.unique(col.dropna()))
    is_binary = set(np.sort(unique_vals.values)) <= {0, 1}

    try:
        if is_binary:
            X0 = X_s.copy()
            X0[var_name] = 0.0
            X1 = X_s.copy() 
            X1[var_name] = 1.0
            Ey0 = expected_y(X0)
            Ey1 = expected_y(X1)
            ame = float((Ey1 - Ey0).mean())
            return ame, 'discrete_0to1'
        else:
            # 对标准化变量（如rarity_score_z）使用dx=1.0来对齐"+1 SD"解释
            if var_name.endswith('_z'):
                dx = 1.0  # 标准化变量，1单位 = 1 SD
            else:
                dx = 1e-4  # 其他连续变量使用小步长数值导数
            
            X_plus = X_s.copy()
            X_plus[var_name] = X_plus[var_name].astype(float) + dx
            Ey = expected_y(X_s)
            Ey_plus = expected_y(X_plus)
            ame = float(((Ey_plus - Ey) / dx).mean())
            
            method_desc = f'per+1SD' if dx == 1.0 else f'derivative_dx={dx}'
            return ame, method_desc
    except Exception as e:
        return np.nan, f'error: {e}'

print(f"\n开始分国家回归分析...")

for ctry in countries:
    print(f"\n--- 分析国家: {ctry} ---")
    dfc = lag_df[lag_df['country'] == ctry].copy()
    if dfc.empty:
        print(f"  {ctry}: 无数据，跳过")
        continue

    print(f"  {ctry}: 数据量 {len(dfc)} 行，个体数 {dfc['worker_id'].nunique()}")

    # 构建设计矩阵：rarity_score_z + controls（类别控制独热编码）
    X_parts = [dfc[['rarity_score_z']]]

    # 数值控制
    if num_controls:
        num_data = dfc[num_controls]
        if not num_data.empty:
            X_parts.append(num_data)

    # 类别控制做独热编码（指定参考组，drop_first 删除参考组列）
    if cat_controls:
        available_cats = [c for c in cat_controls if c in dfc.columns]
        if available_cats:
            df_cat = dfc[available_cats].copy()
            # 指定参考组顺序
            if 'highest_educational_degree' in df_cat.columns:
                vals = list(pd.Series(df_cat['highest_educational_degree'].astype(str).unique()))
                vals = [v for v in vals if v != 'Bachelor']
                ordered = ['Bachelor'] + vals
                df_cat['highest_educational_degree'] = pd.Categorical(
                    df_cat['highest_educational_degree'], categories=ordered, ordered=True
                )
            if 'internationalization' in df_cat.columns:
                vals = list(pd.Series(df_cat['internationalization'].astype(str).unique()))
                vals = [v for v in vals if v != 'Local']
                ordered = ['Local'] + vals
                df_cat['internationalization'] = pd.Categorical(
                    df_cat['internationalization'], categories=ordered, ordered=True
                )
            if 'simplified_company_size' in df_cat.columns:
                ref = 'Micro (0-10 employees)'
                vals = list(pd.Series(df_cat['simplified_company_size'].astype(str).unique()))
                vals = [v for v in vals if v != ref]
                ordered = [ref] + vals
                df_cat['simplified_company_size'] = pd.Categorical(
                    df_cat['simplified_company_size'], categories=ordered, ordered=True
                )

            dummies = pd.get_dummies(df_cat, drop_first=True, dtype=float)
            if not dummies.empty:
                X_parts.append(dummies)

    # === 加入时间固定效应（Year FE：year_current 的虚拟变量）===
    if 'year_current' in dfc.columns:
        year_cat = dfc['year_current'].astype(int).astype('category')
        year_dummies = pd.get_dummies(year_cat, drop_first=True, prefix='year', dtype=float)
        if not year_dummies.empty:
            X_parts.append(year_dummies)

    X = pd.concat(X_parts, axis=1)
    # 去重列名，防止重复列导致后续 dtype 检查把列切成 DataFrame
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    y_ordinal = dfc['seniority_ordinal_next']  # 使用下一年的序数编码seniority
    y_original = dfc['seniority_next']  # 保留原始文本用于参考

    # 去除任何仍有缺失的行
    valid = X.notna().all(axis=1) & y_ordinal.notna()
    X = X.loc[valid]
    y_ordinal = y_ordinal.loc[valid]
    y_original = y_original.loc[valid]
    dfc_valid = dfc.loc[valid]

    print(f"  {ctry}: 清理后剩余 {len(X)} 观测值")

    if len(X) < 50:  # 最小样本量检查
        print(f"  {ctry}: 样本量太小(<50)，跳过")
        continue

    # 方法1: 尝试序数回归（使用自定义gologit2）
    ordered_success = False
    
    if HAS_ORDERED:
        try:
            # 确保所有数据都是数值型
            X_clean = X.copy()
            
            # 检查并转换所有列为数值型；同时消除完美多重共线（零方差）
            for col in list(X_clean.columns):
                s = X_clean[col]
                if pd.api.types.is_object_dtype(s):
                    X_clean[col] = pd.to_numeric(s, errors='coerce')
                elif pd.api.types.is_bool_dtype(s):
                    X_clean[col] = s.astype(float)
                # 去除常数列（零方差）
                if X_clean[col].nunique(dropna=False) <= 1:
                    X_clean.drop(columns=[col], inplace=True)
            
            # 去除转换后产生的NaN行
            mask = X_clean.notna().all(axis=1) & y_ordinal.notna()
            X_clean = X_clean.loc[mask]
            y_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有变量都是数值型（float或int）
            for col in list(X_clean.columns):
                s = X_clean[col]
                if not pd.api.types.is_numeric_dtype(s):
                    X_clean[col] = s.astype(float)
            
            # === 增强数据验证：确保每列都有变异 ===
            problem_cols = []
            for col in list(X_clean.columns):
                nunique = X_clean[col].nunique(dropna=False)
                if nunique <= 1:
                    problem_cols.append(f"{col}(nunique={nunique})")
            
            if problem_cols:
                print(f"  {ctry}: WARNING: Found constant/problem columns: {problem_cols}")
                # 移除问题列
                for col in [c.split('(')[0] for c in problem_cols]:
                    if col in X_clean.columns:
                        X_clean.drop(columns=[col], inplace=True)
                print(f"  {ctry}: Removed {len(problem_cols)} problematic columns")
            
            # 验证最终数据质量
            final_nunique = {col: X_clean[col].nunique() for col in X_clean.columns}
            min_nunique = min(final_nunique.values()) if final_nunique else 0
            
            print(f"  {ctry}: 序数回归数据准备完成，{len(y_clean)} 观测值")
            print(f"  因变量分布: {dict(pd.Series(y_clean).value_counts().sort_index())}")
            print(f"  特征变异性验证: 最小nunique={min_nunique}, 特征数={len(X_clean.columns)}")
            
            # 安全检查：确保至少有基本特征
            if 'rarity_score_z' not in X_clean.columns:
                print(f"  {ctry}: ERROR: rarity_score_z missing after data cleaning!")
                continue
            
            # 使用自定义gologit2模型
            X_final = X_clean
            feature_names = list(X_final.columns)
            
            # === PARALLEL CONSTRAINT STRATEGY ===
            # Put most variables in parallel to reduce dimensionality and improve stability
            # Only key variables (rarity_score_z) remain non-parallel for flexible effects
            
            # === PPOM AUTOMATIC VARIABLE SELECTION ===
            print(f"  {ctry}: 开始PPOM自动变量选择（从全平行到部分非平行）...")
            
            # 定义应该保持parallel的变量（通常不违反比例优势假设）
            always_parallel = []
            
            # 1. 年份dummy通常应该parallel（固定效应性质）
            year_vars = [col for col in feature_names if col.startswith('year_')]
            always_parallel.extend(year_vars)
            
            # 2. 基本人口特征通常parallel
            basic_controls = [col for col in feature_names if col in ['work_years', 'gender_female']]
            always_parallel.extend(basic_controls)
            
            # 3. 公司规模dummy可能parallel
            size_vars = [col for col in feature_names if 'company_size' in col.lower()]
            always_parallel.extend(size_vars)
            
            # 候选非平行变量（可能违反比例优势的变量）
            candidate_npl_vars = []
            
            # 1. 核心变量：稀有度评分
            if 'rarity_score_z' in feature_names:
                candidate_npl_vars.append('rarity_score_z')
            
            # 2. 国际化变量：可能在不同职级有不同效应
            intl_vars = [col for col in feature_names if 'internationalization' in col.lower()]
            candidate_npl_vars.extend(intl_vars)
            
            # 3. 教育变量：可能在职业进阶不同阶段效应不同
            edu_vars = [col for col in feature_names if 'educational_degree' in col.lower() and col not in always_parallel]
            candidate_npl_vars.extend(edu_vars)
            
            # 去重并确保候选变量存在
            always_parallel = list(set(always_parallel) & set(feature_names))
            candidate_npl_vars = list(set(candidate_npl_vars) & set(feature_names))
            
            print(f"  {ctry}: 强制parallel变量 (n={len(always_parallel)}): {always_parallel}")
            print(f"  {ctry}: 候选非平行变量 (n={len(candidate_npl_vars)}): {candidate_npl_vars}")
            
            # 执行自动变量选择
            autofit_model = GeneralizedOrderedModel(link='logit')
            
            selection_result = autofit_model.autofit_variable_selection(
                X_final.values, 
                y_clean.values,
                feature_names=feature_names,
                candidate_vars=candidate_npl_vars,  # 只考虑这些变量作为候选非平行
                significance_level=0.01,  # 严格的显著性水平
                max_npl_vars=3,  # 最多3个非平行变量，控制模型复杂度
                verbose=True
            )
            
            if selection_result['success']:
                # 使用自动选择的结果
                selected_npl_vars = selection_result['selected_npl_vars']
                final_pl_vars = selection_result['final_pl_vars']
                
                # 确保always_parallel中的变量保持parallel
                modified_constraints = False
                for var in always_parallel:
                    if var not in final_pl_vars and var in feature_names:
                        final_pl_vars.append(var)
                        if var in selected_npl_vars:
                            selected_npl_vars.remove(var)
                        modified_constraints = True
                
                # 关键修复：如果约束被修改，需要重新拟合模型
                if modified_constraints:
                    print(f"  {ctry}: 强制并行约束已修改，重新拟合最终模型...")
                    model = GeneralizedOrderedModel(link='logit', pl_vars=final_pl_vars)
                    res = model.fit(
                        X_final.values,
                        y_clean.values,
                        feature_names=feature_names,
                        verbose=False,
                        maxiter=2000,
                        tol=1e-5
                    )
                    print(f"  {ctry}: 最终模型重拟合完成：success={res.success}")
                else:
                    # 约束没有被修改，使用原结果
                    model = selection_result['final_model']
                    res = selection_result['final_result']
                
                print(f"  {ctry}: PPOM选择完成")
                print(f"  {ctry}: 最终平行变量 (n={len(final_pl_vars)}): {final_pl_vars}")
                print(f"  {ctry}: 最终非平行变量 (n={len(selected_npl_vars)}): {selected_npl_vars}")
                print(f"  {ctry}: 模型改进: ΔLL = {selection_result['improvement']:.4f}")
                
                # 保存选择过程用于诊断
                models[(ctry, 'ppom_selection')] = selection_result
                
                # 更新后续使用的变量
                pl_vars = final_pl_vars  # 用于后续门槛分析
                non_parallel_vars = selected_npl_vars
                
            else:
                # 自动选择失败，回退到保守策略
                print(f"  {ctry}: PPOM自动选择失败，使用保守的parallel约束...")
                
                # 将大部分变量设为parallel，只让rarity_score_z非parallel
                pl_vars = [v for v in feature_names if v != 'rarity_score_z']
                non_parallel_vars = ['rarity_score_z'] if 'rarity_score_z' in feature_names else []
                
                model = GeneralizedOrderedModel(link='logit', pl_vars=pl_vars)
                print(f"  {ctry}: 回退到传统约束策略，拟合gologit2模型...")
                
                res = model.fit(
                    X_final.values, 
                    y_clean.values, 
                    feature_names=feature_names,
                    verbose=False,
                    maxiter=2000,
                    tol=1e-5
                )
            
            print(f"  {ctry}: 模型拟合完成：success={res.success}, nit={res.nit if hasattr(res, 'nit') else 'N/A'}")

            
            models[(ctry,'ordered_logit_lag')] = model  # 存储拟合好的模型对象
            models[(ctry,'ordered_logit_lag_result')] = res  # 存储结果对象

            # 从gologit2结果中提取系数信息
            coef = np.nan
            se = np.nan  
            pval = np.nan
            
            # === 修正特征索引错位问题 ===
            # 使用模型存储的训练时变量顺序，而不是推断
            npl_names = model.npl_vars_ if hasattr(model, 'npl_vars_') and model.npl_vars_ else [col for col in feature_names if col not in pl_vars]
            print(f"  {ctry}: Non-parallel variable order (from model): {npl_names}")
            
            # 尝试从结果中获取rarity_score_z的系数、标准误和p值
            if 'rarity_score_z' in npl_names:
                var_idx = npl_names.index('rarity_score_z')  # 在非平行变量列表中的正确索引
                print(f"  {ctry}: rarity_score_z index in npl_names: {var_idx}")
                
                # gologit2的beta_npl是 (K, p_npl) 形状，每个阈值都有系数
                if res.beta_npl is not None:
                    print(f"  {ctry}: beta_npl shape: {res.beta_npl.shape}")
                    if var_idx < res.beta_npl.shape[1]:  # 确保索引不越界
                        coef = float(res.beta_npl[0, var_idx])  # 使用第一个阈值的系数作为代表
                    else:
                        print(f"  {ctry}: WARNING: var_idx {var_idx} >= beta_npl.shape[1] {res.beta_npl.shape[1]}")
                
                # 提取标准误和p值（如果可用）
                if hasattr(res, 'se_beta_npl') and res.se_beta_npl is not None:
                    if var_idx < res.se_beta_npl.shape[1]:
                        se = float(res.se_beta_npl[0, var_idx])  # 第一个阈值的标准误
                
                if hasattr(res, 'pvalues_beta_npl') and res.pvalues_beta_npl is not None:
                    if var_idx < res.pvalues_beta_npl.shape[1]:
                        pval = float(res.pvalues_beta_npl[0, var_idx])  # 第一个阈值的p值
            else:
                print(f"  {ctry}: WARNING: rarity_score_z not found in non-parallel variables: {npl_names}")
            
            # 提取pseudo R2
            pseudo_r2 = getattr(res, 'pseudo_r2', np.nan)
            n = len(y_clean)
            
            # 计算期望职级的平均边际效应（AME）
            ame_rarity, ame_type = compute_ame_expected(model, X_final, 'rarity_score_z')
            models[(ctry,'ordered_logit_lag_ame_rarity')] = {'ame': ame_rarity, 'type': ame_type}
            
            # === 门槛级别详细分析（新功能）===
            print(f"  {ctry}: 开始门槛级别分析...")
            
            try:
                target_var = "rarity_score_z"
                
                # 1) 各门槛"斜率"表（β_j）
                coef_by_cut = res.coef_by_threshold(target_var)
                coef_df = coef_by_cut.to_frame(name="coef")
                
                # 2) 各门槛"阈值概率效应"（∂Pr{Y≥j}/∂x）（自动选择dx）
                dpr_ge = res.dPr_ge_by_threshold(X_final.values, target_var)
                dpr_ge.name = "dPr_ge"
                
                # 3) 各类别"概率效应"（∂Pr{Y=c}/∂x）（自动选择dx）
                dpr_cat = res.dPr_cat_by_threshold(X_final.values, target_var)
                dpr_cat.name = "dPr_cat"
                
                # 合并导出表格
                by_cut_table = pd.concat([coef_df, dpr_ge], axis=1)
                by_cut_table.index.name = "threshold"
                
                # 保存结果到字典（后续写报告时使用）
                models[(ctry, 'threshold_analysis')] = {
                    'coef_by_cut': coef_by_cut,
                    'dpr_ge': dpr_ge,
                    'dpr_cat': dpr_cat,
                    'by_cut_table': by_cut_table
                }
                
                print(f"  {ctry}: rarity_score_z 门槛系数与阈值效应：")
                print(by_cut_table.round(6))
                print(f"  {ctry}: rarity_score_z 各类别概率效应：")
                print(dpr_cat.round(6))
                
                # 生成文字摘要
                def _sign(x): 
                    return "正" if x > 0.01 else ("负" if x < -0.01 else "≈0")
                
                # 门槛可读名称映射
                cuts_readable = {
                    "cut1": "Junior→Assistant",
                    "cut2": "Assistant→Regular", 
                    "cut3": "Regular→Senior",
                    "cut4": "Senior→Leader",
                    "cut5": "Leader→Chief/founder"
                }
                
                # 构造摘要文字
                threshold_lines = []
                for cut_name, coef_val in coef_by_cut.items():
                    readable_name = cuts_readable.get(cut_name, cut_name)
                    dpr_val = dpr_ge[cut_name] if cut_name in dpr_ge.index else 0
                    threshold_lines.append(
                        f"{readable_name}: β≈{coef_val:.3f}（阈值效应≈{dpr_val:.4f}）"
                    )
                
                summary_txt = f"[{ctry}] rarity_score_z的门槛异质性：" + "； ".join(threshold_lines)
                models[(ctry, 'threshold_summary')] = summary_txt
                
                print(f"  {ctry}: 门槛异质性摘要: {summary_txt}")
                
            except Exception as e:
                print(f"  {ctry}: 门槛级别分析失败: {e}")
                models[(ctry, 'threshold_analysis')] = None
                models[(ctry, 'threshold_summary')] = f"[{ctry}] 门槛分析失败: {e}"

            results.append({
                'country': ctry,
                'method': 'Gologit2_Lag',
                'coef_rarity_z': coef,
                'se': se,
                'pval': pval,
                'PseudoR2': pseudo_r2,
                'N': n,
                'AME_rarity_per1SD': ame_rarity
            })
            ordered_success = True
            
            success_msg = f"✓ {ctry}: 时间滞后gologit2回归成功（rarity_score 标准化，1单位=1SD）"
            if not np.isnan(coef):
                success_msg += f"，系数={coef:.4f}"
                if not np.isnan(se):
                    success_msg += f" (SE={se:.4f})"
                if not np.isnan(pval):
                    success_msg += f" (p={pval:.4f})"
            if not np.isnan(pseudo_r2):
                success_msg += f", Pseudo-R²={pseudo_r2:.4f}"
            if not np.isnan(ame_rarity):
                success_msg += f", AME(E[Y])={ame_rarity:.4f} per +1SD"
            print(success_msg)
            
            # === 计算每个变量对各类别概率的 APE（边际效应） ===
            seniority_levels = seniority_order  # 使用实际的级别名称
            per_var_ape = {}
            
            exog_cols = feature_names  # 模型里的所有解释变量
            print(f"  {ctry}: 计算所有变量的边际效应，变量数: {len(exog_cols)}")
            
            for v in exog_cols:
                try:
                    ape_obj = compute_ordered_marginal_effects(model, X_final, v, seniority_levels)
                    # ape_obj['effects'] 是长度为6的数组：对 P(Y=0..5) 的 APE
                    per_var_ape[v] = {
                        "effects": ape_obj["effects"],   # np.ndarray, shape (6,)
                        "se": ape_obj["se"],            # gologit2暂不提供标准误
                        "note": ape_obj["summary"]      # 计算方法说明
                    }
                except Exception as ape_e:
                    per_var_ape[v] = {"effects": None, "se": None, "note": f"APE failed: {ape_e}"}
            
            models[(ctry, 'ordered_logit_lag_ape_allvars')] = per_var_ape
            print(f"  {ctry}: 边际效应计算完成")
            
        except Exception as e:
            print(f"⚠ {ctry}: gologit2回归失败: {str(e)}")
            ordered_success = False
    
    # 方法2: 如果序数回归失败，尝试简化模型
    if not ordered_success:
        print(f"  {ctry}: 尝试简化的时间滞后序数回归...")
        
        try:
            # 只使用 rarity_score_z 和几个主要控制变量
            essential_controls = []
            if 'work_years' in num_controls:
                essential_controls.append('work_years')
            if 'gender_female' in num_controls:
                essential_controls.append('gender_female')
            
            X_simple = dfc[['rarity_score_z'] + essential_controls].copy()

            # 加入时间固定效应到简化模型（Year FE）
            if 'year_current' in dfc.columns:
                year_cat = dfc['year_current'].astype(int).astype('category')
                year_dummies = pd.get_dummies(year_cat, drop_first=True, prefix='year', dtype=float)
                if not year_dummies.empty:
                    X_simple = pd.concat([X_simple, year_dummies], axis=1)
            # 去重列名
            if X_simple.columns.duplicated().any():
                X_simple = X_simple.loc[:, ~X_simple.columns.duplicated()]
            
            # 确保数据类型正确
            for col in list(X_simple.columns):
                s = X_simple[col]
                if pd.api.types.is_object_dtype(s):
                    X_simple[col] = pd.to_numeric(s, errors='coerce')
                elif pd.api.types.is_bool_dtype(s):
                    X_simple[col] = s.astype(float)
                if X_simple[col].nunique(dropna=False) <= 1:
                    X_simple.drop(columns=[col], inplace=True)
            
            # 清理数据
            mask = X_simple.notna().all(axis=1) & y_ordinal.notna()
            X_simple_clean = X_simple.loc[mask]
            y_simple_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有数据类型都是数值型
            for col in list(X_simple_clean.columns):
                s = X_simple_clean[col]
                if not pd.api.types.is_numeric_dtype(s):
                    X_simple_clean[col] = s.astype(float)
            
            # === 增强简化模型数据验证 ===
            problem_cols_simple = []
            for col in list(X_simple_clean.columns):
                nunique = X_simple_clean[col].nunique(dropna=False)
                if nunique <= 1:
                    problem_cols_simple.append(f"{col}(nunique={nunique})")
            
            if problem_cols_simple:
                print(f"  {ctry}: 简化模型 WARNING: Found constant columns: {problem_cols_simple}")
                for col in [c.split('(')[0] for c in problem_cols_simple]:
                    if col in X_simple_clean.columns:
                        X_simple_clean.drop(columns=[col], inplace=True)
                print(f"  {ctry}: 简化模型 Removed {len(problem_cols_simple)} problematic columns")
            
            # 验证简化模型数据质量
            final_nunique_simple = {col: X_simple_clean[col].nunique() for col in X_simple_clean.columns}
            min_nunique_simple = min(final_nunique_simple.values()) if final_nunique_simple else 0
            
            print(f"  {ctry}: 简化模型，变量: {list(X_simple_clean.columns)}, 观测值: {len(y_simple_clean)}")
            print(f"  {ctry}: 简化模型 特征变异性: 最小nunique={min_nunique_simple}")
            
            # 安全检查
            if 'rarity_score_z' not in X_simple_clean.columns:
                print(f"  {ctry}: 简化模型 ERROR: rarity_score_z missing after cleaning!")
                continue
            
            # 使用自定义gologit2简化模型
            feature_names_simple = list(X_simple_clean.columns)
            
            # === SIMPLIFIED PPOM STRATEGY ===
            # For simplified model, use conservative approach: most variables parallel, only test key variable
            print(f"  {ctry}: 简化模型PPOM策略（保守方法）...")
            
            # 简化模型通常变量较少，使用保守的策略
            # 将所有控制变量设为parallel，只让rarity_score_z可能non-parallel
            conservative_parallel = [col for col in feature_names_simple if col != 'rarity_score_z']
            candidate_npl_simple = ['rarity_score_z'] if 'rarity_score_z' in feature_names_simple else []
            
            print(f"  {ctry}: 简化模型 - 保守parallel变量: {conservative_parallel}")
            print(f"  {ctry}: 简化模型 - 候选非平行变量: {candidate_npl_simple}")
            
            # 对于简化模型，由于变量少，直接测试rarity_score_z是否需要非平行
            if candidate_npl_simple:
                print(f"  {ctry}: 测试rarity_score_z是否违反比例优势假设...")
                
                # 全parallel模型
                baseline_simple = GeneralizedOrderedModel(link='logit', pl_vars=feature_names_simple.copy())
                baseline_result_simple = baseline_simple.fit(
                    X_simple_clean.values, 
                    y_simple_clean.values,
                    feature_names=feature_names_simple,
                    verbose=False,
                    maxiter=2000
                )
                
                if baseline_result_simple.success:
                    baseline_ll_simple = -baseline_result_simple.fun
                    
                    # 放开rarity_score_z的模型
                    test_simple = GeneralizedOrderedModel(link='logit', pl_vars=conservative_parallel)
                    test_result_simple = test_simple.fit(
                        X_simple_clean.values,
                        y_simple_clean.values, 
                        feature_names=feature_names_simple,
                        verbose=False,
                        maxiter=2000
                    )
                    
                    if test_result_simple.success:
                        test_ll_simple = -test_result_simple.fun
                        lr_stat_simple = 2 * (test_ll_simple - baseline_ll_simple)
                        K_simple = len(baseline_result_simple.alphas)
                        df_simple = K_simple - 2
                        p_value_simple = 1 - stats.chi2.cdf(lr_stat_simple, df_simple) if lr_stat_simple > 0 else 1.0
                        
                        print(f"  {ctry}: 简化模型LR检验: LR={lr_stat_simple:.2f}, p={p_value_simple:.4f} (df={df_simple})")
                        
                        if p_value_simple < 0.05:  # 更宽松的显著性水平
                            # rarity_score_z显著违反比例优势，使用非平行
                            model_simple = test_simple
                            res_simple = test_result_simple
                            pl_vars_simple = conservative_parallel
                            non_parallel_simple = candidate_npl_simple
                            print(f"  {ctry}: 简化模型选择：rarity_score_z非平行 (p={p_value_simple:.4f})")
                        else:
                            # 保持全parallel
                            model_simple = baseline_simple 
                            res_simple = baseline_result_simple
                            pl_vars_simple = feature_names_simple.copy()
                            non_parallel_simple = []
                            print(f"  {ctry}: 简化模型选择：全parallel (p={p_value_simple:.4f} > 0.05)")
                    else:
                        # 非平行模型拟合失败，使用全parallel
                        model_simple = baseline_simple
                        res_simple = baseline_result_simple
                        pl_vars_simple = feature_names_simple.copy()
                        non_parallel_simple = []
                        print(f"  {ctry}: 简化模型：非平行模型拟合失败，使用全parallel")
                else:
                    # 基准模型拟合失败，使用保守策略
                    pl_vars_simple = conservative_parallel
                    non_parallel_simple = candidate_npl_simple
                    model_simple = GeneralizedOrderedModel(link='logit', pl_vars=pl_vars_simple)
                    res_simple = model_simple.fit(
                        X_simple_clean.values,
                        y_simple_clean.values,
                        feature_names=feature_names_simple,
                        maxiter=2000,
                        verbose=False
                    )
                    print(f"  {ctry}: 简化模型：基准模型拟合失败，使用保守策略")
            else:
                # 没有候选变量，全parallel
                pl_vars_simple = feature_names_simple.copy()
                non_parallel_simple = []
                model_simple = GeneralizedOrderedModel(link='logit', pl_vars=pl_vars_simple) 
                res_simple = model_simple.fit(
                    X_simple_clean.values,
                    y_simple_clean.values,
                    feature_names=feature_names_simple,
                    maxiter=2000,
                    verbose=False
                )
                print(f"  {ctry}: 简化模型：无候选非平行变量，全parallel")
            
            print(f"  {ctry}: 简化模型最终 - Parallel: {pl_vars_simple}")
            print(f"  {ctry}: 简化模型最终 - Non-parallel: {non_parallel_simple}")
            print(f"  {ctry}: 简化模型拟合完成：success={res_simple.success}")
            
            models[(ctry,'ordered_logit_lag_simple')] = model_simple
            models[(ctry,'ordered_logit_lag_simple_result')] = res_simple

            # 从简化模型结果中提取系数信息
            coef = np.nan
            se = np.nan  
            pval = np.nan
            
            # === 修正简化模型特征索引错位问题 ===
            # 使用模型存储的训练时变量顺序，而不是推断
            npl_names_simple = model_simple.npl_vars_ if hasattr(model_simple, 'npl_vars_') and model_simple.npl_vars_ else [col for col in feature_names_simple if col not in pl_vars_simple]
            print(f"  {ctry}: 简化模型 Non-parallel variable order (from model): {npl_names_simple}")
            
            if 'rarity_score_z' in npl_names_simple:
                var_idx = npl_names_simple.index('rarity_score_z')  # 在非平行变量列表中的正确索引
                print(f"  {ctry}: 简化模型 rarity_score_z index in npl_names: {var_idx}")
                
                if res_simple.beta_npl is not None:
                    print(f"  {ctry}: 简化模型 beta_npl shape: {res_simple.beta_npl.shape}")
                    if var_idx < res_simple.beta_npl.shape[1]:  # 确保索引不越界
                        coef = float(res_simple.beta_npl[0, var_idx])  # 使用第一个阈值的系数作为代表
                    else:
                        print(f"  {ctry}: 简化模型 WARNING: var_idx {var_idx} >= beta_npl.shape[1] {res_simple.beta_npl.shape[1]}")
                
                # 提取标准误和p值（如果可用）
                if hasattr(res_simple, 'se_beta_npl') and res_simple.se_beta_npl is not None:
                    if var_idx < res_simple.se_beta_npl.shape[1]:
                        se = float(res_simple.se_beta_npl[0, var_idx])  # 第一个阈值的标准误
                
                if hasattr(res_simple, 'pvalues_beta_npl') and res_simple.pvalues_beta_npl is not None:
                    if var_idx < res_simple.pvalues_beta_npl.shape[1]:
                        pval = float(res_simple.pvalues_beta_npl[0, var_idx])  # 第一个阈值的p值
            else:
                print(f"  {ctry}: 简化模型 WARNING: rarity_score_z not found in non-parallel variables: {npl_names_simple}")
            
            # 提取pseudo R2
            pseudo_r2 = getattr(res_simple, 'pseudo_r2', np.nan)
            n = len(y_simple_clean)
            
            # 简化模型的期望值AME
            ame_rarity_s, ame_type_s = compute_ame_expected(model_simple, X_simple_clean, 'rarity_score_z')
            models[(ctry,'ordered_logit_lag_simple_ame_rarity')] = {'ame': ame_rarity_s, 'type': ame_type_s}
            
            # === 简化模型门槛级别详细分析 ===
            print(f"  {ctry}: 简化模型门槛级别分析...")
            
            try:
                target_var = "rarity_score_z"
                
                # 1) 各门槛"斜率"表（β_j）
                coef_by_cut_simple = res_simple.coef_by_threshold(target_var)
                coef_df_simple = coef_by_cut_simple.to_frame(name="coef")
                
                # 2) 各门槛"阈值概率效应"（∂Pr{Y≥j}/∂x）（自动选择dx）
                dpr_ge_simple = res_simple.dPr_ge_by_threshold(X_simple_clean.values, target_var)
                dpr_ge_simple.name = "dPr_ge"
                
                # 3) 各类别"概率效应"（∂Pr{Y=c}/∂x）（自动选择dx）
                dpr_cat_simple = res_simple.dPr_cat_by_threshold(X_simple_clean.values, target_var)
                dpr_cat_simple.name = "dPr_cat"
                
                # 合并导出表格
                by_cut_table_simple = pd.concat([coef_df_simple, dpr_ge_simple], axis=1)
                by_cut_table_simple.index.name = "threshold"
                
                # 保存简化模型结果
                models[(ctry, 'threshold_analysis_simple')] = {
                    'coef_by_cut': coef_by_cut_simple,
                    'dpr_ge': dpr_ge_simple,
                    'dpr_cat': dpr_cat_simple,
                    'by_cut_table': by_cut_table_simple
                }
                
                print(f"  {ctry}: 简化模型 rarity_score_z 门槛系数与阈值效应：")
                print(by_cut_table_simple.round(6))
                print(f"  {ctry}: 简化模型 rarity_score_z 各类别概率效应：")
                print(dpr_cat_simple.round(6))
                
                # 生成简化模型摘要
                cuts_readable = {
                    "cut1": "Junior→Assistant",
                    "cut2": "Assistant→Regular", 
                    "cut3": "Regular→Senior",
                    "cut4": "Senior→Leader",
                    "cut5": "Leader→Chief/founder"
                }
                
                threshold_lines_simple = []
                for cut_name, coef_val in coef_by_cut_simple.items():
                    readable_name = cuts_readable.get(cut_name, cut_name)
                    dpr_val = dpr_ge_simple[cut_name] if cut_name in dpr_ge_simple.index else 0
                    threshold_lines_simple.append(
                        f"{readable_name}: β≈{coef_val:.3f}（阈值效应≈{dpr_val:.4f}）"
                    )
                
                summary_txt_simple = f"[{ctry}] 简化模型 rarity_score_z的门槛异质性：" + "； ".join(threshold_lines_simple)
                models[(ctry, 'threshold_summary_simple')] = summary_txt_simple
                
                print(f"  {ctry}: 简化模型门槛异质性摘要: {summary_txt_simple}")
                
            except Exception as e:
                print(f"  {ctry}: 简化模型门槛级别分析失败: {e}")
                models[(ctry, 'threshold_analysis_simple')] = None
                models[(ctry, 'threshold_summary_simple')] = f"[{ctry}] 简化模型门槛分析失败: {e}"

            results.append({
                'country': ctry,
                'method': 'Gologit2_Lag_Simple',
                'coef_rarity_z': coef,
                'se': se,
                'pval': pval,
                'PseudoR2': pseudo_r2,
                'N': n,
                'AME_rarity_per1SD': ame_rarity_s
            })
            
            success_msg = f"✓ {ctry}: 简化时间滞后gologit2回归成功（rarity_score 标准化，1单位=1SD）"
            if not np.isnan(coef):
                success_msg += f"，系数={coef:.4f}"
                if not np.isnan(se):
                    success_msg += f" (SE={se:.4f})"
                if not np.isnan(pval):
                    success_msg += f" (p={pval:.4f})"
            if not np.isnan(pseudo_r2):
                success_msg += f", Pseudo-R²={pseudo_r2:.4f}"
            if not np.isnan(ame_rarity_s):
                success_msg += f", AME(E[Y])={ame_rarity_s:.4f} per +1SD"
            print(success_msg)
            
            # === 计算简化模型每个变量对各类别概率的 APE ===
            seniority_levels = seniority_order
            per_var_ape_simple = {}
            
            exog_cols_simple = feature_names_simple
            print(f"  {ctry}: 计算简化模型所有变量的边际效应，变量数: {len(exog_cols_simple)}")
            
            for v in exog_cols_simple:
                try:
                    ape_obj = compute_ordered_marginal_effects(model_simple, X_simple_clean, v, seniority_levels)
                    per_var_ape_simple[v] = {
                        "effects": ape_obj["effects"],
                        "se": ape_obj["se"],
                        "note": ape_obj["summary"]
                    }
                except Exception as ape_e:
                    per_var_ape_simple[v] = {"effects": None, "se": None, "note": f"APE failed: {ape_e}"}
            
            models[(ctry, 'ordered_logit_lag_simple_ape_allvars')] = per_var_ape_simple
            print(f"  {ctry}: 简化模型边际效应计算完成")
            
        except Exception as e2:
            print(f"✗ {ctry}: 所有时间滞后gologit2回归方法都失败了: {str(e2)}")
            results.append({'country': ctry, 'method': 'FAILED', 'coef_rarity_z': np.nan, 'se': np.nan, 'pval': np.nan, 'PseudoR2': np.nan, 'N': 0, 'AME_rarity_per1SD': np.nan})

lag_results_summary = pd.DataFrame(results).sort_values(['country']).reset_index(drop=True)
print('\n' + '='*60)
print('时间滞后回归结果：t年rarity_score预测t+1年seniority')
print('='*60)
print(lag_results_summary)

# === 系统化输出逐阈值系数表到CSV文件 ===
threshold_coef_data = []

print(f"\n生成逐阈值系数表...")
for ctry in countries:
    # 寻找该国家可用的结果对象
    result_obj = None
    model_type = None
    
    if (ctry, 'ordered_logit_lag_result') in models:
        result_obj = models[(ctry, 'ordered_logit_lag_result')]
        model_type = 'Full_Model'
    elif (ctry, 'ordered_logit_lag_simple_result') in models:
        result_obj = models[(ctry, 'ordered_logit_lag_simple_result')]
        model_type = 'Simple_Model'
    
    if result_obj is None or not result_obj.success:
        print(f"  {ctry}: 无可用结果对象")
        continue
    
    # 获取非并行变量
    if hasattr(result_obj, 'pl_vars') and result_obj.pl_vars:
        npl_vars = [v for v in result_obj.feature_names if v not in result_obj.pl_vars]
    else:
        npl_vars = result_obj.feature_names  # 全部非并行
    
    if not npl_vars:
        print(f"  {ctry}: 无非并行变量")
        continue
        
    print(f"  {ctry}: 处理 {len(npl_vars)} 个非并行变量: {npl_vars}")
    
    # 为每个非并行变量生成阈值系数行
    for var in npl_vars:
        try:
            coef_by_threshold = result_obj.coef_by_threshold(var)
            
            # 提取标准误和p值（如果可用）
            se_by_threshold = None
            pval_by_threshold = None
            
            if hasattr(result_obj, 'se_beta_npl') and result_obj.se_beta_npl is not None:
                var_idx = npl_vars.index(var)
                if var_idx < result_obj.se_beta_npl.shape[1]:
                    se_by_threshold = result_obj.se_beta_npl[:, var_idx]
            
            if hasattr(result_obj, 'pvalues_beta_npl') and result_obj.pvalues_beta_npl is not None:
                var_idx = npl_vars.index(var)
                if var_idx < result_obj.pvalues_beta_npl.shape[1]:
                    pval_by_threshold = result_obj.pvalues_beta_npl[:, var_idx]
            
            # 生成每个阈值的记录
            for i, (cut_name, coef_val) in enumerate(coef_by_threshold.items()):
                row = {
                    'country': ctry,
                    'model_type': model_type,
                    'variable': var,
                    'threshold': cut_name,
                    'threshold_readable': {
                        "cut1": "Junior→Assistant",
                        "cut2": "Assistant→Regular", 
                        "cut3": "Regular→Senior",
                        "cut4": "Senior→Leader",
                        "cut5": "Leader→Chief/founder"
                    }.get(cut_name, cut_name),
                    'coefficient': coef_val,
                    'std_error': se_by_threshold[i] if se_by_threshold is not None else np.nan,
                    'p_value': pval_by_threshold[i] if pval_by_threshold is not None else np.nan,
                    'significant_05': (pval_by_threshold[i] < 0.05) if pval_by_threshold is not None else np.nan,
                    'significant_01': (pval_by_threshold[i] < 0.01) if pval_by_threshold is not None else np.nan
                }
                threshold_coef_data.append(row)
                
        except Exception as e:
            print(f"  {ctry}: 处理变量 {var} 时出错: {e}")
            continue

# 保存阈值系数表
if threshold_coef_data:
    threshold_df = pd.DataFrame(threshold_coef_data)
    threshold_csv_path = os.path.join(os.path.dirname(__file__), "threshold_coefficients_detailed.csv")
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"详细阈值系数表已保存到: {threshold_csv_path}")
    print(f"包含 {len(threshold_df)} 行记录，涵盖 {threshold_df['country'].nunique()} 个国家")
else:
    print("未生成阈值系数数据")

# === 将结果写入一个 txt 报告 ===
import datetime

report_path = os.path.join(os.path.dirname(__file__), "time_lag_regression_report.txt")
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Time Lag Ordinal Regression Report\nGenerated: {ts}\n\n")
    f.write("RESEARCH QUESTION: Does t-year rarity_score predict t+1 year seniority?\n\n")
    f.write("IMPORTANT: seniority is an ordinal variable (Junior < Assistant < Regular < Senior < Leader < Chief or founder)\n\n")
    f.write("Statistical Methods Used:\n")
    f.write("1. Primary: Generalized Ordered Model (gologit2) with time lag\n")
    f.write("2. Fallback: Simplified Generalized Ordered Model with essential controls only\n")
    f.write("Note: Using custom Python implementation of gologit2 (similar to Stata's gologit2)\n\n")
    f.write("Time Lag Structure:\n")
    f.write("- Explanatory variables (X): t-year data (rarity_score, controls)\n")
    f.write("- Outcome variable (Y): t+1-year seniority\n")
    f.write("- Note: Final year observations are dropped (no t+1 data available)\n\n")
    f.write("Controls used:\n")
    f.write(f"- Numeric controls: {num_controls}\n")
    f.write(f"- Categorical controls (one-hot): {cat_controls}\n")
    f.write("- rarity_score standardized to z-score within country (1 unit = +1 SD)\n\n")
    f.write("Seniority encoding:\n")
    for level, code in seniority_mapping.items():
        f.write(f"- {level} -> {code}\n")
    f.write("\nNotes:\n")
    f.write("- gender encoded as gender_female: female=1, male=0; other/unknown as NaN\n")
    f.write("- Generalized ordered model (gologit2) is appropriate for ordinal outcomes\n")
    f.write("- Allows different coefficients across thresholds (non-proportional odds)\n")
    f.write("- Positive coefficient: higher rarity predicts higher future seniority\n")
    f.write("- Negative coefficient: higher rarity predicts lower future seniority\n")
    f.write("- Standard errors computed via numerical Hessian (when successful)\n")
    f.write("- P-values based on z-tests (coefficient/standard error)\n")
    f.write("- Pseudo R² is McFadden's R² = 1 - (logL_full/logL_null)\n")
    f.write("- Note: gologit2 uses different parameterization than standard ordered logit\n\n")

    # 固定效应与面板说明
    f.write("Fixed effects included:\n")
    f.write("- Year FE via dummies on year_current (drop_first)\n")
    f.write("- No individual FE; pooled across workers\n\n")

    f.write("Country-level summary (one row per country):\n")
    f.write(lag_results_summary.to_string(index=False))
    f.write("\n\n")

    # 写入各国"系数的平均边际效应（对期望职级E[Y]）"
    f.write("Average Marginal Effects (on expected seniority, E[Y]) by Country:\n\n")
    f.write("IMPORTANT NOTE: For gologit2 non-parallel variables, AME may differ from individual coefficients.\n")
    f.write("This is normal because:\n")
    f.write("1. gologit2 coefficients represent 'cutpoint-specific differences', not 'global slopes'\n")
    f.write("2. Non-parallel variables can have opposite effects at different thresholds\n")
    f.write("3. AME is a weighted average across all thresholds, closer to actual average effect\n")
    f.write("4. Use AME for interpretation, not individual coefficient values\n\n")
    
    for ctry in countries:
        ame_key = (ctry, 'ordered_logit_lag_ame_rarity') if (ctry, 'ordered_logit_lag_ame_rarity') in models else (
                  (ctry, 'ordered_logit_lag_simple_ame_rarity') if (ctry, 'ordered_logit_lag_simple_ame_rarity') in models else None)
        if ame_key is None:
            f.write(f"Country: {ctry} - 无AME结果\n\n")
            continue
        ame_obj = models[ame_key]
        ame = ame_obj.get('ame', np.nan)
        ame_type = ame_obj.get('type', '')
        f.write(f"- {ctry}: AME(E[Y]) for rarity_score (per +1 SD) = {ame:.6f} ({ame_type})\n")
    f.write("\n")
    
    # === 新增：门槛级别详细分析结果 ===
    f.write("=" * 80 + "\n")
    f.write("THRESHOLD-LEVEL ANALYSIS: rarity_score_z Effects by Career Transition\n") 
    f.write("=" * 80 + "\n")
    f.write("This section reveals the heterogeneous effects across different career thresholds,\n")
    f.write("which is the key advantage of gologit2 over standard ordered logit.\n\n")
    
    f.write("KEY FINDINGS & PPOM INTERPRETATION:\n")
    f.write("- coef: β_j coefficient for threshold j (raw slope parameter)\n")
    f.write("- dPr_ge: ∂Pr(Y≥j)/∂x (effect on probability of reaching level j or higher)\n")
    f.write("- PPOM Advantage: Reveals where non-parallel effects occur across career ladder\n")
    f.write("- Direction vs. Intensity: Some variables affect early transitions (direction),\n")
    f.write("  others affect senior-level transitions (intensity), or show mixed patterns\n")
    f.write("- Positive β_j: rarity facilitates crossing that specific threshold\n")
    f.write("- Negative β_j: rarity hinders crossing that specific threshold\n")
    f.write("- Interpretation Framework (aligned with Williams 2016):\n")
    f.write("  * Consistent signs across thresholds: uniform career advantage/disadvantage\n")
    f.write("  * Varying magnitudes: differential effects at different career stages\n")
    f.write("  * Sign changes: non-monotonic effects (e.g., helps early but hinders later)\n\n")
    
    for ctry in countries:
        # 寻找门槛分析结果（优先使用完整模型，回退到简化模型）
        threshold_key = (ctry, 'threshold_analysis') if (ctry, 'threshold_analysis') in models else (
                       (ctry, 'threshold_analysis_simple') if (ctry, 'threshold_analysis_simple') in models else None)
        
        summary_key = (ctry, 'threshold_summary') if (ctry, 'threshold_summary') in models else (
                     (ctry, 'threshold_summary_simple') if (ctry, 'threshold_summary_simple') in models else None)
        
        if threshold_key is None or models[threshold_key] is None:
            f.write(f"\nCountry: {ctry}\n")
            f.write("  Threshold analysis: Not available (model fitting failed)\n")
            continue
            
        f.write(f"\nCountry: {ctry}\n")
        f.write("-" * 40 + "\n")
        
        # 写入门槛系数表格
        threshold_data = models[threshold_key]
        by_cut_table = threshold_data['by_cut_table']
        
        f.write("Threshold Effects Table:\n")
        f.write(by_cut_table.to_string(float_format='%.6f'))
        f.write("\n\n")
        
        # 写入各类别概率效应
        dpr_cat = threshold_data['dpr_cat']
        f.write("Category Probability Effects:\n")
        f.write(dpr_cat.to_string(float_format='%.6f'))
        f.write("\n\n")
        
        # 写入文字摘要
        if summary_key and summary_key in models:
            summary_text = models[summary_key]
            f.write("Summary:\n")
            f.write(summary_text)
            f.write("\n\n")
        
        # 解读提示
        coef_by_cut = threshold_data['coef_by_cut']
        significant_effects = [(cut, coef) for cut, coef in coef_by_cut.items() if abs(coef) > 0.02]
        
        if significant_effects:
            f.write("Key Insights:\n")
            cuts_readable = {
                "cut1": "Junior→Assistant",
                "cut2": "Assistant→Regular", 
                "cut3": "Regular→Senior",
                "cut4": "Senior→Leader",
                "cut5": "Leader→Chief/founder"
            }
            
            for cut, coef in significant_effects:
                direction = "facilitates" if coef > 0 else "hinders"
                readable = cuts_readable.get(cut, cut)
                f.write(f"  - {readable}: rarity {direction} this transition (β={coef:.3f})\n")
        else:
            f.write("Key Insights: No strong threshold-specific effects detected\n")
        
        f.write("\n" + "-" * 40 + "\n")
    
    # === 添加PPOM非平行性总结 ===
    f.write("\n" + "=" * 80 + "\n")
    f.write("PPOM NON-PARALLEL EFFECTS SUMMARY: Where Proportional Odds Assumption Fails\n")
    f.write("=" * 80 + "\n")
    f.write("This summary identifies threshold-specific violations of proportional odds,\n")
    f.write("which is the core insight that PPOM provides over standard ordered logit.\n\n")
    
    non_parallel_summary = {}
    for ctry in countries:
        # 检查是否有PPOM选择结果
        selection_key = (ctry, 'ppom_selection')
        if selection_key in models:
            selection_info = models[selection_key]
            selected_npl = selection_info.get('selected_npl_vars', [])
            if selected_npl:
                non_parallel_summary[ctry] = {
                    'selected_vars': selected_npl,
                    'improvement': selection_info.get('improvement', 0)
                }
    
    if non_parallel_summary:
        f.write("Countries with Significant Non-Parallel Effects Detected:\n\n")
        for ctry, info in non_parallel_summary.items():
            f.write(f"- {ctry}: {', '.join(info['selected_vars'])} (ΔLL: {info['improvement']:.4f})\n")
    else:
        f.write("No countries showed significant violations of proportional odds assumption.\n")
        f.write("This suggests the proportional odds model is adequate for most variables.\n")
    
    f.write("\nINTERPRETATION: Variables with non-parallel effects show differential impacts\n")
    f.write("across career stages, revealing where standard ordered logit assumptions break down.\n")
    f.write("\n")

    for ctry in countries:
        # 寻找该国家可用的模型
        model_key = None
        result_key = None
        
        if (ctry, 'ordered_logit_lag') in models:
            model_key = (ctry, 'ordered_logit_lag')
            result_key = (ctry, 'ordered_logit_lag_result')
        elif (ctry, 'ordered_logit_lag_simple') in models:
            model_key = (ctry, 'ordered_logit_lag_simple')
            result_key = (ctry, 'ordered_logit_lag_simple_result')
        
        if model_key is None or result_key is None or result_key not in models:
            f.write(f"Country: {ctry} - 时间滞后gologit2回归失败，无可用模型\n\n")
            continue
            
        f.write("=" * 80 + "\n")
        f.write(f"Country: {ctry} | Model: {model_key[1]} (gologit2)\n")
        f.write("=" * 80 + "\n")
        
        # 使用gologit2的结果对象来生成摘要
        result_obj = models[result_key]
        f.write(str(result_obj.summary()))
        f.write("\n\n")
        
        # 写入边际效应 (APE)
        ape_key = None
        if model_key[1] == 'ordered_logit_lag':
            ape_key = (ctry, 'ordered_logit_lag_ape_allvars')
        elif model_key[1] == 'ordered_logit_lag_simple':
            ape_key = (ctry, 'ordered_logit_lag_simple_ape_allvars')
        
        if ape_key and ape_key in models:
            f.write("Marginal Effects (APE) by Variable:\n")
            f.write("(Change in probability for each seniority level per 1-unit increase)\n")
            f.write("-" * 60 + "\n")
            per_var_ape = models[ape_key]
            
            for v, obj in per_var_ape.items():
                f.write(f"\nVariable: {v}\n")
                if obj["effects"] is None:
                    f.write(f"  APE calculation failed: {obj['note']}\n")
                    continue
                
                eff = obj["effects"]
                f.write("  Seniority Level Effects:\n")
                for j, level_name in enumerate(seniority_order):
                    if j < len(eff):
                        f.write(f"    {level_name} (level {j}): {eff[j]:.6f}\n")
                
                if obj["se"] is not None:
                    f.write("  (Standard errors available from analytical margins)\n")
                
                f.write(f"  Method: {obj['note']}\n")
            
            f.write("\n" + "-" * 60 + "\n")
        else:
            f.write("Marginal Effects (APE): Not computed for this model.\n\n")

print(f"\n完整时间滞后回归报告已写入: {report_path}")

# 比较同期与滞后效应
print(f"\n" + "="*60)
print("分析总结:")
print("="*60)
print("时间滞后分析帮助我们理解:")
print("1. t年的rarity_score是否能预测t+1年的职业发展")
print("2. 这种预测效应在不同国家是否不同")
print("3. 相比同期效应，滞后效应的大小和方向")
print("\n建议对比前一个分析（03_fixed_effects_model.py）的同期效应!")
