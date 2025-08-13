## 面板版 OQR 使用指南（panel_oqr）

本指南面向初学者，帮助你用面板设定（年份固定效应 + 性别×年份交互）来估计 Bayesian Ordinal Quantile Regression（OQR），并生成“女性−男性”的逐年差异曲线与概率比较。

依赖：numpy、pandas；可选：matplotlib（绘图）

### 1. 准备数据

数据需为“长表”（每人每年一行），至少包含：
- id: 个体ID（可选）
- year: 年份（任意可排序类型，如整数或字符串）
- seniority: 因变量（有序等级，整数 1..J）
- gender: 性别指示（建议 0=男性，1=女性）
- 可选控制变量：教育、工龄、行业、公司规模等（数值型或分类型均可）
- 可选 cohort：分组固定效应（如毕业届）

注意：
- 若 J==3，将自动使用 OR2 模型；若 J≥4，将使用 OR1 模型。
- 性别方向默认解释为 女性(1)−男性(0)。

### 2. 核心接口

从模块导入：
```python
from glm_plus.ordinal_quantile_regression.panel_oqr import (
    build_panel_design, fit_panel_oqr,
    extract_gender_year_effects, probability_top_greater_bottom,
    plot_gender_gap_trends,
)
```

### 3. 快速上手（最小可运行示例）

```python
import numpy as np
import pandas as pd

# 示例：构造一份长表 df（实际使用你自己的数据）
np.random.seed(42)
N, T = 200, 5
df = pd.DataFrame({
    'id': np.repeat(np.arange(N), T),
    'year': np.tile(np.arange(1, T+1), N),
})
df['gender'] = np.random.binomial(1, 0.5, size=len(df))
df['edu'] = np.random.choice(['low','mid','high'], size=len(df))
df['tenure'] = np.random.randint(0, 20, size=len(df))

# 人为生成一个 1..5 的有序因变量（示例用随机方式，实际请用真实数据）
df['seniority'] = np.random.choice([1,2,3,4,5], size=len(df), p=[0.2,0.3,0.3,0.15,0.05])

# 1) 构建设计矩阵：年固定效应 + 性别×年份交互 + 控制变量
design = build_panel_design(
    df=df,
    outcome_col='seniority',
    year_col='year',
    gender_col='gender',
    controls=['edu', 'tenure'],   # 可以混合数值/分类型
    cohort_col=None,              # 如需 cohort 固定效应，传列名
    drop_first_year=True,         # 丢弃基年，避免完全共线
    gender_is_female_one=True,    # 1 表示女性，0 表示男性
)

# 2) 在多个分位数上拟合（例如 0.2、0.5、0.8）
fit = fit_panel_oqr(
    design,
    burn=1000,
    mcmc=4000,
    quantiles=(0.2, 0.5, 0.8),
    verbose=False,
)

# 3) 提取逐年“女性−男性”差异（以 p=0.2 为例）
df20, draws20 = extract_gender_year_effects(fit, p=0.2)
print(df20)  # 列：year, mean, l95, u95

# 4) 比较顶部与底部的总体差异概率：P(diff_top > diff_bottom)
p_overall, per_year = probability_top_greater_bottom(
    fit, p_top=0.8, p_bottom=0.2, aggregate=True
)
print('P(top>bottom):', p_overall)

# 5) 可视化：三条分位数的时间曲线（需 matplotlib）
try:
    import matplotlib.pyplot as plt
    _ = plot_gender_gap_trends(
        fit, quantiles=(0.2, 0.5, 0.8),
        title='Female − Male readiness gap over time',
    )
    plt.show()
except Exception as e:
    print('绘图需要 matplotlib；若不可用，可跳过此步骤。')
```

### 4. 参数与返回值

- build_panel_design(
  - df: 长表 DataFrame
  - outcome_col: 因变量列（整数 1..J）
  - year_col: 年份列（会按分类处理，保持原始顺序）
  - gender_col: 性别列（建议 0/1）
  - controls: 控制变量列表；数值型原样使用，分类型会自动做哑变量（drop_first=True）
  - cohort_col: 可选 cohort 固定效应（哑变量，drop_first=True）
  - drop_first_year: 是否丢弃基年 dummy（True 时，用“主效应”代表基年差异）
  - gender_is_female_one: 若为 True，方向解释为 女性(1)−男性(0)
  ) → PanelDesign

- fit_panel_oqr(
  - design: 上一步返回的 PanelDesign
  - burn, mcmc: MCMC 设置
  - quantiles: 分位点，如 (0.2,0.5,0.8)
  - B0_scale, D0_scale: 先验规模（一般默认即可）
  - verbose: 是否打印采样摘要
  ) → PanelOQRFit（内含各分位数的模型结果）

- extract_gender_year_effects(fit, p)
  - 返回 (summary_df, draws_dict)
  - summary_df 列含义：
    - year: 年份（字符串）
    - mean: 该年“女性−男性”的后验均值（潜在“准备度”差）
    - l95/u95: 95% 后验区间
  - draws_dict: {year -> (1, nsim) 抽样向量}

- probability_top_greater_bottom(fit, p_top=0.8, p_bottom=0.2, aggregate=True)
  - 返回 (p_overall, per_year_table)
  - p_overall: 若 aggregate=True，先按年平均后比较；否则按年分别算 P(top>bottom) 再取均值

- plot_gender_gap_trends(fit, quantiles, ax=None, title=None)
  - 绘制时间趋势曲线并填充 95% 区间；返回 matplotlib Axes

### 5. 结果如何解释

- 时间曲线：纵轴为“女性−男性”的潜在准备度差。为正表示女性更高；为负相反。
- 可信区间：区间完全在 0 上方/下方，说明该年差异方向较稳。
- 概率 P(top>bottom)：若 > 0.8，可表述为“较高概率存在玻璃天花板（顶部差异大于底部）”。

### 6. 常见问题与提示

- 年份顺序：内部严格使用原始 `year_col` 的分类顺序，避免字符串排序造成 “10” 在 “2” 前。
- 基年处理：
  - drop_first_year=True：基年不含 dummy，基年的“女性−男性”差异由性别主效应给出；其他年份为“主效应 + 对应交互系数”。
  - drop_first_year=False：每年都有 dummy，所有年份差异都等于“主效应 + 交互系数”。
- J 的选择：因变量等级数 J==3 → OR2；J≥4 → OR1。
- 性别方向：默认 gender=1 代表女性。若你的编码相反，请在图或文档中明示方向解释。
- 稀疏年份：若某年样本太少，交互项可能弱识别；可在可视化中添加样本量注记，或合并年份。
- 个体固定效应：可将 id 做成哑变量加入 controls，但会显著增大维度；随机效应需要进一步的模型扩展。

### 7. 一个更贴近实战的片段

```python
# 真实数据时：
design = build_panel_design(
    df, 'seniority', 'year', 'gender',
    controls=['edu','tenure','industry','firm_size'],
    cohort_col='cohort',
    drop_first_year=True,
)
fit = fit_panel_oqr(design, burn=2000, mcmc=8000, quantiles=(0.2,0.5,0.8))
df20, _ = extract_gender_year_effects(fit, 0.2)
df50, _ = extract_gender_year_effects(fit, 0.5)
df80, _ = extract_gender_year_effects(fit, 0.8)
p_overall, per_year = probability_top_greater_bottom(fit, 0.8, 0.2, aggregate=True)
```

需要进一步的帮助（如加入部门/地区固定效应、做稳健性分析等），可以在 `controls` 中补充相应变量或调整分位点集合。


