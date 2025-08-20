## TORQUE 频率学版使用指南（FrequentistOQR）

### 这是什么
- 对有序响应变量做分位数回归，联合估计单调变换 h(y)（可选第二个变换 g）和线性索引 Xβ。
- 支持样本权重、自动选维（上限 k≤2）、预测区间与覆盖率评估、诊断与自助法置信区间。

### 安装依赖
```bash
pip install numpy scipy scikit-learn statsmodels matplotlib
```

### 导入
```python
from glm_plus.frequentist.torque import FrequentistOQR
import numpy as np
```

### 数据要求
- X: 二维数组 (n, p)
- y: 一维整数类别，取值 1..J（内部会做微抖动处理）
- sample_weight: 可选一维数组，长度 n

## 快速上手

### 最小例子（单指数）
```python
n, p = 1000, 5
rng = np.random.default_rng(0)
X = rng.normal(size=(n, p))
beta = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
# 生成一个有序y（示意）
z = X @ beta + rng.normal(scale=0.8, size=n)
y = np.clip(np.floor(np.interp(z, np.quantile(z, [0, 0.25, 0.5, 0.75, 1]), [1, 2, 3, 4, 5])).astype(int), 1, 5)

model = FrequentistOQR(
    quantiles=(0.25, 0.5, 0.75),
    use_two_index=False,       # 单指数
    auto_select_k=True,        # 自动选维（实现最多到 k≤2）
    subsample_n=2000           # 可选：加速 rank 目标（大样本建议启用）
).fit(X, y)

# 分位数预测（离散类别）
preds = model.predict_quantiles(X[:5])
# 连续尺度预测（用于诊断/作图）
preds_disc, preds_cont = model.predict_quantiles(X[:5], return_continuous=True)

# 预测区间
lo, hi = model.predict_interval(X[:5], tau_low=0.25, tau_high=0.75)
```

### 加权工作流
```python
w = rng.uniform(0.5, 2.0, size=n)
model = FrequentistOQR(use_two_index=True, auto_select_k=True).fit(X, y, sample_weight=w)
```
- 全流程加权：CANCOR 用加权白化，β1 的中位数回归、β2,τ 的分位回归都传权重。

### 双指数（k≤2）
```python
model = FrequentistOQR(
    use_two_index=True,
    auto_select_k=True
).fit(X, y)

info = model.summary()
# 包含：selected_k/selected_k_full，k_truncated（若选出 k>2，则截断到2），
# corr_xbeta1_xbeta2（样本内相关诊断），approximation（rank 网格与子样本配置）
```

## 评估与表格复现

### 覆盖率与区间长度分层（Table 6 风格）
```python
res = model.evaluate_intervals(X, y, tau_low=0.25, tau_high=0.75, sample_weight=w)
# 关键输出：
# res["mean_length"], res["mean_coverage"]
# res["length_hist"]                            # 各长度频数（list）
# res["coverage_by_length"], res["counts_by_length"]  # L=0..4，及“5+”聚合
```

### 报告用“系数=1”重标定（Table 3 风格）
将指定变量的系数缩放到 1（仅用于展示，不改模型参数）：
```python
report = model.get_reporting_scaled_betas(feature_index=0)
# report["beta1_scaled"]
# report["beta2_tau_scaled"][tau]
```

## 作图与诊断

### 单调变换 h 和 g
```python
import matplotlib.pyplot as plt
ax = model.plot_transforms()  # 返回两个轴：h 与 g
plt.show()
```

### 连续尺度的预测
```python
preds_cont = model.predict_quantiles_continuous(X[:5])
```

### 配置与可复现
```python
cfg = model.get_config()
# 包含 quantiles/use_two_index/auto_select_k/alpha_cancor
# rank_grid/t_grid/subsample_n/k_cap（k≤2）
```

## 自助法置信区间（分位数回归阶段）
```python
boot = model.bootstrap_inference(
    n_boot=400,
    block_length=None,   # 可选：移动区块自助用于时序/面板
    ci=0.95,
    random_state=123
)
# boot["beta1"]["se"], ["ci_low"], ["ci_high"]
# 若双指数：boot["beta2_tau"][tau]["se"/"ci_low"/"ci_high"]
```
说明：
- 该接口固定已估的 h(.)/g(.)，在每个自助样本内重估分位回归，捕捉 QR 阶段的不确定性；不包含变换估计的额外不确定性。

## 实用建议

- LSOA 复现实证：按“Wave1 协变量预测 Wave2 结果”的口径分割训练/验证，并使用样本权重，便于复现论文表格。
- 数值近似与子样本：rank 目标用有限网格 + Isotonic 的近似；子样本可降低 O(n^2) 成本。建议记录 `rank_grid_* / t_grid_* / subsample_n` 并做敏感度检查。
- 选维与实现范围：CANCOR 可选出 k>2；当前实现聚焦双指数（k≤2），若选出更高维，将截断（`summary()['k_truncated']`）。

## 常见问题

- 响应必须是整数类别吗？
  - 是。内部会为有序数据进行轻微抖动（jitter）。
- 权重如何传？
  - 在 `fit(X, y, sample_weight=w)` 传入一维数组，长度与样本一致。
- 依赖缺失？
  - 如遇 `scikit-learn` 或 `statsmodels` 未安装，按“安装依赖”一节安装即可。

## 简要 API 参考
- 类：
  - FrequentistOQR(quantiles, use_two_index, auto_select_k, alpha_cancor, n_spline_knots, spline_degree, rank_grid_n, t_grid_n, rank_grid_low/high, t_grid_low/high, subsample_n, random_state)
- 训练与预测：
  - fit(X, y, sample_weight=None)
  - predict_quantiles(X, quantiles=None, return_continuous=False)
  - predict_quantiles_continuous(X, quantiles=None)
  - predict_interval(X, tau_low=0.25, tau_high=0.75)
  - make_prediction_interval(X, tau_low=0.25, tau_high=0.75)
- 评估与报告：
  - evaluate_intervals(X, y_true, tau_low=0.25, tau_high=0.75, sample_weight=None)
  - get_reporting_scaled_betas(feature_index)
  - plot_transforms(ax=None)
  - summary()
  - get_config()
  - bootstrap_inference(n_boot=200, block_length=None, ci=0.95, random_state=None)

## 给小白的快速上手与参数选择

### 我需要改哪些参数？
- quantiles：你要的分位点。常用 (0.25, 0.5, 0.75)。做预测区间用 (0.1, 0.9) 或 (0.25, 0.75)。
- use_two_index：是否启用第二条指数（更灵活，计算更慢）。样本小或想要快，先设 False；样本中大（例如 n≥2000）且想提高精度，设 True。
- auto_select_k：建议 True。模型会用 CANCOR 选择维数，但实现最多到 k≤2，多出来会截断。
- sample_weight：如果有调查/抽样权重，一定传；没有就不用传。
- subsample_n：加速 rank 目标（O(n^2)）。
  - 经验：n≤2000 设 None；2000<n≤20000 设 2000–5000；n>20000 设 5000–10000。
- rank_grid_n、t_grid_n：网格密度。41–81 是常用范围；越大越慢，结果通常对密度不太敏感。
- rank_grid_low/high、t_grid_low/high：网格范围。默认 0.05–0.95 一般就好。
- random_state：为了可重复，建议设定。

### 单指数还是双指数？
- 单指数（use_two_index=False）：一条“综合得分”Xβ1 已够用，速度快、稳健，适合小样本或基线模型。
- 双指数（use_two_index=True）：在第一条指数的残差上再学习一条方向，能吸收更多结构，通常更准，但更慢。适合中大样本，或需要较高精度时。
- 诊断：`summary()['corr_xbeta1_xbeta2']` 应接近 0（已做去相关）。`summary()['k_truncated']` 若为 True，表示 CANCOR 认为还有更多结构，但实现聚焦双指数。

### 快速配方
- 最快入门（单指数）
```python
model = FrequentistOQR(quantiles=(0.25, 0.5, 0.75), use_two_index=False, auto_select_k=True, random_state=0)
model.fit(X, y)
```
- 稳健准确（双指数，大样本）
```python
model = FrequentistOQR(quantiles=(0.25, 0.5, 0.75), use_two_index=True, auto_select_k=True,
                       subsample_n=5000, random_state=0)
model.fit(X, y, sample_weight=w)  # 若有权重
```
- 做预测区间
```python
lo, hi = model.predict_interval(X_test, tau_low=0.25, tau_high=0.75)
res = model.evaluate_intervals(X_test, y_test, 0.25, 0.75, sample_weight=w_test)
```
- 需要系数不确定性
```python
boot = model.bootstrap_inference(n_boot=400, ci=0.95, random_state=0)
```

### 计算耗时与内存
- rank 目标的复杂度是 O(m^2)，其中 m=subsample_n（若未设则为 n）。大样本务必设置 `subsample_n`。
- 网格密度（rank_grid_n/t_grid_n）越大越慢。通常 41–81 足够。

### 结果不理想时怎么调
- 覆盖率偏低：增大区间跨度（例如从 0.25–0.75 换到 0.2–0.8），检查是否传入权重，适度提高网格密度，或先用单指数做基准对比。
- 训练太慢：设置 `subsample_n`（例如 3000–10000），或减小网格密度。
- 诊断提示相关大：已做去相关，这个相关仅作为事后诊断。一般不需要手动处理。

### 默认值是否需要改
- 一般不需要。先用默认，确认流程跑通，再做敏感性检查：改动 `subsample_n` 与网格密度，结果应稳定。


