# 有序定量回归（Ordinal Quantile Regression）使用教程

## 简介

这个教程将教你如何使用我们的Python模块来进行有序定量回归分析。有序定量回归是一种统计方法，用于分析有序类别结果变量（比如满意度评分、疾病严重程度等级等）。

## 什么是有序定量回归？

**传统回归 vs 有序定量回归：**
- 传统线性回归：预测连续数值（如身高、收入）
- 有序定量回归：预测有序类别（如"不满意/一般/满意"，"轻微/中度/严重"）

**为什么需要定量回归？**
- 定量回归不仅告诉你平均效应，还能告诉你在分布的不同位置（比如第25%、50%、75%分位数）的效应
- 比如：某个治疗可能对病情轻微的患者效果很好，但对重症患者效果一般

## 模型选择指南

我们提供两个模型：

### OR1模型（ori.py）- 适用于4个或更多类别
**什么时候用：**
- 你的结果变量有4个或更多有序类别
- 例如：疾病严重程度（无症状/轻微/中度/严重/极严重）
- 例如：学术成绩（A/B/C/D/F）

### OR2模型（orii.py）- 专门用于3个类别
**什么时候用：**
- 你的结果变量恰好有3个有序类别
- 例如：满意度（不满意/一般/满意）
- 例如：风险等级（低/中/高）

## 数据准备

### 1. 结果变量（y）
```python
# 必须是整数，从1开始编码
# 错误示例：[0, 1, 2] 或 ["低", "中", "高"]
# 正确示例：[1, 2, 3] 表示三个类别

import numpy as np

# 示例：100个观察值，3个类别
y = np.array([1, 2, 3, 2, 1, 3, 2, 1, 3, 2] * 10).reshape(-1, 1)
print(f"结果变量形状: {y.shape}")  # 应该是 (n, 1)
print(f"类别: {np.unique(y)}")     # 应该是 [1 2 3]
```

### 2. 协变量矩阵（x）
```python
# 包含截距项（第一列全为1）和其他预测变量
n = 100  # 样本量
k = 3    # 变量数（包括截距）

# 生成示例数据
np.random.seed(42)
x = np.column_stack([
    np.ones(n),                    # 截距项
    np.random.normal(0, 1, n),     # 连续变量1
    np.random.binomial(1, 0.5, n)  # 二元变量
])

print(f"协变量矩阵形状: {x.shape}")  # 应该是 (n, k)
```

### 3. 变量命名（可选）
```python
x_names = ["截距", "年龄", "性别"]  # 与x的列数对应
```

## 使用OR2模型（3类别）

### 基本用法
```python
from glm_plus.ordinal_quantile_regression.orii import quantregOR2
import numpy as np

# 1. 准备数据
n = 200
np.random.seed(123)

# 真实的回归系数
true_beta = np.array([0.5, 1.2, -0.8])

# 生成协变量
x = np.column_stack([
    np.ones(n),                     # 截距
    np.random.normal(0, 1, n),      # 连续预测变量
    np.random.binomial(1, 0.4, n)   # 分类预测变量
])

# 生成有序结果（简化版，实际会更复杂）
linear_pred = x @ true_beta
noise = np.random.normal(0, 1, n)
latent = linear_pred + noise

# 将连续潜变量转换为3个有序类别
y = np.ones((n, 1), dtype=int)
y[latent > 0] = 2
y[latent > 1.5] = 3

print(f"类别分布: {np.bincount(y.flatten())[1:]}")

# 2. 设置先验参数
k = x.shape[1]  # 变量数
b0 = np.zeros((k, 1))      # beta的先验均值
B0 = 10 * np.eye(k)        # beta的先验协方差矩阵
n0 = 5.0                   # sigma的先验形状参数
d0 = 8.0                   # sigma的先验尺度参数

# 3. 运行模型
result = quantregOR2(
    y=y,
    x=x,
    b0=b0,
    B0=B0,
    n0=n0,
    d0=d0,
    gammacp2=3.0,              # 中间切点（除了0之外）
    burn=1000,                 # 预热迭代次数
    mcmc=2000,                 # MCMC迭代次数
    p=0.5,                     # 分位数（0.5=中位数回归）
    verbose=True,              # 显示结果
    x_names=["截距", "连续变量", "分类变量"]
)
```

### 解读结果
```python
# 查看主要结果
print("后验均值 (beta):")
print(result['postMeanbeta'])

print("后验标准差 (beta):")
print(result['postStdbeta'])

print("DIC (越小越好):")
print(result['dicQuant']['DIC'])

print("边际似然对数:")
print(result['logMargLike'])

# 查看汇总表
print("完整汇总:")
print(result['summary'])
```

## 使用OR1模型（4+类别）

```python
from glm_plus.ordinal_quantile_regression.ori import quantregOR1
import numpy as np

# 1. 准备4类别数据
n = 200
np.random.seed(456)

x = np.column_stack([
    np.ones(n),
    np.random.normal(0, 1, n),
    np.random.normal(0, 1, n)
])

# 生成4个类别的结果
linear_pred = x @ np.array([0.0, 1.0, -0.5])
noise = np.random.normal(0, 1, n)
latent = linear_pred + noise

y = np.ones((n, 1), dtype=int)
y[latent > -1] = 2
y[latent > 0] = 3
y[latent > 1] = 4

print(f"类别分布: {np.bincount(y.flatten())[1:]}")

# 2. 设置先验
k = x.shape[1]
J = len(np.unique(y))  # 类别数

b0 = np.zeros((k, 1))
B0 = 10 * np.eye(k)
d0 = np.zeros((J-2, 1))      # delta的先验均值
D0 = 0.25 * np.eye(J-2)      # delta的先验协方差

# 3. 运行模型
result = quantregOR1(
    y=y,
    x=x,
    b0=b0,
    B0=B0,
    d0=d0,
    D0=D0,
    burn=500,
    mcmc=1000,
    p=0.25,                    # 第25百分位数回归
    tune=0.1,                  # MH算法调优参数
    verbose=True
)
```

## 高级功能

### 1. 协变量效应分析
```python
from glm_plus.ordinal_quantile_regression.orii import covEffectOR2

# 比较增加连续变量0.1单位的效应
xMat1 = x.copy()  # 基线
xMat2 = x.copy()  
xMat2[:, 1] += 0.1  # 第二列（连续变量）增加0.1

effect = covEffectOR2(
    modelOR2=result,
    y=y,
    xMat1=xMat1,
    xMat2=xMat2,
    gammacp2=3.0,
    p=0.5,
    verbose=True
)

print("每个类别的概率变化:")
print(effect['avgDiffProb'])
```

### 2. 模型诊断
```python
# 检查收敛性
import matplotlib.pyplot as plt

# 绘制轨迹图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(result['betadraws'][0, :])  # 第一个beta的轨迹
plt.title('Beta 1 轨迹图')
plt.xlabel('迭代次数')

plt.subplot(1, 3, 2)
plt.plot(result['sigmadraws'][0, :])  # sigma的轨迹
plt.title('Sigma 轨迹图')
plt.xlabel('迭代次数')

plt.subplot(1, 3, 3)
plt.hist(result['betadraws'][1, 1000:], bins=30)  # 第二个beta的后验分布
plt.title('Beta 2 后验分布')
plt.xlabel('值')

plt.tight_layout()
plt.show()

# 检查无效因子（应该接近1）
print("无效因子:")
print(result['ineffactor'])
```

## 实际应用示例

### 案例：客户满意度分析
```python
# 假设我们有客户满意度数据
# y: 1=不满意, 2=一般, 3=满意
# 预测变量: 年龄, 服务时间, 是否VIP客户

import pandas as pd
import numpy as np

# 模拟真实数据
np.random.seed(789)
n = 500

data = pd.DataFrame({
    '年龄': np.random.normal(40, 15, n),
    '服务时间': np.random.exponential(2, n),  # 分钟
    'VIP客户': np.random.binomial(1, 0.3, n)
})

# 标准化连续变量
data['年龄_标准化'] = (data['年龄'] - data['年龄'].mean()) / data['年龄'].std()
data['服务时间_标准化'] = (data['服务时间'] - data['服务时间'].mean()) / data['服务时间'].std()

# 构建设计矩阵
x = np.column_stack([
    np.ones(n),                      # 截距
    data['年龄_标准化'],
    data['服务时间_标准化'],
    data['VIP客户']
])

# 生成满意度（基于合理的关系）
linear_pred = (0.2 * data['年龄_标准化'] + 
               -0.5 * data['服务时间_标准化'] +  # 服务时间长降低满意度
               0.8 * data['VIP客户'])           # VIP客户更满意

# 转换为有序类别
y = np.ones((n, 1), dtype=int)
y[linear_pred > -0.5] = 2
y[linear_pred > 0.5] = 3

print("满意度分布:")
labels = ['不满意', '一般', '满意']
for i, label in enumerate(labels, 1):
    count = np.sum(y == i)
    print(f"{label}: {count} ({count/n*100:.1f}%)")

# 分析第75百分位数的效应
result = quantregOR2(
    y=y,
    x=x,
    b0=np.zeros((4, 1)),
    B0=5 * np.eye(4),
    burn=1500,
    mcmc=3000,
    p=0.75,  # 关注高满意度客户
    verbose=True,
    x_names=["截距", "年龄", "服务时间", "VIP状态"]
)

# 解释结果
print("\n=== 结果解释 ===")
beta_means = result['postMeanbeta'].flatten()
print(f"年龄效应: {beta_means[1]:.3f}")
print(f"服务时间效应: {beta_means[2]:.3f}")
print(f"VIP效应: {beta_means[3]:.3f}")

if beta_means[2] < 0:
    print("服务时间越长，客户满意度越低")
if beta_means[3] > 0:
    print("VIP客户满意度更高")
```

## 常见问题和解决方案

### 1. 收敛问题
```python
# 如果模型不收敛，尝试：
# - 增加burn-in次数
# - 调整tune参数（仅OR1）
# - 检查数据是否有问题

# 检查接受率（仅OR1）
if 'acceptancerate' in result:
    print(f"接受率: {result['acceptancerate']:.1f}%")
    if result['acceptancerate'] < 20:
        print("接受率太低，尝试减小tune参数")
    elif result['acceptancerate'] > 80:
        print("接受率太高，尝试增大tune参数")
```

### 2. 数据编码问题
```python
# 确保y正确编码
def check_y_coding(y):
    unique_vals = np.unique(y)
    if not np.array_equal(unique_vals, np.arange(1, len(unique_vals) + 1)):
        print(f"警告：y的值为 {unique_vals}，应该是连续的1, 2, 3, ...")
        return False
    return True

# 自动重编码函数
def recode_y(y_original):
    """将任意有序类别重编码为1, 2, 3, ..."""
    unique_vals = np.sort(np.unique(y_original))
    y_recoded = np.zeros_like(y_original)
    for i, val in enumerate(unique_vals, 1):
        y_recoded[y_original == val] = i
    return y_recoded, unique_vals

# 示例使用
y_original = np.array([0, 1, 2, 1, 0, 2])  # 错误编码
y_correct, mapping = recode_y(y_original)
print(f"原始: {y_original}")
print(f"重编码: {y_correct}")
print(f"映射: {dict(zip(range(1, len(mapping)+1), mapping))}")
```

### 3. 选择合适的分位数
```python
# 不同分位数的含义：
quantiles_meaning = {
    0.1: "关注底部10%（最不满意的情况）",
    0.25: "关注底部25%（第一四分位数）",
    0.5: "关注中位数（典型情况）",
    0.75: "关注上部25%（第三四分位数）",
    0.9: "关注顶部10%（最满意的情况）"
}

# 可以运行多个分位数进行比较
for p in [0.25, 0.5, 0.75]:
    print(f"\n=== 分位数 p={p} ===")
    result_p = quantregOR2(y=y, x=x, b0=b0, B0=B0, 
                          burn=500, mcmc=1000, p=p, verbose=False)
    print(f"Beta估计: {result_p['postMeanbeta'].flatten()}")
```

## 总结

1. **选择正确的模型**: 3类别用OR2，4+类别用OR1
2. **数据准备**: 确保y从1开始编码，x包含截距项
3. **先验设置**: 通常使用弱信息先验（大方差）
4. **分位数选择**: 根据研究问题选择合适的分位数
5. **诊断检查**: 查看轨迹图、无效因子、接受率
6. **结果解释**: 关注系数符号、置信区间、模型比较指标

这些模型特别适用于：
- 医学研究（疾病严重程度）
- 市场研究（满意度调查）
- 教育评估（成绩等级）
- 风险评估（风险等级）

记住：贝叶斯方法给出的是参数的概率分布，而不是点估计，这让我们能更好地量化不确定性！
