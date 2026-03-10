import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(num_inliers=50, num_outliers=100):
    """生成用于多模型拟合的合成二维点云数据（两条直线 + 随机离群点）"""
    # 模型 1: y = 2x + 1
    x1 = np.random.uniform(0, 10, num_inliers)
    y1 = 2 * x1 + 1 + np.random.normal(0, 0.5, num_inliers) # 加入高斯内点噪声
    
    # 模型 2: y = -0.5x + 8
    x2 = np.random.uniform(0, 10, num_inliers)
    y2 = -0.5 * x2 + 8 + np.random.normal(0, 0.5, num_inliers)
    
    # 离群点 (Outliers)
    x_out = np.random.uniform(0, 10, num_outliers)
    y_out = np.random.uniform(0, 20, num_outliers)
    
    # 合并数据
    X = np.concatenate((x1, x2, x_out))
    Y = np.concatenate((y1, y2, y_out))
    data = np.vstack((X, Y)).T
    return data

def calculate_residuals(data, hypotheses):
    """计算数据点到假设模型的残差矩阵 R"""
    # 直线一般式：Ax + By + C = 0
    # hypotheses 是一个包含多个 [A, B, C] 列表的集合
    N = data.shape[0]
    M = len(hypotheses)
    R = np.zeros((N, M))
    
    for j, (A, B, C) in enumerate(hypotheses):
        denom = np.sqrt(A**2 + B**2)
        for i in range(N):
            x, y = data[i]
            # 点到直线的正交距离
            R[i, j] = np.abs(A*x + B*y + C) / denom 
    return R

# ==========================================
# 核心执行逻辑
# ==========================================

# 1. 生成测试数据
print("正在生成合成数据...")
data = generate_synthetic_data()

# 2. 模拟几个初始假设模型 (实际算法中，这一步是通过两点随机采样生成的)
# 假设我们采样得到了两条接近真实的直线，以及一条完全错误的直线
# L1: 2x - y + 1 = 0
# L2: -0.5x - y + 8 = 0
# L3: 0x + y - 10 = 0 (错误模型)
mock_hypotheses = [[2, -1, 1], [-0.5, -1, 8], [0, 1, -10]]

# 3. 计算 N x M 的残差矩阵 R
print("正在计算残差矩阵...")
R = calculate_residuals(data, mock_hypotheses)

# 4. 构建 HVF 的偏好矩阵 P (这一步是建立在固定阈值上的)
tau = 1.5 # 这是一个人工设定的固定阈值，后续将被 AIKOSE 替换
P = np.zeros_like(R)
# 只有残差小于阈值的点才分配偏好分数
mask = R < tau
P[mask] = np.exp(-R[mask] / tau)

print(f"数据总数: {data.shape[0]}")
print(f"偏好矩阵的维度: {P.shape} (N个点 x M个模型)")

# 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], s=15, c='gray', alpha=0.6, label='Data Points (Inliers + Outliers)')
plt.title('Synthetic Data for Robust Multi-Model Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()