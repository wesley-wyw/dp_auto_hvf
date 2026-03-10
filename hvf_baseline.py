import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据生成模块
# ==========================================
def generate_synthetic_data(num_inliers=50, num_outliers=100):
    """生成用于多模型拟合的合成二维点云数据（两条直线 + 随机离群点）"""
    # 真实模型 1: y = 2x + 1  => 2x - y + 1 = 0
    x1 = np.random.uniform(0, 10, num_inliers)
    y1 = 2 * x1 + 1 + np.random.normal(0, 0.5, num_inliers)
    
    # 真实模型 2: y = -0.5x + 8 => -0.5x - y + 8 = 0
    x2 = np.random.uniform(0, 10, num_inliers)
    y2 = -0.5 * x2 + 8 + np.random.normal(0, 0.5, num_inliers)
    
    # 离群点 (完全随机的噪声)
    x_out = np.random.uniform(0, 10, num_outliers)
    y_out = np.random.uniform(0, 20, num_outliers)
    
    X = np.concatenate((x1, x2, x_out))
    Y = np.concatenate((y1, y2, y_out))
    data = np.vstack((X, Y)).T
    return data

def calculate_residuals(data, hypotheses):
    """计算数据点到假设模型的正交距离（残差）"""
    N = data.shape[0]
    M = len(hypotheses)
    R = np.zeros((N, M))
    
    for j, (A, B, C) in enumerate(hypotheses):
        denom = np.sqrt(A**2 + B**2)
        for i in range(N):
            x, y = data[i]
            R[i, j] = np.abs(A*x + B*y + C) / denom 
    return R

# ==========================================
# 2. HVF 核心算法模块
# ==========================================
print("正在生成数据并构建假设模型...")
data = generate_synthetic_data()

# 为了演示，我们直接投放3个假设模型（前两个是正确的，第三个是瞎编的）
mock_hypotheses = np.array([
    [2, -1, 1],       # 模型 1 (真实)
    [-0.5, -1, 8],    # 模型 2 (真实)
    [0, 1, -10]       # 模型 3 (错误：y = 10 的水平线)
])

# 计算残差矩阵
R = calculate_residuals(data, mock_hypotheses)

# --- 重点关注这里：固定阈值 ---
tau = 1.2
P = np.zeros_like(R)
mask = R < tau
P[mask] = np.exp(-R[mask] / tau)

print("正在计算相似度矩阵与层级投票...")
# 计算关联矩阵 S (用偏好矩阵的内积来衡量点与点的一致性)
S = np.dot(P, P.T) 
row_sums = S.sum(axis=1)
# 避免除以 0 的情况
row_sums[row_sums == 0] = 1e-10 
S_normalized = S / row_sums[:, np.newaxis]

# 层级投票：结合相似度矩阵进行加权投票
M = mock_hypotheses.shape[0]
V_models = np.zeros(M)
for j in range(M):
    p_j = P[:, j]
    voted_score = np.dot(S_normalized, p_j)
    V_models[j] = np.sum(voted_score)

# ==========================================
# 3. 结果输出模块
# ==========================================
print("\n=== HVF 层级投票最终结果 ===")
for j, (A, B, C) in enumerate(mock_hypotheses):
    print(f"模型 {j+1} (A={A}, B={B}, C={C}): 最终得票分 = {V_models[j]:.2f}")

# 画图看看原始数据
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], s=15, c='gray', alpha=0.6, label='Data Points')
plt.title(f'HVF Baseline Testing (tau={tau})')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()