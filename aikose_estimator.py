"""
AIKOSE (Adaptive Inlier-Scale Estimator) 模块
用于鲁棒多模型拟合中的自适应内点噪声尺度估计。
"""
import numpy as np


def _estimate_single_model_scale(residuals: np.ndarray, k_min: int = 3) -> float:
    """
    核心数学逻辑：分析单个模型的残差分布，寻找最佳截断点。
    
    参数:
        residuals (np.ndarray): 某个模型对应所有数据点的残差向量 (1D array)。
        k_min (int): 最小内点数量假设，防止在极少数重合点中误判。
        
    返回:
        float: 该模型专属的动态内点阈值 (tau)。
    """
    # 1. 提取有效残差并从小到大排序
    r_sorted = np.sort(residuals)
    N = len(r_sorted)
    
    # 2. 极值兜底：如果数据点少于要求的最小内点数，直接返回均值
    if N <= k_min:
        return max(np.mean(r_sorted), 1e-5)
        
    # 3. 计算残差分布的梯度突变 (寻找内点聚集区到离群点断层的拐点)
    gradients = np.diff(r_sorted)
    
    # 4. 在有效区间内搜索最大梯度突变点的位置 (即 K 值)
    best_k = np.argmax(gradients[k_min:]) + k_min
    
    # 5. 输出动态阈值，并设置下限防止除零错误
    dynamic_tau = r_sorted[best_k] 
    return max(dynamic_tau, 1e-5)


def build_adaptive_preference_matrix(R: np.ndarray) -> tuple:
    """
    对外接口：接收 HVF 的残差矩阵，返回自适应偏好矩阵和对应的阈值列表。
    
    参数:
        R (np.ndarray): N x M 的残差矩阵。
        
    返回:
        P_auto (np.ndarray): N x M 的动态偏好矩阵。
        dynamic_taus (list): M 个模型分别对应的 tau 值。
    """
    N, M = R.shape
    P_auto = np.zeros_like(R)
    dynamic_taus = []
    
    for j in range(M):
        # 提取第 j 个模型的残差列向量
        residuals_j = R[:, j]
        
        # 调用核心数学逻辑，计算 tau_j
        tau_j = _estimate_single_model_scale(residuals_j)
        dynamic_taus.append(tau_j)
        
        # 计算该模型的偏好分数
        mask = residuals_j < tau_j
        P_auto[mask, j] = np.exp(-residuals_j[mask] / tau_j)
        
    return P_auto, dynamic_taus