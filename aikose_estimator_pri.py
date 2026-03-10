import numpy as np

def _estimate_single_model_scale_dp(residuals: np.ndarray, epsilon: float = None, k_min: int = 3) -> float:
    """
    终极修正版：带有自适应量级对齐的隐私惩罚 AIKOSE
    """
    r_sorted = np.sort(residuals)
    N = len(r_sorted)
    
    if N <= k_min:
        return max(np.mean(r_sorted), 1e-5)
        
    gradients = np.diff(r_sorted)
    
    if epsilon is not None and epsilon > 0:
        # 1. 找到当前梯度的最大值，作为量级参考
        max_grad = np.max(gradients[k_min:]) if len(gradients[k_min:]) > 0 else 1.0
        
        # 2. 调小基础系数，并乘以 max_grad 进行量级对齐！
        lambda_param = 0.05 
        penalty_factor = lambda_param * (1.0 / epsilon) * max_grad
        
        k_values = np.arange(k_min, N - 1) 
        # 使用线性惩罚或平缓的 log 惩罚
        penalties = penalty_factor * np.log(k_values - k_min + 1.1) 
        
        adjusted_objective = gradients[k_min:] - penalties
        best_k = np.argmax(adjusted_objective) + k_min
    else:
        best_k = np.argmax(gradients[k_min:]) + k_min
    
    dynamic_tau = r_sorted[best_k] 
    return max(dynamic_tau, 1e-5)
    
def build_adaptive_preference_matrix(R: np.ndarray, epsilon: float = None) -> tuple:
    """接口也需要更新，把 epsilon 传进去"""
    N, M = R.shape
    P_auto = np.zeros_like(R)
    dynamic_taus = []
    
    for j in range(M):
        residuals_j = R[:, j]
        # 调用新版的 DP-AIKOSE
        tau_j = _estimate_single_model_scale_dp(residuals_j, epsilon)
        dynamic_taus.append(tau_j)
        
        mask = residuals_j < tau_j
        P_auto[mask, j] = np.exp(-residuals_j[mask] / tau_j)
        
    return P_auto, dynamic_taus