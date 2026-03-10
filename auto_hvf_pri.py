import numpy as np
import matplotlib.pyplot as plt

from aikose_estimator_pri import build_adaptive_preference_matrix


DEFAULT_NUM_INLIERS = 30
DEFAULT_NUM_OUTLIERS = 50
DEFAULT_EPSILON = 0.1
MIN_EPSILON = 1e-5

MOCK_HYPOTHESES = np.array([
    [2, -1, 1],
    [-0.5, -1, 8],
    [0, 1, -10],
], dtype=float)


def get_experiment_config():
    """Return the default configuration for the Auto-HVF experiment."""
    return {
        "num_inliers": DEFAULT_NUM_INLIERS,
        "num_outliers": DEFAULT_NUM_OUTLIERS,
        "epsilon": DEFAULT_EPSILON,
        "hypotheses": MOCK_HYPOTHESES.copy(),
    }


def generate_synthetic_data(num_inliers=DEFAULT_NUM_INLIERS, num_outliers=DEFAULT_NUM_OUTLIERS):
    """Generate 2D line-fitting data with two inlier groups and random outliers."""
    x1 = np.random.uniform(0, 10, num_inliers)
    y1 = 2 * x1 + 1 + np.random.normal(0, 0.5, num_inliers)

    x2 = np.random.uniform(0, 10, num_inliers)
    y2 = -0.5 * x2 + 8 + np.random.normal(0, 0.5, num_inliers)

    x_out = np.random.uniform(0, 10, num_outliers)
    y_out = np.random.uniform(0, 20, num_outliers)

    x_values = np.concatenate((x1, x2, x_out))
    y_values = np.concatenate((y1, y2, y_out))
    return np.vstack((x_values, y_values)).T


def prepare_experiment_inputs(config):
    """Create the synthetic dataset and hypothesis set for one experiment run."""
    data = generate_synthetic_data(
        num_inliers=config["num_inliers"],
        num_outliers=config["num_outliers"],
    )
    hypotheses = np.asarray(config["hypotheses"], dtype=float).copy()
    return data, hypotheses


def calculate_residuals_fast(data, hypotheses):
    """Compute point-to-line residuals for all data points and hypotheses."""
    num_points = data.shape[0]
    data_homogeneous = np.hstack((data, np.ones((num_points, 1))))

    hypotheses_array = np.asarray(hypotheses, dtype=float)
    denominators = np.sqrt(hypotheses_array[:, 0] ** 2 + hypotheses_array[:, 1] ** 2)
    denominators = np.where(denominators == 0, 1e-10, denominators)

    numerators = np.abs(data_homogeneous @ hypotheses_array.T)
    return numerators / denominators



def compute_similarity_matrix(preference_matrix):
    """Build the point similarity matrix from the preference matrix."""
    return preference_matrix @ preference_matrix.T



def compute_hvf_votes(preference_matrix):
    """Compute HVF model scores using the similarity-weighted votes."""
    similarity_matrix = compute_similarity_matrix(preference_matrix)
    num_models = preference_matrix.shape[1]
    model_votes = np.zeros(num_models)

    for model_index in range(num_models):
        preference_column = preference_matrix[:, model_index]
        voted_score = similarity_matrix @ preference_column
        model_votes[model_index] = np.sum(voted_score)

    return model_votes, similarity_matrix



def run_hvf_analysis(data, hypotheses, epsilon=DEFAULT_EPSILON):
    """Run the complete HVF analysis block."""
    residuals = calculate_residuals_fast(data, hypotheses)
    preference_matrix, dynamic_thresholds = build_adaptive_preference_matrix(residuals, epsilon=epsilon)
    model_votes, similarity_matrix = compute_hvf_votes(preference_matrix)

    return {
        "residuals": residuals,
        "preference_matrix": preference_matrix,
        "dynamic_thresholds": dynamic_thresholds,
        "similarity_matrix": similarity_matrix,
        "model_votes": model_votes,
    }



def add_laplace_noise(votes, epsilon, sensitivity=1.0):
    """Add Laplace noise to the vote vector for differential privacy."""
    safe_epsilon = max(epsilon, MIN_EPSILON)
    scale = sensitivity / safe_epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(votes))
    return votes + noise, noise



def run_dp_analysis(model_votes, epsilon=DEFAULT_EPSILON, sensitivity=1.0):
    """Run the differential privacy block on top of the HVF votes."""
    noisy_votes, generated_noise = add_laplace_noise(
        model_votes,
        epsilon=epsilon,
        sensitivity=sensitivity,
    )
    return {
        "clean_votes": model_votes,
        "noisy_votes": noisy_votes,
        "generated_noise": generated_noise,
        "epsilon": epsilon,
        "sensitivity": sensitivity,
    }



def print_hvf_results(hypotheses, hvf_results):
    """Print the adaptive thresholds and final HVF vote scores."""
    print(f"Dynamic thresholds: {np.round(hvf_results['dynamic_thresholds'], 3)}")
    print("\n=== Final HVF voting results ===")

    for model_index, (a_value, b_value, c_value) in enumerate(hypotheses, start=1):
        print(
            f"Model {model_index} (A={a_value}, B={b_value}, C={c_value}): "
            f"score = {hvf_results['model_votes'][model_index - 1]:.2f}"
        )



def print_dp_results(hypotheses, dp_results):
    """Print the vote scores before and after DP noise injection."""
    print("\n=== Differential privacy vote results ===")

    for model_index, _ in enumerate(hypotheses, start=1):
        print(
            f"Model {model_index}: clean = {dp_results['clean_votes'][model_index - 1]:.2f} | "
            f"noise = {dp_results['generated_noise'][model_index - 1]:.2f} | "
            f"noisy = {dp_results['noisy_votes'][model_index - 1]:.2f}"
        )



def plot_data(data):
    """Visualize the generated 2D data."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=15, c="gray", alpha=0.6, label="Data Points")
    plt.title("Auto-HVF Testing (Dynamic Thresholds)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()



def report_experiment_results(data, hypotheses, hvf_results, dp_results):
    """Handle all reporting for one experiment run."""
    print_hvf_results(hypotheses, hvf_results)
    print_dp_results(hypotheses, dp_results)
    plot_data(data)



def run_full_experiment():
    """Run the full Auto-HVF experiment from setup to reporting."""
    print("Generating synthetic data and mock hypotheses...")
    config = get_experiment_config()
    data, hypotheses = prepare_experiment_inputs(config)

    print("Running adaptive preference estimation and HVF voting...")
    hvf_results = run_hvf_analysis(data, hypotheses, epsilon=config["epsilon"])

    print("Injecting differential privacy noise into the vote scores...")
    dp_results = run_dp_analysis(hvf_results["model_votes"], epsilon=config["epsilon"])

    report_experiment_results(data, hypotheses, hvf_results, dp_results)



def main():
    """Program entry point."""
    run_full_experiment()


if __name__ == "__main__":
    main()
