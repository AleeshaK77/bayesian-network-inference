import statistics

from bayesnet.inference import enumeration_ask
from bayesnet.sampling import rejection_sampling
from networks.medical import create_medical_network


def run_trial(network, query, evidence, num_samples, exact_probability):
    """Run one rejection-sampling trial and return its absolute error."""

    result = rejection_sampling(
        network,
        query,
        evidence,
        num_samples
    )

    estimate = result[True]

    if estimate != estimate:  # NaN check
        return None

    return abs(estimate - exact_probability)


def benchmark_sample_size(
    network,
    query,
    evidence,
    exact_probability,
    num_samples,
    num_trials=20
):
    """Run multiple trials and return the mean absolute error."""

    errors = []

    for _ in range(num_trials):
        error = run_trial(
            network,
            query,
            evidence,
            num_samples,
            exact_probability
        )

        if error is not None:
            errors.append(error)

    if not errors:
        return float("nan")

    return statistics.mean(errors)


def main():
    network = create_medical_network()

    query = "HeartDisease"

    evidence = {
        "ChestPain": True,
        "AbnormalECG": True
    }

    exact_result = enumeration_ask(
        network,
        query,
        evidence
    )

    exact_probability = exact_result[True]

    sample_sizes = [
        1_000,
        10_000,
        100_000,
        1_000_000,
    ]

    print("Sampling Convergence Benchmark")
    print("=" * 65)
    print()
    print(
        f"P({query}=True | {evidence}) exact = "
        f"{exact_probability:.6f}"
    )
    print()

    print(
        f"{'Samples':>12} "
        f"{'Mean Absolute Error':>22}"
    )
    print("-" * 65)

    for num_samples in sample_sizes:
        mean_error = benchmark_sample_size(
            network,
            query,
            evidence,
            exact_probability,
            num_samples
        )

        print(
            f"{num_samples:>12,} "
            f"{mean_error:>22.6f}"
        )


if __name__ == "__main__":
    main()