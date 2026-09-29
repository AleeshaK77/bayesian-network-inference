import time

from bayesnet.inference import (
    min_degree_order,
    min_fill_order,
    variable_elimination,
)
from networks.synthetic import (
    create_branching_network,
    create_chain_network,
    create_irregular_network,
)


def benchmark_order(network, query, evidence, order):
    """
    Run variable elimination with a fixed ordering.

    Returns runtime and maximum intermediate factor size.
    """

    start = time.perf_counter()

    _, stats = variable_elimination(
        network,
        query,
        evidence,
        order=order,
        return_stats=True
    )

    runtime = time.perf_counter() - start

    return runtime, stats["max_factor_size"]


def get_orders(network, query, evidence):
    """Generate the three elimination orders being compared."""

    hidden_variables = [
        variable
        for variable in network
        if variable != query and variable not in evidence
    ]

    topological_order = [
        variable
        for variable in network.topological_order()
        if variable in hidden_variables
    ]

    degree_order = min_degree_order(
        network,
        query,
        evidence
    )

    fill_order = min_fill_order(
        network,
        query,
        evidence
    )

    return [
        ("Topological", topological_order),
        ("Min-degree", degree_order),
        ("Min-fill", fill_order),
    ]


def run_experiment(network, query, evidence):
    """Benchmark all elimination orders on one network."""

    results = []

    for name, order in get_orders(network, query, evidence):
        runtime, max_factor_size = benchmark_order(
            network,
            query,
            evidence,
            order
        )

        results.append(
            (name, max_factor_size, runtime)
        )

    return results


def main():
    print("Elimination Order Benchmark")
    print("=" * 70)

    networks = [
        ("Chain", create_chain_network(30)),
        ("Branching", create_branching_network(30)),
        ("Irregular", create_irregular_network(30)),
    ]

    for network_name, network in networks:
        query = "X15"

        evidence = {
            "X0": True,
            "X29": False
        }

        print(f"\n{network_name} Network")
        print("-" * 70)

        print(
            f"{'Ordering':>15} "
            f"{'Max Factor':>15} "
            f"{'Runtime (s)':>15}"
        )

        print("-" * 70)

        results = run_experiment(
            network,
            query,
            evidence
        )

        for name, max_factor_size, runtime in results:
            print(
                f"{name:>15} "
                f"{max_factor_size:>15} "
                f"{runtime:>15.6f}"
            )


if __name__ == "__main__":
    main()