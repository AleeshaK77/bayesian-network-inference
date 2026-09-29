import random

from .network import BayesianNetwork


def prior_sample(network):
    """
    Generate one complete assignment by sampling variables
    in topological order.
    """
    sample = {}

    for variable in network.topological_order():
        parents = network.parents(variable)
        parent_values = tuple(sample[parent] for parent in parents)

        probability_true = network.probability(
            variable,
            True,
            parent_values
        )

        sample[variable] = random.random() < probability_true

    return sample


def rejection_sampling(network, query, evidence, num_samples):
    """
    Estimate P(query | evidence) using rejection sampling.

    Samples are generated from the prior distribution. Any sample
    inconsistent with the evidence is rejected.

    Returns
    -------
    dict[bool, float]
        Estimated posterior distribution for the query variable.
    """
    _validate_inputs(network, query, evidence, num_samples)

    counts = {
        False: 0,
        True: 0
    }

    accepted = 0

    for _ in range(num_samples):
        sample = prior_sample(network)

        if all(
            sample[variable] == value
            for variable, value in evidence.items()
        ):
            counts[sample[query]] += 1
            accepted += 1

    if accepted == 0:
        return {
                False: float("nan"),
                True: float("nan")
        }

    return {
        False: counts[False] / accepted,
        True: counts[True] / accepted
    }


def _validate_inputs(network, query, evidence, num_samples):
    """Validate sampling inputs."""
    if query not in network:
        raise ValueError(f"Unknown query variable: '{query}'.")

    for variable, value in evidence.items():
        if variable not in network:
            raise ValueError(f"Unknown evidence variable: '{variable}'.")

        if not isinstance(value, bool):
            raise ValueError(
                f"Evidence for '{variable}' must be Boolean."
            )

    if query in evidence:
        raise ValueError("Query variable cannot also be evidence.")

    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")