from itertools import product

from .factor import Factor
from .network import BayesianNetwork


def enumeration_ask(network, query, evidence=None):
    """
    Compute P(query | evidence) using enumeration.

    Parameters
    ----------
    network : BayesianNetwork
        Bayesian network to perform inference on.

    query : str
        Variable whose distribution is requested.

    evidence : dict[str, bool], optional
        Observed variable assignments.

    Returns
    -------
    dict[bool, float]
        Posterior probability distribution for the query variable.
    """
    evidence = {} if evidence is None else dict(evidence)

    if query not in network:
        raise ValueError(f"Unknown query variable: '{query}'.")

    for variable in evidence:
        if variable not in network:
            raise ValueError(f"Unknown evidence variable: '{variable}'.")

    if query in evidence:
        raise ValueError("Query variable cannot also be evidence.")

    distribution = {}

    for query_value in [False, True]:
        extended_evidence = dict(evidence)
        extended_evidence[query] = query_value

        distribution[query_value] = _enumerate_all(
            network,
            network.topological_order(),
            extended_evidence
        )

    return _normalize_distribution(distribution)


def _enumerate_all(network, variables, evidence):
    """
    Recursively compute the probability of an evidence assignment.

    This is the recursive ENUMERATE-ALL algorithm from AIMA.
    """
    if not variables:
        return 1.0

    variable = variables[0]
    rest = variables[1:]

    parents = network.parents(variable)

    if variable in evidence:
        parent_values = tuple(evidence[parent] for parent in parents)

        probability = network.probability(
            variable,
            evidence[variable],
            parent_values
        )

        return probability * _enumerate_all(
            network,
            rest,
            evidence
        )

    total = 0.0

    for value in [False, True]:
        extended_evidence = dict(evidence)
        extended_evidence[variable] = value

        parent_values = tuple(
            extended_evidence[parent]
            for parent in parents
        )

        probability = network.probability(
            variable,
            value,
            parent_values
        )

        total += probability * _enumerate_all(
            network,
            rest,
            extended_evidence
        )

    return total


def variable_elimination(network, query, evidence=None, order=None, return_stats = False):
    """
    Compute P(query | evidence) using variable elimination.

    Parameters
    ----------
    network : BayesianNetwork
        Bayesian network to perform inference on.

    query : str
        Variable whose distribution is requested.

    evidence : dict[str, bool], optional
        Observed variable assignments.

    order : list[str], optional
        Order in which hidden variables are eliminated.
        If omitted, the network's topological order is used.

    Returns
    -------
    dict[bool, float]
        Posterior probability distribution for the query variable.
    """
    evidence = {} if evidence is None else dict(evidence)

    _validate_query_and_evidence(network, query, evidence)

    factors = _create_factors(network, evidence)

    max_factor_size = max(
        len(factor.variables)
        for factor in factors
    )

    hidden_variables = [
        variable
        for variable in network
        if variable != query and variable not in evidence
    ]

    if order is None:
        order = [
            variable
            for variable in network.topological_order()
            if variable in hidden_variables
        ]
    else:
        order = list(order)

        if set(order) != set(hidden_variables):
            raise ValueError(
                "Elimination order must contain exactly the hidden variables."
            )

    for variable in order:
        relevant_factors = [
            factor
            for factor in factors
            if variable in factor.variables
        ]

        if not relevant_factors:
            continue

        factors = [
            factor
            for factor in factors
            if variable not in factor.variables
        ]

        product_factor = relevant_factors[0]

        for factor in relevant_factors[1:]:
            product_factor = product_factor.multiply(factor)

        new_factor = product_factor.sum_out(variable)

        factors.append(new_factor)

        max_factor_size = max(
            max_factor_size,
            len(product_factor.variables)
        )

    result = factors[0]

    for factor in factors[1:]:
        result = result.multiply(factor)

    distribution = _factor_to_distribution(result, query)

    if return_stats:
        return distribution, {
            "max_factor_size": max_factor_size
        }

    return distribution


def _create_factors(network, evidence):
    """
    Create one factor for each network variable.

    Evidence is incorporated by restricting the corresponding
    variable in each factor.
    """
    factors = []

    for variable in network.topological_order():
        parents = network.parents(variable)
        variables = parents + [variable]

        values = {}

        for assignment in product([False, True], repeat=len(variables)):
            parent_values = assignment[:-1]
            variable_value = assignment[-1]

            probability = network.probability(
                variable,
                variable_value,
                parent_values
            )

            values[assignment] = probability

        factor = Factor(variables, values)

        for observed_variable, observed_value in evidence.items():
            if observed_variable in factor.variables:
                factor = factor.restrict(
                    observed_variable,
                    observed_value
                )

        factors.append(factor)

    return factors


def min_degree_order(network, query, evidence=None):
    """
    Return a variable elimination order using the min-degree heuristic.

    At each step, eliminate the variable with the fewest neighbors
    in the current interaction graph.
    """
    evidence = {} if evidence is None else dict(evidence)

    _validate_query_and_evidence(network, query, evidence)

    graph = _interaction_graph(network, evidence)

    hidden = {
        variable
        for variable in network
        if variable != query and variable not in evidence
    }

    order = []

    while hidden:
        variable = min(
            hidden,
            key=lambda node: len(graph[node])
        )

        order.append(variable)

        neighbors = list(graph[variable])

        for neighbor in neighbors:
            graph[neighbor].discard(variable)

        del graph[variable]
        hidden.remove(variable)

    return order


def min_fill_order(network, query, evidence=None):
    """
    Return a variable elimination order using the min-fill heuristic.

    At each step, eliminate the variable that requires the fewest
    additional edges to connect its current neighbors.
    """
    evidence = {} if evidence is None else dict(evidence)

    _validate_query_and_evidence(network, query, evidence)

    graph = _interaction_graph(network, evidence)

    hidden = {
        variable
        for variable in network
        if variable != query and variable not in evidence
    }

    order = []

    while hidden:
        variable = min(
            hidden,
            key=lambda node: _fill_count(graph, node)
        )

        order.append(variable)

        neighbors = list(graph[variable])

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                first = neighbors[i]
                second = neighbors[j]

                graph[first].add(second)
                graph[second].add(first)

        for neighbor in neighbors:
            graph[neighbor].discard(variable)

        del graph[variable]
        hidden.remove(variable)

    return order


def _interaction_graph(network, evidence):
    """
    Construct the variable interaction graph used by elimination
    ordering heuristics.

    Variables appearing together in a factor are connected.
    """
    graph = {
        variable: set()
        for variable in network
        if variable not in evidence
    }

    for variable in network:
        if variable in evidence:
            continue

        variables = [
            variable
        ] + [
            parent
            for parent in network.parents(variable)
            if parent not in evidence
        ]

        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                first = variables[i]
                second = variables[j]

                if first in graph and second in graph:
                    graph[first].add(second)
                    graph[second].add(first)

    return graph


def _fill_count(graph, variable):
    """Return the number of missing edges among a variable's neighbors."""
    neighbors = list(graph[variable])
    missing = 0

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] not in graph[neighbors[i]]:
                missing += 1

    return missing


def _factor_to_distribution(factor, query):
    """Convert a one-variable factor into a probability distribution."""
    if query not in factor.variables:
        raise ValueError(
            f"Query variable '{query}' was eliminated unexpectedly."
        )

    if len(factor.variables) != 1:
        raise ValueError(
            "Final factor contains variables other than the query."
        )

    query_index = factor.variables.index(query)

    distribution = {}

    for assignment, value in factor.values.items():
        distribution[assignment[query_index]] = value

    return _normalize_distribution(distribution)


def _normalize_distribution(distribution):
    """Normalize a dictionary of unnormalized probabilities."""
    total = sum(distribution.values())

    if total == 0:
        raise ValueError("Cannot normalize a zero-probability distribution.")

    return {
        value: probability / total
        for value, probability in distribution.items()
    }


def _validate_query_and_evidence(network, query, evidence):
    """Validate query and evidence variables."""
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