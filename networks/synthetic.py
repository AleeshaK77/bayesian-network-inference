from bayesnet.network import BayesianNetwork


def create_chain_network(num_nodes):
    """
    Create a chain-structured Bayesian network.

    X0 -> X1 -> X2 -> ... -> X(n-1)
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")

    network = BayesianNetwork()

    network.add_node(
        "X0",
        cpt={
            (): 0.5
        }
    )

    for i in range(1, num_nodes):
        network.add_node(
            f"X{i}",
            parents=[f"X{i - 1}"],
            cpt={
                (True,): 0.8,
                (False,): 0.2,
            }
        )

    network.validate()

    return network


def create_branching_network(num_nodes):
    """
    Create a branching Bayesian network.

    Each new variable depends on the two preceding variables.
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")

    network = BayesianNetwork()

    network.add_node(
        "X0",
        cpt={
            (): 0.5
        }
    )

    network.add_node(
        "X1",
        cpt={
            (): 0.5
        }
    )

    for i in range(2, num_nodes):
        network.add_node(
            f"X{i}",
            parents=[
                f"X{i - 2}",
                f"X{i - 1}"
            ],
            cpt={
                (True, True): 0.90,
                (True, False): 0.70,
                (False, True): 0.60,
                (False, False): 0.10,
            }
        )

    network.validate()

    return network


def create_irregular_network(num_nodes):
    """
    Create an irregular Bayesian network with varying parent structure.

    The structure contains:
        - independent root variables
        - nodes with one parent
        - nodes with two parents
        - nodes with three parents

    This produces a less regular interaction graph, making it
    useful for studying elimination-order heuristics.
    """
    if num_nodes < 8:
        raise ValueError("num_nodes must be at least 8.")

    network = BayesianNetwork()

    # Root variables.
    network.add_node(
        "X0",
        cpt={
            (): 0.5
        }
    )

    network.add_node(
        "X1",
        cpt={
            (): 0.4
        }
    )

    network.add_node(
        "X2",
        cpt={
            (): 0.3
        }
    )

    # One-parent variables.
    network.add_node(
        "X3",
        parents=["X0"],
        cpt={
            (True,): 0.8,
            (False,): 0.2,
        }
    )

    network.add_node(
        "X4",
        parents=["X1"],
        cpt={
            (True,): 0.7,
            (False,): 0.3,
        }
    )

    # Two-parent variables.
    network.add_node(
        "X5",
        parents=["X0", "X1"],
        cpt={
            (True, True): 0.9,
            (True, False): 0.6,
            (False, True): 0.5,
            (False, False): 0.1,
        }
    )

    network.add_node(
        "X6",
        parents=["X2", "X3"],
        cpt={
            (True, True): 0.85,
            (True, False): 0.65,
            (False, True): 0.55,
            (False, False): 0.15,
        }
    )

    # Three-parent variable.
    network.add_node(
        "X7",
        parents=["X3", "X4", "X5"],
        cpt={
            (True, True, True): 0.95,
            (True, True, False): 0.80,
            (True, False, True): 0.75,
            (True, False, False): 0.50,
            (False, True, True): 0.70,
            (False, True, False): 0.40,
            (False, False, True): 0.35,
            (False, False, False): 0.05,
        }
    )

    # Extend the network while keeping an irregular structure.
    for i in range(8, num_nodes):
        if i % 4 == 0:
            parents = [f"X{i - 1}"]

        elif i % 4 == 1:
            parents = [f"X{i - 2}", f"X{i - 1}"]

        elif i % 4 == 2:
            parents = [f"X{i - 3}", f"X{i - 1}"]

        else:
            parents = [
                f"X{i - 4}",
                f"X{i - 2}",
                f"X{i - 1}"
            ]

        num_parent_values = 2 ** len(parents)

        cpt = {}

        for assignment_index in range(num_parent_values):
            assignment = tuple(
                bool(assignment_index & (1 << j))
                for j in range(len(parents))
            )

            cpt[assignment] = 0.2 + 0.1 * (
                assignment_index % 6
            )

        network.add_node(
            f"X{i}",
            parents=parents,
            cpt=cpt
        )

    network.validate()

    return network


if __name__ == "__main__":
    print("Chain network:")
    chain = create_chain_network(5)

    print("Nodes:", list(chain.nodes))
    print("Topological order:", chain.topological_order())

    print("\nBranching network:")
    branching = create_branching_network(6)

    print("Nodes:", list(branching.nodes))
    print("Topological order:", branching.topological_order())

    print("\nIrregular network:")
    irregular = create_irregular_network(12)

    print("Nodes:", list(irregular.nodes))
    print("Topological order:", irregular.topological_order())

    print("\nNetwork validation: passed")