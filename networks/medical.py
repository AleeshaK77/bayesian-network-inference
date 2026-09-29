from bayesnet.network import BayesianNetwork


def create_medical_network():
    """
    Create a small synthetic Bayesian network for medical diagnosis.

    The network models relationships among:
        - Age
        - Smoking
        - Heart Disease
        - Chest Pain
        - Abnormal ECG
        - Shortness of Breath

    All probabilities are illustrative and are not medical statistics.
    """

    network = BayesianNetwork()

    # Prior probability of being in the older age group.
    network.add_node(
        "Age",
        cpt={
            (): 0.30
        }
    )

    # Prior probability of smoking.
    network.add_node(
        "Smoking",
        cpt={
            (): 0.25
        }
    )

    # P(Heart Disease=True | Age, Smoking)
    network.add_node(
        "HeartDisease",
        parents=["Age", "Smoking"],
        cpt={
            (True, True): 0.35,
            (True, False): 0.15,
            (False, True): 0.12,
            (False, False): 0.03,
        }
    )

    # P(Chest Pain=True | Heart Disease)
    network.add_node(
        "ChestPain",
        parents=["HeartDisease"],
        cpt={
            (True,): 0.75,
            (False,): 0.10,
        }
    )

    # P(Abnormal ECG=True | Heart Disease)
    network.add_node(
        "AbnormalECG",
        parents=["HeartDisease"],
        cpt={
            (True,): 0.80,
            (False,): 0.05,
        }
    )

    # P(Shortness of Breath=True | Heart Disease)
    network.add_node(
        "ShortnessOfBreath",
        parents=["HeartDisease"],
        cpt={
            (True,): 0.65,
            (False,): 0.08,
        }
    )

    network.validate()

    return network


if __name__ == "__main__":
    network = create_medical_network()

    print("Nodes:", list(network.nodes))
    print("Topological order:", network.topological_order())

    print("\nParents of HeartDisease:")
    print(network.parents("HeartDisease"))

    print("\nChildren of HeartDisease:")
    print(network.children("HeartDisease"))

    print("\nP(HeartDisease=True | Age=True, Smoking=True):")
    print(
        network.probability(
            "HeartDisease",
            True,
            (True, True)
        )
    )

    print("\nNetwork validation: passed")