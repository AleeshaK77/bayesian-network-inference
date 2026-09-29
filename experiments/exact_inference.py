from bayesnet.inference import enumeration_ask, variable_elimination
from networks.medical import create_medical_network


def run_query(network, query, evidence):
    """Compare enumeration and variable elimination for one query."""

    enumeration_result = enumeration_ask(
        network,
        query,
        evidence
    )

    elimination_result = variable_elimination(
        network,
        query,
        evidence
    )

    print(f"Query: P({query} | {evidence})")
    print(f"  Enumeration:       {enumeration_result}")
    print(f"  Variable Elimination: {elimination_result}")

    difference = abs(
        enumeration_result[True]
        - elimination_result[True]
    )

    print(f"  Difference:        {difference:.12f}")
    print()


def main():
    network = create_medical_network()

    queries = [
        (
            "HeartDisease",
            {
                "ChestPain": True,
                "AbnormalECG": True
            }
        ),
        (
            "HeartDisease",
            {
                "ChestPain": True,
                "ShortnessOfBreath": True
            }
        ),
        (
            "HeartDisease",
            {
                "Age": True,
                "Smoking": True,
                "ChestPain": True
            }
        ),
        (
            "AbnormalECG",
            {
                "ChestPain": True
            }
        ),
    ]

    print("Exact Inference: Enumeration vs. Variable Elimination")
    print("=" * 55)
    print()

    for query, evidence in queries:
        run_query(network, query, evidence)


if __name__ == "__main__":
    main()