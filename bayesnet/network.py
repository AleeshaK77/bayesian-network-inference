from collections import deque


class BayesianNetwork:
    """
    Represents a discrete Bayesian network.

    Each variable is assumed to be Boolean. A node stores:
        - its parents
        - its conditional probability table (CPT)

    The CPT maps tuples of parent values to P(variable = True | parents).
    For a root node, the key is the empty tuple.
    """

    def __init__(self):
        self.nodes = {}
        self.parents_map = {}
        self.children_map = {}
        self.cpts = {}

    def add_node(self, name, parents=None, cpt=None):
        """
        Add a variable to the Bayesian network.

        Parameters
        ----------
        name : str
            Name of the variable.

        parents : list[str], optional
            Names of the variable's parents.

        cpt : dict[tuple, float], optional
            Conditional probability table.
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists.")

        parents = [] if parents is None else list(parents)

        self.nodes[name] = name
        self.parents_map[name] = parents
        self.children_map[name] = []
        self.cpts[name] = {} if cpt is None else dict(cpt)

        for parent in parents:
            if parent not in self.nodes:
                raise ValueError(
                    f"Parent '{parent}' must be added before '{name}'."
                )

            self.children_map[parent].append(name)

    def parents(self, name):
        """Return the parents of a variable."""
        self._check_node(name)
        return list(self.parents_map[name])

    def children(self, name):
        """Return the children of a variable."""
        self._check_node(name)
        return list(self.children_map[name])

    def cpt(self, name):
        """Return the conditional probability table of a variable."""
        self._check_node(name)
        return dict(self.cpts[name])

    def probability(self, name, value, parent_values=()):
        """
        Return P(name = value | parent_values).

        The CPT stores probabilities for the variable being True.
        Probabilities for False are obtained by complementing them.
        """
        self._check_node(name)

        if not isinstance(value, bool):
            raise ValueError("Variable values must be Boolean.")

        probability_true = self.cpts[name].get(tuple(parent_values))

        if probability_true is None:
            raise ValueError(
                f"No CPT entry for '{name}' with parents {parent_values}."
            )

        if not 0.0 <= probability_true <= 1.0:
            raise ValueError(
                f"Invalid probability for '{name}': {probability_true}"
            )

        return probability_true if value else 1.0 - probability_true

    def topological_order(self):
        """
        Return the variables in topological order.

        Every parent appears before its children.
        """
        indegree = {
            node: len(self.parents_map[node])
            for node in self.nodes
        }

        queue = deque(
            node for node in self.nodes
            if indegree[node] == 0
        )

        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in self.children_map[node]:
                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Bayesian network contains a cycle.")

        return order

    def validate(self):
        """
        Validate the Bayesian network.
        """
        for node in self.nodes:
            for parent in self.parents_map[node]:
                if parent not in self.nodes:
                    raise ValueError(
                        f"Node '{node}' has unknown parent '{parent}'."
                    )

            for probability in self.cpts[node].values():
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Invalid probability in CPT for '{node}'."
                    )

        self.topological_order()

    def _check_node(self, name):
        """Raise an error if a variable does not exist."""
        if name not in self.nodes:
            raise ValueError(f"Unknown node: '{name}'.")

    def __contains__(self, name):
        return name in self.nodes

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)