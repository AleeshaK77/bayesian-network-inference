from itertools import product


class Factor:
    """
    Represents a factor over Boolean variables.

    Variables are stored in a fixed order. Each assignment is represented
    by a tuple of Boolean values corresponding to that order.
    """

    def __init__(self, variables, values):
        self.variables = list(variables)
        self.values = dict(values)

        expected_assignments = 2 ** len(self.variables)

        if len(self.values) != expected_assignments:
            raise ValueError(
                f"Factor over {len(self.variables)} variables must contain "
                f"{expected_assignments} assignments."
            )

        for assignment, value in self.values.items():
            if len(assignment) != len(self.variables):
                raise ValueError(
                    "Assignment length does not match number of variables."
                )

            if not all(isinstance(v, bool) for v in assignment):
                raise ValueError("Factor assignments must be Boolean.")

            if value < 0:
                raise ValueError("Factor values cannot be negative.")

    def get(self, assignment):
        """Return the value associated with an assignment."""
        return self.values[tuple(assignment)]

    def restrict(self, variable, value):
        """
        Restrict a variable to a particular value.

        Returns a new factor with the variable removed.
        """
        if variable not in self.variables:
            return self

        index = self.variables.index(variable)

        new_variables = [
            var for var in self.variables
            if var != variable
        ]

        new_values = {}

        for assignment, probability in self.values.items():
            if assignment[index] != value:
                continue

            new_assignment = tuple(
                value
                for i, value in enumerate(assignment)
                if i != index
            )

            new_values[new_assignment] = probability

        return Factor(new_variables, new_values)

    def multiply(self, other):
        """
        Multiply this factor with another factor.

        The result contains the union of variables from both factors.
        """
        variables = list(self.variables)

        for variable in other.variables:
            if variable not in variables:
                variables.append(variable)

        new_values = {}

        for assignment in product([False, True], repeat=len(variables)):
            assignment_map = dict(zip(variables, assignment))

            self_assignment = tuple(
                assignment_map[var]
                for var in self.variables
            )

            other_assignment = tuple(
                assignment_map[var]
                for var in other.variables
            )

            value = (
                self.values[self_assignment]
                * other.values[other_assignment]
            )

            new_values[assignment] = value

        return Factor(variables, new_values)

    def sum_out(self, variable):
        """
        Sum out a variable.

        For a factor f(X, Y), summing out X gives:

            f'(Y) = sum_X f(X, Y)
        """
        if variable not in self.variables:
            return self

        index = self.variables.index(variable)

        new_variables = [
            var for var in self.variables
            if var != variable
        ]

        new_values = {}

        for assignment in product([False, True], repeat=len(new_variables)):
            total = 0.0

            for variable_value in [False, True]:
                full_assignment = []

                assignment_index = 0

                for i in range(len(self.variables)):
                    if i == index:
                        full_assignment.append(variable_value)
                    else:
                        full_assignment.append(
                            assignment[assignment_index]
                        )
                        assignment_index += 1

                total += self.values[tuple(full_assignment)]

            new_values[assignment] = total

        return Factor(new_variables, new_values)

    def normalize(self):
        """
        Normalize the factor so that all values sum to 1.

        Returns a new factor.
        """
        total = sum(self.values.values())

        if total == 0:
            raise ValueError("Cannot normalize a factor whose sum is zero.")

        normalized_values = {
            assignment: value / total
            for assignment, value in self.values.items()
        }

        return Factor(self.variables, normalized_values)

    def __repr__(self):
        return f"Factor(variables={self.variables}, values={self.values})"