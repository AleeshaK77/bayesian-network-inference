# Bayesian Network Inference: Exact and Approximate Methods

This project implements and benchmarks exact and approximate inference algorithms for probabilistic reasoning over Bayesian networks, based on the material in Chapter 14 of *Artificial Intelligence: A Modern Approach* (Russell & Norvig, 2010). The project covers inference by enumeration, variable elimination with elimination-order heuristics, rejection sampling, and an empirical study of sampling convergence. The central tension motivating the project is that exact inference in multiply connected networks is NP-hard in general, which makes approximate methods essential for large-scale problems; understanding where each approach is appropriate requires examining both their theoretical properties and their empirical behaviour.

---

## Project Architecture

| Module | Responsibility |
| :--- | :--- |
| `bayesnet/network.py` | Bayesian network representation and CPT storage |
| `bayesnet/factor.py` | Factor representation and pointwise product and sum-out operations |
| `bayesnet/inference.py` | Enumeration and variable elimination algorithms |
| `bayesnet/sampling.py` | Rejection sampling |
| `networks/medical.py` | Synthetic medical diagnosis network |
| `networks/synthetic.py` | Parameterised synthetic networks for benchmarking |
| `experiments/` | Experiment runners for exact inference, ordering, and convergence |

---

## Theoretical Background

### Bayesian Networks

A Bayesian network is a directed acyclic graph in which each node corresponds to a random variable and each directed edge represents a direct probabilistic dependency. Each node $X_i$ is annotated with a **conditional probability table** (CPT) specifying $\mathbf{P}(X_i \mid \text{Parents}(X_i))$. The network encodes the full joint distribution as a product of these local conditional distributions:

$$P(x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid \text{parents}(X_i)).$$

This factorisation follows from the chain rule of probability together with the conditional independence assertion that each variable is independent of its non-descendants given its parents (Russell & Norvig, 2010). The result is a representation that is often exponentially more compact than the explicit joint distribution: if each variable has at most $k$ parents, the network requires $n \cdot 2^k$ parameters for $n$ Boolean variables, compared with $2^n$ for the full joint.

The graph topology encodes a second important independence property: each variable is conditionally independent of all other variables in the network given its **Markov blanket** — its parents, children, and children's parents. This property is exploited directly by Gibbs sampling and underlies the efficiency of local inference algorithms.

---

### Exact Inference

The basic inference task is to compute the posterior distribution $\mathbf{P}(X \mid \mathbf{e})$ for a query variable $X$ given observed evidence $\mathbf{e}$. Denoting the hidden variables as $\mathbf{Y}$, this is expressed as

$$\mathbf{P}(X \mid \mathbf{e}) = \alpha \sum_{\mathbf{y}} \mathbf{P}(X, \mathbf{e}, \mathbf{y}),$$

where $\alpha$ is a normalising constant and the sum ranges over all assignments to $\mathbf{Y}$.

**Inference by enumeration** evaluates this expression directly using depth-first recursion over the variables, computing sums of products of CPT entries without ever constructing the full joint distribution explicitly. Space complexity is linear in the number of variables; time complexity is $O(2^n)$ for a network of $n$ Boolean variables, since every assignment to the hidden variables must be considered.

**Variable elimination** improves on enumeration by avoiding repeated computation through dynamic programming. Rather than summing over the full joint, the algorithm works right-to-left, maintaining a set of **factors** — tables of values over subsets of variables. For each hidden variable in turn, all factors containing that variable are multiplied into a single pointwise product, and the variable is summed out of the result. The process continues until only the query variable remains, at which point the result is normalised. The key identity is that factors not involving the variable being summed out can be moved outside the summation, avoiding redundant computation that enumeration performs repeatedly (Russell & Norvig, 2010).

The computational cost of variable elimination is dominated by the size of the largest intermediate factor produced during the computation. For a factor over $m$ Boolean variables, the table has $2^m$ entries, so minimising the maximum factor size is the central concern of the algorithm.

---

### Elimination Ordering and Complexity

The order in which variables are eliminated determines the sizes of intermediate factors and hence the time and space cost of the algorithm. Different orderings can produce dramatically different factor sizes on the same network.

The complexity of variable elimination is closely related to a graph-theoretic property of the network called **treewidth**. For **singly connected networks** (polytrees), in which there is at most one undirected path between any two nodes, exact inference runs in time linear in the size of the network. For **multiply connected networks**, the worst-case complexity is exponential, and since inference in Bayesian networks subsumes inference in propositional logic, the general problem is NP-hard (Russell & Norvig, 2010). Finding the optimal elimination ordering is itself intractable, but effective heuristics are available.

This project compares three ordering strategies:

**Topological ordering** eliminates variables according to their position in the network's topological sort. This is the simplest strategy and requires no additional computation, but takes no account of the resulting factor sizes.

**Min-degree** ordering greedily eliminates whichever variable currently has the fewest neighbours in the **interaction graph** — the undirected graph in which two variables are connected if they appear together in any factor. Eliminating a variable with fewer neighbours tends to produce smaller factors.

**Min-fill** ordering greedily eliminates the variable whose elimination introduces the fewest new edges into the interaction graph. Fewer new edges implies fewer variables becoming coupled in future factors, typically leading to smaller maximum factor sizes than min-degree.

Both heuristics are greedy and do not guarantee the optimal ordering, but perform well in practice across a wide range of network structures.

---

### Approximate Inference: Rejection Sampling

For large or densely connected networks where exact inference is intractable, sampling methods provide approximate posteriors whose accuracy improves with the number of samples generated.

**Rejection sampling** generates complete samples from the prior joint distribution by sampling each variable in topological order, conditioned on the values of its parents. Samples inconsistent with the observed evidence are discarded; the remaining samples are used to estimate the posterior. Let $\hat{\mathbf{P}}(X \mid \mathbf{e})$ denote the estimated distribution. Since the accepted samples are drawn from the prior conditioned on the evidence, the estimate satisfies

$$\hat{\mathbf{P}}(X \mid \mathbf{e}) \approx \frac{\mathbf{P}(X, \mathbf{e})}{P(\mathbf{e})} = \mathbf{P}(X \mid \mathbf{e}),$$

and the estimate is consistent: it converges to the true posterior as the number of samples grows, with standard deviation of error proportional to $1/\sqrt{n}$ (Russell & Norvig, 2010).

The principal limitation of rejection sampling is sample efficiency. The fraction of samples consistent with the evidence decreases exponentially as the number of evidence variables grows or as the evidence becomes unlikely under the prior. In the extreme case, nearly all samples are rejected and the estimator requires an impractically large number of samples. This motivates likelihood weighting and Markov chain Monte Carlo methods, which are not implemented in this project but represent the natural next step.

---

## Experiments

### Enumeration vs. Variable Elimination

Both exact algorithms were applied to the same medical diagnosis network. The query

$$P(\text{HeartDisease} \mid \text{ChestPain}, \text{AbnormalECG})$$

was evaluated by both methods. The numerical difference between the two results was zero to twelve decimal places, confirming the correctness of the variable elimination implementation against the enumeration baseline.

---

### Elimination Ordering

Three synthetic network structures were benchmarked at 30 variables. The reported metric is the maximum number of variables appearing in any intermediate factor during elimination — a structural measure of inference complexity that is independent of the specific variable values.

| Network | Topological | Min-Degree | Min-Fill |
| :--- | :---: | :---: | :---: |
| Chain | 3 | 2 | 2 |
| Branching | 4 | 3 | 3 |
| Irregular | 6 | 4 | 4 |

The effect of ordering is most pronounced on the irregular network, where the maximum factor size decreases from 6 variables under topological ordering to 4 under both heuristic orderings. Since factor size is exponential in the number of variables, a reduction from 6 to 4 variables represents a fourfold reduction in the size of the largest intermediate table. The chain and branching networks show smaller gains, consistent with their more regular structure limiting the benefit of greedy reordering. Runtime measurements follow the same pattern, though at these network sizes the runtimes are small enough to be sensitive to measurement noise.

---

### Sampling Convergence

Rejection sampling was evaluated against the exact posterior for

$$P(\text{HeartDisease} = \text{true} \mid \text{ChestPain} = \text{true}, \text{AbnormalECG} = \text{true}) = 0.927817.$$

Twenty independent trials were conducted at each sample count and the mean absolute error from the exact value was recorded.

| Samples | Mean Absolute Error |
| :---: | :---: |
| 1,000 | 0.028166 |
| 10,000 | 0.010897 |
| 100,000 | 0.002329 |
| 1,000,000 | 0.000771 |

The error decreases with sample count at a rate broadly consistent with the $O(1/\sqrt{n})$ theoretical prediction: a tenfold increase in samples from 1,000 to 10,000 reduces the error by a factor of approximately 2.6, close to the expected $\sqrt{10} \approx 3.16$. The residual discrepancy reflects Monte Carlo variance across trials. The absolute errors remain non-negligible even at one million samples, which illustrates that for problems requiring high-precision posteriors, the sample requirements of rejection sampling can be prohibitive — particularly when the evidence probability under the prior is small.

---

## Reproduction

```bash
pip install -r requirements.txt

python -m experiments.exact_inference
python -m experiments.elimination_order
python -m experiments.sampling_convergence
```

---

## References

Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall. Chapter 14, *Probabilistic Reasoning*.
