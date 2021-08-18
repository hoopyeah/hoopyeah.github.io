Title: Notes on Quantum Advantage
Date: 2021-08-10 12:00
Tags: Quantum, AI, ML
Author: Cody Fernandez
Summary: Collecting the notes I took while reading Chapter 4 of __Supervised Learning with Quantum Computers__ by Maria Schuld and Franceso Petruccione

# Quantum Advantage
1. Computational Complexity
2. Sample Complexity
3. Model Complexity

- Asymptotic complexity - rate of growth of the runtime with size $n$ of the input
- exponential growth makes an algorithm intractable and a problem hard to solve (impossible)
- Estimating runtime of quantum algorithms is problematic
    - qubit-efficient: polynomial algorithms regarding the number of qubits
    - amplitude-efficient: efficient with respect to the number of amplitudes
- quantum complexity: asymptotic runtime complexity 
    - quantum enhancement
    - quantum advantage
    - quantum speedup
    - quantum supremacy: demonstrating an exponential speedup
- Quantum speedup, broken down
    1. Provable quantum speedup: proof there can be no classical algorithm that performs as well or better. Grover's algorithm: quadratically better than classical
    2. Strong quantum speedup: compared to best-known classical. Shor's algorithm grows polynomially rather than exponentially with the number of digits in the prime number.  
    3. Common quantum speedup: relax to "best vailable classical algorithm"
    4. Potential quantum speedup: compare two specific algorithms and relate speedup to this instance only.
    5. limited quantum speedup: compare "corresponding" algorithms such as quantum and classical annealing.
- Common Pitfalls
    1. quantum algorithms must be compared with classical sampling (non-deterministic)
    2. Complexity can be hidden in spatial resources so it's only fair to compare quantum to cluster computing (? - I don't understand the significance)
- Algorithms can be specifically "data-efficient"
- Notation
    - 'size' of the input: $M \otimes N$
    - $M$: the number of data points
    - $N$: the number of features
    - or $s$: maximum number of nonzero elements in a sparse training input
    - $\epsilon$: error of mathematical object $z$ is precision to which $z'$ claculated by the algorithm is correct. $\epsilon = \left\lvert z-z' \right\rvert$ (suitable norm)
    - $\kappa$: condition number, ratio of largest and smallest eigenvalues r a singular value. Finds an upper bound for the runtime. Gives an idea for hte eigenvalue spectrum of the matrix.
    - $p_s$: success probability. Many quantum machine learning algorithms have a chance of failure (depend on a conditional measurement) so average the number of attempts required. Factor $\frac{1}{p_s}$ into the runtime
- For quantum machine learning:
    1. Provable quadratic quantum speedups from variations of Grover's algorithm or quantum walks applied to search problems. Learning can always be a search in a space of hypotheses. 
    2. Exponential speedups: execute linear algebra computations in amplitude encoding. Comes with serious dataset conditions.
    3. Quantum annealing can be of advantage in specific sampling problems (limited advantage).
- Sample Complexity
    - statistical learning theory
    1. What does it mean to learn in a quantum setting?
    2. How can we formulate a quantum learning theoy?
- sample complexity: the number of samples needed to generalize from data.
- samples can be:
    1. examples: training instances drawn from a certain distribution
    2. queries: computing outputs to specifically chosen inputs.
- sample complexity: the number of samples required to learn a concept from a given concept class.
- Analyze sample complexity in:
    1. exact learning from membership queries
    2. Probably Approximately Correct (PAC) learning
- **classical and quantum sample complexity are polynomially equivalent**
- Explain quantum computing: *Apply a quantum oracle to a register in uniform superposition, thereby querying all possible inputs in parallel. Write the outcome into the phase and interfere the amplitudes to reveal information.*
1. **NO** exponential speedup can be expected from quantum sample complexity for exact learning with membership queries. Proven by mathematics.
2. quantum and classical learners require the same number of examples up to a constant factor. **NO** exponential speedup.
3. Robustness to noise hinted at in studies, in both cases 1 and 2. Noise made classical problem intractable, quantum problem only needed logarithmically more samples.
- Model Complexity
    - flexibility, capacity, richness, expressice power fo a model
    - We want low training error and low model flexibility. the "slimmest" model that can still learn the pattern.
    - Model complexity is least explored option
        - compare quantum models to equivalent classes of classical models
            - compare quanutm and classical Hopfield networks
        - use a unique quantum model as a useful ansatz to capture patters in a dataset.
            - use quantum models to model quantum systems: so-called QQ.