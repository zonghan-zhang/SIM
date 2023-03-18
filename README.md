# SIM (Sobol Information Maximization)

This repository reproduces the results in the paper `Explain Influence Maximization with Sobol Indices` (Zonghan Zhang et al., SIAM Data Mining 2023), whose objective is to learn the higher-order relationship in selecting seeds in information maximization problem.





## Usage

For empirical experiments in the paper, please refer to the files below:

```python
python case-study.py
```

Inside our experiments, IC (independen cascade) and LT (linear threshold) will be evaluated on each method. Details for IC and LT can be found in
```
SIM-IC.py
SIM-LT.py
```
