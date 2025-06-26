
# TPFL Trusted Personalized Federated Learning Framework

This repository is built upon the [PFLlib](https://github.com/TsingZ0/PFLlib) framework, with **significant modifications** in the following aspects:

- Rewritten **execution logic** for better modularity and extensibility
- Customized **checkpoint saving** mechanisms to support resumption and evaluation
- Designed **batch execution workflows** for large-scale experimental automation

---

## 🔧 Dataset Preparation

To prepare CIFAR-10 with non-IID partitioning using Dirichlet distribution, navigate to the `PFL.dataset` directory and run:

```bash
python generate_cifar10.py noniid -dir 20 0.1
````

* `20` denotes the number of clients
* `0.1` is the Dirichlet alpha parameter controlling label skewness

---

## 🚀 Running Experiments

Once the data is prepared, you can run experiments in one of the following ways:

### 1. Run a Single Algorithm

```bash
python main.py --config template/benchxxx.yaml
```

This will execute the algorithm specified under the `algorithm` field in the YAML config file.

### 2. Run Multiple Algorithms in Batch

```bash
python multimain.py --config template/benchxxx.yaml
```

This will execute all algorithms listed under the `algorithm_list` field in the YAML config file.

---

Feel free to fork or contribute. For any issues or suggestions, please open an issue or pull request.


