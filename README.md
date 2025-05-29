# TabSTAR Research Repository

<img src="tabstar_logo.png" alt="TabSTAR Logo" width="50%">

Welcome to the TabSTAR Research repo!

🚧 This repository is still work in progress. 🚧

---

## 📚 Resources

* 📄 **Paper**: [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)
* 🌐 **Project Website**: [TabSTAR](https://eilamshapira.com/TabSTAR/)

---

To install the repository, run:

```bash
source init.sh
```

The main scripts provided are:

* `do_pretrain` – pretrain a TabSTAR model.
* `do_finetune` – finetune a pretrained TabSTAR model on a downstream task.
* `do_baseline` – run a different baseline model on a downstream task.

<img src="tabstar_arch.png" alt="TabSTAR Arch" width="100%">

### Pretraining

To pretrain TabSTAR with a specified number of datasets:

```bash
python do_pretrain.py --n_datasets=256
```

For debugging, you can decrease this number (note it may impact downstream performance).

### Finetuning

After pretraining, take the printed `<pretrain_exp>` identifier and run:

```bash
python do_finetune.py --pretrain_exp=<PRETRAINED_EXP> --dataset_id=46655
```

### Baseline Comparison

To compare against a baseline model:

```bash
python do_baseline.py --model=rf --dataset_id=46655
```

---

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use TabSTAR in your research, please cite:

```bibtex
@article{arazi2025tabstarf,
  title   = {TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations},
  author  = {Alan Arazi and Eilam Shapira and Roi Reichart},
  journal = {arXiv preprint arXiv:2505.18125},
  year    = {2025},
}
```
