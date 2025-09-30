
# TODO: the code runs into a lot of edge-case errors when using some well known datasets... doing something wrong?
# OpenMLDatasetID.REG_SPORTS_MONEYBALL,
# OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN,
# OpenMLDatasetID.MUL_NATURE_EUCALYPTUS_SEED,
'''
  File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/generate_data.py", line 34, in augment_with_tabpfn
    exp_synthetic.run(
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/experiments.py", line 132, in run
    self.synthetic_X = tabpfn.generate_synthetic_data(n_samples=n_samples, t=temp)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 410, in generate_synthetic_data
    return self.impute_(
           ^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 172, in impute_
    pred = torch.stack([d for d in densities]).mean(dim=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
'''