
# TODO: REG_HOUSES_CALIFORNIA_PRICES_2020 fails somewhere in the middle where max_examples = 1000, max_features = 10
'''
💾 Downloading OpenML dataset REG_HOUSES_CALIFORNIA_PRICES_2020
 30%|███       | 3/10 [00:35<01:21, 11.67s/it]
Traceback (most recent call last):
  File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/do_augmented_benchmark.py", line 74, in <module>
    eval_augmented_benchmark()
  File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/do_augmented_benchmark.py", line 67, in eval_augmented_benchmark
    res = evaluate_on_augmented_dataset(model_cls=t.model, dataset_id=t.dataset,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/evaluator.py", line 36, in evaluate_on_augmented_dataset
    synth_x, synth_y = augment_with_tabpfn(x_train, y_train, is_cls=is_cls, cache_key=cache_key)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/generate_data.py", line 34, in augment_with_tabpfn
    exp_synthetic.run(
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/experiments.py", line 132, in run
    self.synthetic_X = tabpfn.generate_synthetic_data(n_samples=n_samples, t=temp)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 410, in generate_synthetic_data
    return self.impute_(
           ^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 158, in impute_
    _, pred = self.impute_single_permutation_(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 214, in impute_single_permutation_
    model, X_predict, _ = self.density_(
                          ^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 278, in density_
    model.fit(X_fit.numpy(), y_fit.numpy())
  File "/Users/alanarazi/.pyenv/versions/3.11.4/lib/python3.11/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn/regressor.py", line 449, in fit
    X, y, feature_names_in, n_features_in = validate_Xy_fit(
                                            ^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn/utils.py", line 353, in validate_Xy_fit
    X, y = validate_data(
           ^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn/misc/_sklearn_compat.py", line 435, in validate_data
    return _estimator._validate_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/sklearn/base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1318, in check_X_y
    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1340, in _check_y
    _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/sklearn/utils/validation.py", line 123, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/sklearn/utils/validation.py", line 172, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
  ValueError: Input y contains NaN.
    '''

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