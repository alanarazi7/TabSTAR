from os.path import dirname, join

from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS


def get_dataset_metadata_df():
    datasets_dir = join(dirname(__file__), 'analysis')
    df = pd.read_csv(join(datasets_dir, 'full_datasets_metadata.csv'))
    text_datasets = {get_sid(d) for d in set(NON_BENCHMARK_TEXTUAL).union(TEXTUAL_DATASETS)}
    df['textual'] = df['sid'].apply(lambda d: d in text_datasets)
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    df.drop(columns=feat_cols, inplace=True)

    df['name'] = df.apply(get_dataset_name, axis=1)
    df['url'] = df['sid'].apply(get_dataset_url)
    df['benchmark'] = df['sid'].apply(is_textual_dataset)
    df['random'] = df.apply(is_random_dataset, axis=1)
    df.drop(columns=['benchmark_AMLB', 'benchmark_TABZILLA', 'benchmark_CTR23', 'benchmark_GRINSZTAJN'], inplace=True)
    df['ID'] = df['sid'].apply(lambda s: BENCH2IDX.get(get_dataset_from_sid(s), ""))
    df.drop(columns=['sid'], inplace=True)
    assert 'benchmark' in df.columns
    df = df.sort_values(by=['ID'])
    return df