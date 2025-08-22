import streamlit as st


def display_datasets_metadata():
    original_dataset_df = get_dataset_metadata_df()
    pretrain_df = original_dataset_df[~original_dataset_df['benchmark']].copy()
    benchmark_df = original_dataset_df[original_dataset_df['benchmark']].copy()
    benchmark_df = benchmark_df.drop(columns=['benchmark', 'random', 'textual'])
    pretrain_df = pretrain_df.drop(columns=['ID'] + [c for c in pretrain_df.columns if 'benchmark' in c])

    st.markdown("## Benchmark Metadata")
    st.markdown("### CLS Benchmarks")
    cls_benchmark_df = benchmark_df[benchmark_df['task_type'].apply(lambda x: 'reg' not in x)]
    cls_benchmark_df = cls_benchmark_df.copy()
    cls_benchmark_df.drop(columns=['task_type'], inplace=True)
    st.dataframe(cls_benchmark_df)
    # with st.expander("📋 Copy LaTeX for CLS Benchmarks"):
    #     cls_benchmark_df = cls_benchmark_df.drop(columns=['context'])
    #     cls_benchmark_df = tick_benchmarks(cls_benchmark_df)
    #     caption = "CLS benchmark datasets metadata"
    #     label   = "tab:cls_benchmarks_metadata"
    #     cls_benchmark_df = urlfy_name(cls_benchmark_df)
    #     cls_benchmark_df = beautify_n_m_c(cls_benchmark_df)
    #     cls_benchmark_df = cls_benchmark_df[['ID', 'name', 'examples', 'features', 'd_output', 'SHI', 'VEC', 'CRT']]
    #     cls_benchmark_df = cls_benchmark_df.rename(columns={'name': 'Dataset', 'examples': '$n$', 'features': '$m$',
    #                                                         'd_output': '$C$'})
    #     st_latex(df=cls_benchmark_df, label=label, caption=caption)
    st.markdown("### REG Benchmarks")
    reg_benchmark_df = benchmark_df[benchmark_df['task_type'].apply(lambda x: 'regression' in x)]
    reg_benchmark_df = reg_benchmark_df.copy()
    reg_benchmark_df.drop(columns=['task_type', 'd_output'], inplace=True)
    st.dataframe(reg_benchmark_df)
    # with st.expander("📋 Copy LaTeX for REG Benchmarks"):
    #     reg_benchmark_df = reg_benchmark_df.drop(columns=['context'])
    #     reg_benchmark_df = tick_benchmarks(reg_benchmark_df)
    #     caption = "REG benchmark datasets metadata"
    #     label   = "tab:reg_benchmarks_metadata"
    #     reg_benchmark_df = urlfy_name(reg_benchmark_df)
    #     reg_benchmark_df = beautify_n_m_c(reg_benchmark_df)
    #     reg_benchmark_df = reg_benchmark_df[['ID', 'name', 'examples', 'features', 'SHI', 'VEC', 'CRT']]
    #     reg_benchmark_df = reg_benchmark_df.rename(columns={'name': 'Dataset', 'examples': '$n$', 'features': '$m$'})
    #     st_latex(df=reg_benchmark_df, label=label, caption=caption)

    # with st.expander("📋 Copy LaTeX for Context Benchmarks"):
    #     caption = "Benchmark Datasets Description"
    #     label   = "tab:benchmarks_description"
    #     context_df = benchmark_df[['ID', 'context']]
    #     st_latex(df=context_df, label=label, caption=caption)

    st.markdown("## Pretrain Datasets")
    st.markdown("### REG Pretrain Datasets")
    reg_pretrain_df = pretrain_df[pretrain_df['task_type'].apply(lambda x: 'reg' in x)]
    reg_pretrain_df = reg_pretrain_df.copy()
    reg_pretrain_df.drop(columns=['task_type', 'd_output'], inplace=True)
    st.dataframe(reg_pretrain_df)
    # with st.expander("📋 Copy LaTeX for REG Pretrain"):
    #     reg_pretrain_df['B'] = reg_pretrain_df['random'].apply(lambda x: "\checkmark" if not x else "")
    #     reg_pretrain_df['T'] = reg_pretrain_df['textual'].apply(lambda x: "\checkmark" if x else "")
    #     caption = "REG pretrain datasets metadata"
    #     label   = "tab:reg_pretrain_metadata"
    #     reg_pretrain_df = reg_pretrain_df.sort_values('examples', ascending=False)
    #     reg_pretrain_df = urlfy_name(reg_pretrain_df)
    #     reg_pretrain_df = beautify_n_m_c(reg_pretrain_df)
    #     reg_pretrain_df = reg_pretrain_df[['name', 'examples', 'features', 'B', 'T']]
    #     reg_pretrain_df = reg_pretrain_df.rename(columns={'name': 'Dataset', 'examples': '$n$', 'features': '$m$'})
    #     st_latex(df=reg_pretrain_df, label=label, caption=caption, longtable=True)

    st.markdown("### CLS Pretrain Datasets")
    cls_pretrain_df = pretrain_df[pretrain_df['task_type'].apply(lambda x: 'reg' not in x)]
    cls_pretrain_df = cls_pretrain_df.copy()
    cls_pretrain_df.drop(columns=['task_type'], inplace=True)
    st.dataframe(cls_pretrain_df)
    # with st.expander("📋 Copy LaTeX for CLS Pretrain"):
    #     cls_pretrain_df['B'] = cls_pretrain_df['random'].apply(lambda x: "\checkmark" if not x else "")
    #     cls_pretrain_df['T'] = cls_pretrain_df['textual'].apply(lambda x: "\checkmark" if x else "")
    #     caption = "CLS pretrain datasets metadata"
    #     label   = "tab:cls_pretrain_metadata"
    #     cls_pretrain_df = cls_pretrain_df.sort_values('examples', ascending=False)
    #     cls_pretrain_df = urlfy_name(cls_pretrain_df)
    #     cls_pretrain_df = beautify_n_m_c(cls_pretrain_df)
    #     cls_pretrain_df = cls_pretrain_df[['name', 'examples', 'features', 'd_output', 'B', 'T']]
    #     cls_pretrain_df = cls_pretrain_df.rename(columns={'name': 'Dataset', 'examples': '$n$', 'features': '$m$',
    #                                                       'd_output': '$C$'})
    #     st_latex(df=cls_pretrain_df, label=label, caption=caption, longtable=True)



# def st_latex(df, label, caption, longtable: bool = False):
#     latex_str = df.to_latex(
#         index=False,
#         caption=caption,
#         label=label,
#         escape=False,
#         longtable=longtable,
#         column_format="l" * df.shape[1]
#     )
#     st.code(latex_str, language="latex")
#
#
# def tick_benchmarks(df):
#     col2new = {'benchmark_CARTE': 'CRT', 'benchmark_VECTORIZING': 'VEC', 'benchmark_MULTIMODAL': 'SHI'}
#     for c_old, c_new in col2new.items():
#         df[c_new] = df[c_old].apply(lambda x: "\checkmark" if x else "")
#         df = df.drop(columns=[c_old])
#     return df
#
#
# def beautify_n_m_c(df):
#     df['examples'] = df['examples'].apply(with_commas)
#     df['features'] = df['features'].apply(with_commas)
#     return df
#
#
# def urlfy_name(df):
#     df['name'] = df.apply(name_to_url, axis=1)
#     df = df.drop(columns=['url'])
#     return df
#
# def name_to_url(row) -> str:
#     name = row['name']
#     for s in ['second-hand-mercedes-benz',
#               'nba-draft-basketball',
#               'animeplanet-recommendation',
#               'beer-profile-and-ratings',
#               'google_qa_question_type_reason',
#               'New-York-Taxi-Trip',
#               'FIFA20-Players',
#               'Myanmar-Air-Quality',
#               'New-York-Citi-Bike-Trip',
#               'climate_change_impact',
#               'GAMETES_Heterogeneity',
#               'jungle_chess',
#               'League-of-Legends-Diamond',
#               'Multiclass_Classification_for_Corporate_Credit',
#               'human-choice-prediction',
#               'meta_stream_intervals',
#               'Student_Performance',
#               ]:
#         if name.startswith(s):
#             name = s
#     return rf"\href{{{row['url']}}}{{{tex_escape(name)}}}"
#
# def tex_escape(s: str) -> str:
#     # escape only the underscore (add others if you need)
#     return s.replace("_", r"\_")
#
# def with_commas(x):
#     return f"{x:,}"
#
#
# def provide_head_to_head_comparison(original_df):
#     df = original_df.copy()[[LogKey.DATASET, LogKey.SCORE, LogKey.RUN_NUM, LogKey.MODEL]]
#     if 'TabSTAR-100K 🌟' in set(df[LogKey.MODEL]):
#         sota_model = 'TabSTAR-100K 🌟'
#     else:
#         sota_model = 'TabSTAR 🌟'
#
#     star_df = df[df[LogKey.MODEL] == sota_model]
#     star_df = star_df.rename(columns={LogKey.SCORE: 'star_score'}).drop(columns=[LogKey.MODEL])
#     other_models = set(df[LogKey.MODEL]) - {sota_model}
#     rows = []
#     for model in other_models:
#         model_df = df[df[LogKey.MODEL] == model][[LogKey.DATASET, LogKey.SCORE, LogKey.RUN_NUM]]
#         compare_df = pd.merge(model_df, star_df, on=['dataset', 'run_num'])
#         compare_df['head'] = compare_df.apply(lambda r: 1 if r.star_score > r.score else 0.5 if r.star_score == r.score else 0, axis=1)
#         ci = get_var_ci(compare_df, score_key='head')
#         avg = ci.avg * 100
#         half = ci.half * 100
#         ci_str = f"{avg:.1f}±{half:.1f}"
#         if ci_str == "100.0±0.0":
#             ci_str = "100±0.0"
#         model2short = {"CARTE 📚": 'CARTE',
#                        TabPFNv2.MODEL_NAME: 'TabPFN',
#                        TabStarTrainer.MODEL_NAME: 'TabSTAR',
#                        "CatBoost-Opt 😼": 'CatB-T',
#                        "XGBoost-Opt 🍃": 'XGB-T',
#                        CatBoost.MODEL_NAME: 'CatB',
#                        XGBoost.MODEL_NAME: 'XGB',
#                        RandomForest.MODEL_NAME: 'RF',
#                        "CatBoost-Opt-100K 😼": 'CatB-T-U',
#                        "TabSTAR-100K 🌟": 'TabSTAR-U',
#                        "XGBoost-Opt-100K 🍃": 'XGB-T-U',}
#         rows.append({LogKey.MODEL: model2short.get(model, model), 'ci_str': ci_str})
#     df = pd.DataFrame(rows).sort_values(LogKey.MODEL).set_index(LogKey.MODEL).transpose()
#     st.dataframe(df)

