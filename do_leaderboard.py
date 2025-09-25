import streamlit as st

from tabstar_paper.leaderboard.analysis.experiments import display_experiments_info
from tabstar_paper.leaderboard.analysis.win_rate import do_win_rate_analysis
from tabstar_paper.leaderboard.data.loading import load_leaderboard_data
from tabstar_paper.leaderboard.main_results import display_main_results
from tabstar_paper.leaderboard.metadata_info import display_metadata_info


def display_leaderboard():
    st.title("TabSTAR Leaderboard 🌟")
    # Add experiments emoji
    results, data, win, meta, exp = st.tabs(["🏆 Results", "🗂️ Data", "⚔️ Win Rate", "ℹ️️ Metadata",  "🧪 Experiments"])
    df = load_leaderboard_data()
    with results:
        display_main_results(df=df)
    with data:
        # TODO: add legacy display_datasets_metadata()
        st.info("Dataset display coming soon!")
    with win:
        do_win_rate_analysis(df=df)
    with meta:
        display_metadata_info()
    with exp:
        display_experiments_info(leaderboard_df=df)



if __name__ == "__main__":
    display_leaderboard()
