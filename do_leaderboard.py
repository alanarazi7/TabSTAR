import streamlit as st

from tabstar_paper.leaderboard.analysis.win_rate import do_win_rate_analysis
from tabstar_paper.leaderboard.data.loading import load_leaderboard_data
from tabstar_paper.leaderboard.main_results import display_main_results
from tabstar_paper.leaderboard.metadata_info import display_metadata_info


def display_leaderboard():
    st.title("TabSTAR Leaderboard 🌟")
    results_tab, data_tab, win_tab, metadata_tab = st.tabs(["🏆 Results", "🗂️ Data", "⚔️ Win Rate", "ℹ️️ Metadata"])
    df = load_leaderboard_data()
    with results_tab:
        display_main_results(df=df)
    with data_tab:
        # TODO: add legacy display_datasets_metadata()
        st.info("Dataset display coming soon!")
    with win_tab:
        do_win_rate_analysis(df=df)
    with metadata_tab:
        display_metadata_info()



if __name__ == "__main__":
    display_leaderboard()
