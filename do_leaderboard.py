import streamlit as st

from tabstar_paper.leaderboard.analysis.win_rate import do_win_rate_analysis
from tabstar_paper.leaderboard.data.loading import load_leaderboard_data
from tabstar_paper.leaderboard.main_results import display_main_results


def display_leaderboard():
    st.title("TabSTAR Leaderboard 🌟")
    results_tab, metadata_tab, win_tab = st.tabs(["🏆 Results", "🗂️ Data", "⚔️ Win Rate"])
    df = load_leaderboard_data()
    with results_tab:
        display_main_results(df=df)
    with metadata_tab:
        # TODO: add legacy display_datasets_metadata()
        st.info("Metadata display coming soon!")
    with win_tab:
        do_win_rate_analysis(df=df)



if __name__ == "__main__":
    display_leaderboard()
