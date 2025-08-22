import streamlit as st

from tabstar_paper.leaderboard.datasets_metadata import display_datasets_metadata
from tabstar_paper.leaderboard.main_results import display_main_results


def display_leaderboard():
    st.title("TabSTAR Leaderboard 🌟")
    results_tab, metadata_tab = st.tabs(["🏆 Results", "🗂️ Data"])
    with results_tab:
        display_main_results()
    with metadata_tab:
        display_datasets_metadata()



if __name__ == "__main__":
    display_leaderboard()
