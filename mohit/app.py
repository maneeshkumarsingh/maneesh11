import main
import analysis
import Retrain
import streamlit as st



PAGES = {
    "SPAM Prediction": main,
    "Training & Test": analysis,
    "Retrain Model": Retrain
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()




