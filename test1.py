import matplotlib.pyplot as pltimport pandas as pdfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import plot_confusion_matrixfrom sklearn.model_selection import train_test_splitimport streamlit as stwith st.form("my_form"):    st.write("Inside the form")    submitted = st.form_submit_button("Submit")    if submitted:        st.write("you clicked the submit button")