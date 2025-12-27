import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.title("💳 KNN Fraud Detection App")
st.write("Enter transaction details to check whether it is **Fraud** or **Genuine**")


df = pd.read_csv("creditcard.csv")


st.sidebar.header("🔢 Enter Input Values")

amount = st.sidebar.number_input("Transaction Amount", value=0.0)
feature2 = st.sidebar.number_input("Second Feature", value=0.0)


if st.button("Predict"):
    # This is frontend demo output
    if amount > 2000:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")


st.subheader("📊 Dataset Visualization")

if st.checkbox("Show Scatter Plot"):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df[x_col],
        y=df[y_col],
        hue=df['Class'],
        palette={0:'blue', 1:'red'},
        alpha=0.6,
        ax=ax
    )

    ax.set_title("Fraud vs Genuine Transactions")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    st.pyplot(fig)
