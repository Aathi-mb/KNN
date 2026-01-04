import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.title("💳 KNN Fraud Detection App")
st.write("Enter transaction details to check whether it is Fraud or Genuine")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Example: Train simple KNN (for demo)
    # Use first 100 rows for speed
    df_sample = df.sample(100)
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    st.write("Model trained on sample data")

    # Optional: Let user input new transaction
    st.subheader("Predict new transaction")
    input_data = []
    for col in X.columns[:5]:  # just example first 5 features
        val = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        pred = model.predict([input_data])
        if pred[0] == 1:
            st.error("⚠️ Fraud Transaction!")
        else:
            st.success("✅ Genuine Transaction")
