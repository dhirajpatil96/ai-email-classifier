import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from urgency_inference6 import predict_urgency

st.set_page_config(page_title="Email Intelligence Dashboard", layout="wide")

st.title("Email Classification Dashboard")

df = pd.read_csv("labeled_emails.csv")

# Sidebar filters
st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Category", df["category"].unique(), df["category"].unique()
)

urgency_filter = st.sidebar.multiselect(
    "Urgency", df["urgency"].unique(), df["urgency"].unique()
)

filtered_df = df[
    (df["category"].isin(category_filter)) &
    (df["urgency"].isin(urgency_filter))
]

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Emails", len(filtered_df))
c2.metric("High Urgency", len(filtered_df[filtered_df["urgency"] == "high"]))
c3.metric("Spam Emails", len(filtered_df[filtered_df["category"] == "spam"]))

st.divider()

# Charts
st.subheader("Category Distribution")
fig1, ax1 = plt.subplots()
filtered_df["category"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

st.subheader("Urgency Distribution")
fig2, ax2 = plt.subplots()
filtered_df["urgency"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
st.pyplot(fig2)

st.divider()

# Live prediction
st.subheader("Live Prediction")

email_text = st.text_area("Enter email text")

if st.button("Predict"):
    if email_text.strip():
        result = predict_urgency(email_text)
        st.success(f"Predicted Urgency: {result.upper()}")
    else:
        st.warning("Please enter email content")
