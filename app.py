import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Heart Disease Explorer", layout="wide")
st.title("Heart Disease Dataset Explorer")


@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


df = load_data()

# Exploring the dataset
st.header("Exploring the Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Total Missing Values", int(df.isnull().sum().sum()))

with st.expander("Column Names & Data Types"):
    st.write("**Columns:**", list(df.columns))
    dtypes_df = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.count(),
            "has_null": df.isnull().any(),
        }
    )
    st.dataframe(dtypes_df, use_container_width=True)

n_rows = st.slider("Number of rows to preview", min_value=1, max_value=20, value=5)
st.subheader(f"First {n_rows} Rows (`head`)")
st.dataframe(df.head(n_rows), use_container_width=True)

st.subheader(f"Last {n_rows} Rows (`tail`)")
st.dataframe(df.tail(n_rows), use_container_width=True)

st.subheader("Summary Statistics (`describe().T`)")
st.dataframe(df.describe().T, use_container_width=True)

# Data Visualization
st.header("Data Visualization")

st.subheader("Target Distribution (`countplot`)")
fig_count, ax_count = plt.subplots(figsize=(6, 4))
sns.countplot(x="target", data=df, ax=ax_count)
ax_count.set_xlabel("Target")
ax_count.set_ylabel("Count")
st.pyplot(fig_count)
plt.close(fig_count)

st.subheader("Histograms for All Features")
fig_hist, ax_hist = plt.subplots(figsize=(15, 15))
df.hist(ax=ax_hist)
st.pyplot(fig_hist)
plt.close(fig_hist)