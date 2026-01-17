import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(layout="wide")
st.title("Restaurant Recommendation System")

# Load data
cleaned_df = pd.read_csv("swiggy_CLEANED.csv")
preprocessed_df = pd.read_pickle("preprocessed_data.pkl")

# Sidebar visuals
st.sidebar.header("Visualizations")

fig1, ax1 = plt.subplots()
ax1.hist(cleaned_df["rating"], bins=20)
st.sidebar.pyplot(fig1)

# User preferences (center)
st.subheader("Select Preferences")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        sorted(cleaned_df["city"].astype(str).unique())
    )
    min_rating = st.slider(
        "Minimum Rating",
        0.0, 5.0, 4.0, 0.1
    )

with col2:
    cuisine = st.selectbox(
        "Cuisine",
        sorted(cleaned_df["cuisine"].astype(str).unique())
    )
    max_cost = st.slider(
        "Maximum Cost",
        int(cleaned_df["cost"].min()),
        int(cleaned_df["cost"].max()),
        2000,
        500
    )

# Create user vector
user_vector = pd.DataFrame(
    np.zeros((1, preprocessed_df.shape[1])),
    columns=preprocessed_df.columns
)

user_vector["rating"] = min_rating
user_vector["cost"] = max_cost

if city in user_vector.columns:
    user_vector[city] = 1

if cuisine in user_vector.columns:
    user_vector[cuisine] = 1

# Cosine similarity
similarities = cosine_similarity(user_vector, preprocessed_df)[0]
top_indices = similarities.argsort()[::-1][:5]

# Output
st.subheader("Recommended Restaurants")

st.dataframe(
    cleaned_df.iloc[top_indices][
        ["name", "city", "cuisine", "cost", "rating", "rating_count"]
    ],
    use_container_width=True
)
