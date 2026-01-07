import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("Swiggy Restaurant Recommendation System")

try:
    cleaned_df = pd.read_csv("swiggy_CLEANED.csv")
    
except Exception as e:
    st.error(e)
    st.stop()

try:
    with open("preprocessed_data.pkl", "rb") as f:
        preprocessed_df = pickle.load(f)
    
except Exception as e:
    st.error(e)
    st.stop()

rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0)
cost = st.slider("Maximum Cost", 100, 1000, 500)

st.subheader("Select Restaurant Index")

index = st.number_input(
    "Enter restaurant index",
    min_value=0,
    max_value=len(preprocessed_df) - 1,
    value=0,
    step=1
)

user_vector = preprocessed_df.iloc[index]
user_vector = pd.DataFrame([user_vector])
user_vector["rating"] = rating
user_vector["cost"] = cost

similarity = cosine_similarity(user_vector, preprocessed_df)[0]
top_idx = np.argsort(similarity)[::-1][:5]

st.subheader("Recommended Restaurants")
st.dataframe(
    cleaned_df.iloc[top_idx][["rating_count", "city", "cuisine", "rating", "cost"]]
)
