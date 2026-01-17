import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


#  Load cleaned data and preprocessed data:

cleaned_df = pd.read_csv("swiggy_CLEANED.csv")
with open("preprocessed_data.pkl", "rb") as f:
    preprocessed_df = pickle.load(f)

#  Reset indices to align cleaned_df and preprocessed_df:

cleaned_df = cleaned_df.reset_index(drop=True)
preprocessed_df = preprocessed_df.reset_index(drop=True)


from sklearn.metrics.pairwise import cosine_similarity

def recommend_restaurants(base_filters, top_n=5):
    """
    base_filters: dict with keys 'city', 'cuisine', 'rating', 'cost'
    """
    # Filter DataFrame first (city, cuisine, rating, cost)
    subset = cleaned_df[
        (cleaned_df["city"] == base_filters['city']) &
        (cleaned_df["cuisine"] == base_filters['cuisine']) &
        (cleaned_df["rating"] >= base_filters['rating']) &
        (cleaned_df["cost"] <= base_filters['cost'])
    ]
    
    if subset.empty:
        # If nothing matches exactly, pick first row as base
        base_index = 0
    else:
        base_index = subset.index[0]
    
    subset_vectors = preprocessed_df.loc[subset.index]

    base_vector = preprocessed_df.loc[[base_index]]
    similarities = cosine_similarity(base_vector, subset_vectors)[0]

    top_indices = subset.index[similarities.argsort()[::-1][:top_n]]
    return cleaned_df.loc[top_indices]

