import pandas as pd
import pickle
from sklearn.cluster import KMeans

cleaned_df = pd.read_csv("swiggy_CLEANED.csv")

with open("preprocessed_data.pkl", "rb") as f:
    preprocessed_df = pickle.load(f)

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
kmeans.fit(preprocessed_df)

cleaned_df = cleaned_df.reset_index(drop=True)
preprocessed_df = preprocessed_df.reset_index(drop=True)

cleaned_df["cluster"] = kmeans.labels_

def recommend_restaurants(index, top_n=5):

    cluster_id = cleaned_df.loc[index, "cluster"]

    same_cluster = cleaned_df[
        cleaned_df["cluster"] == cluster_id
    ].drop(index)

    same_cluster = same_cluster.sort_values(
        by=["rating", "rating_count"],
        ascending=[False, False]
    )

    return same_cluster.head(top_n)

if __name__ == "__main__":

    input_index = int(input("Enter restaurant index: "))

    recommendations = recommend_restaurants(input_index)

    print("\nRecommended Restaurants:\n")
    print(
        recommendations[
            ["city", "cuisine", "cost", "rating", "rating_count"]
        ]
    )
