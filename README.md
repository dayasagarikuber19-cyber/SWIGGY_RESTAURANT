
# Swiggy Restaurant Recommendation System

 Project Overview:
This project builds a restaurant recommendation system using Swiggy restaurant data.  
Recommendations are generated based on restaurant similarity using clustering and similarity measures, and results are displayed through a Streamlit web application.

The dataset contains restaurant-level information with the following key features:
- City
- Cuisine
- Rating
- Rating Count
- Cost etc



 Data Cleaning and Preprocessing:( recomm.ipynb), how to run in terminal, New terminal : python recomm.ipynb
The raw dataset was inspected and cleaned using Pandas. The following steps were performed:

- Inspected dataset structure and data types
- Checked for missing values and duplicate records
- Removed duplicates and handled missing values
- Selected relevant numerical and categorical features
- Created two datasets:
  - **`swigy_CLEANED.csv`**: Cleaned dataset with original readable features
  - **`preprocessed_data.csv`**: Fully numerical dataset used for clustering and recommendations and saved this preprocessed data in pickle 


Recommendation Methodology:(recommendation.py), how to run in terminal, New terminal : python recommendation.py
-Cosine similarity is used in this project to recommend restaurants by measuring how similar two restaurants are based on their features such as city, cuisine, rating, and cost after preprocessing. It compares the direction of feature vectors rather than their exact magnitudes, which makes it suitable for high-dimensional data created by encoding categorical variables like city and cuisine. Since the goal is to find restaurants that are most similar to a user’s preferences, cosine similarity allows direct comparison without grouping the data into clusters.

## Streamlit Application
The Streamlit app provides:
- User input controls (rating, cost, restaurant index)
- Real-time restaurant recommendations
- Tabular display of recommended restaurants.

 How to Run in terminal:

 streamlit run st.py


