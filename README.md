# Movie-Recommendation-Using-ML-Models
Introduction

In the era of digital information overload, recommendation systems play a crucial role in helping users discover content that matches their interests. One of the most popular applications of such systems is in the movie industry, where users are often overwhelmed by the vast number of choices available. In this project, we develop a movie recommendation system using machine learning techniques to predict user preferences based on historical data. Our objective is to implement and compare the performance of multiple models in order to identify the most effective approach for building a personalized recommendation system.

Data Collection

Netflix Movies & Shows Dataset, available on Kaggle, is an inclusive collection framing by: https://www.kaggle.com/datasets/ashfakyeafi/netflix-movies-and-shows-dataset/data. There are a total 8807 records and 12 columns available in the dataset in which there are 2 numerical columns and 11 categorical columns.

Modeling

After cleaning and preparing the data, categorical columns were encoded and multi-label genres were transformed using MultiLabelBinarizer. Movie descriptions were vectorized with TF-IDF to capture semantic meaning.

We experimented with multiple models:

Random Forest (OvR): Limited to 50% of data due to technical issues, achieving low Precision@k (P@1 = 8.5%).
KNN: Used standardized data and TF-IDF features. Precision improved slightly but remained modest (P@1 ≈ 10–15%). Also implemented a similarity-based movie suggestion function.
SVM (LinearSVC): Best performing model with strong results (P@1 = 0.45, P@5 = 0.85) before fine-tuning, competitive with benchmark studies. After GridSearch, performance declined due to class imbalance.
KMeans: Applied clustering on TF-IDF vectors of genres. Determined optimal k = 5 via Elbow Method, with Silhouette Score improving from 0.24 to 0.42 after tuning. A recommendation function suggests movies from the same cluster using cosine similarity.
Model comparison

It showed that SVM had the highest accuracy among supervised models, while KMeans offered meaningful genre-based clustering. Overall, the system demonstrates the trade-offs of different approaches and provides both predictive and similarity-based recommendations.
