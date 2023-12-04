from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input data
keyword = "2 Casino Ransomware Attacks: Caesars, MGM"
text = "MGM reeling from cyber 'chaos' 5 days after attack as Caesars ... \n Caesars Entertainment -- which runs more than 50 resorts including, Caesars Palace and Harrah's in Las Vegas -- acknowledged the attack occurred on Sept. 7 in a filing Thursday with the U.S ..."

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the keyword and text
tfidf_matrix = vectorizer.fit_transform([keyword, text])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix)

print(f"Similarity score: {similarity[0][1]}")
