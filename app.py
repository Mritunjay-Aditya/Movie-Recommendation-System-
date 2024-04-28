import pandas as pd
import numpy as np
import ast
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge data
movies = movies.merge(credits, on="title")
movies.dropna(inplace=True)

# Function to convert objects
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break     
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L

# Apply conversion functions
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert3)
movies["crew"] = movies["crew"].apply(fetch_director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())
movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ","") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ","") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ","") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ","") for i in x])

# Create 'tags' column
movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

# Create new DataFrame with required columns
new_df = movies[["movie_id","title","tags"]]

# Process 'tags' column
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].str.lower()

# Stemming
ps = PorterStemmer()
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

# Cosine Similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def get_similar_movies(movie):
    movie = ps.stem(movie.lower())
    movie_index = new_df[new_df["title"].apply(lambda x: ps.stem(x.lower())) == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    print(recommended_movies)
    return recommended_movies
get_similar_movies("Pirates of the Caribbean: At World's End")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        movie = request.form['movie']
        recommended_movies = get_similar_movies(movie)
        return render_template('index.html', recommended_movies=recommended_movies)
    except Exception as e:
        print(e)
        return render_template('index.html', recommended_movies=["Sorry, an error occurred."])

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
