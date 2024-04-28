# Movie Recommendation System

This is a movie recommendation system built using Flask and Python. It recommends similar movies based on user input.

## Overview

This movie recommendation system uses cosine similarity to recommend similar movies. It processes movie data from two CSV files: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`. The system extracts relevant features such as genres, keywords, cast, crew, and overview for each movie. Then, it computes the cosine similarity between movies based on these features.

## Features

- Provides movie recommendations based on user input.
- Utilizes natural language processing (NLP) techniques for feature extraction.
- Implements cosine similarity for recommending similar movies.
- Built using Flask for the backend server.
- Frontend interface provided for user interaction.

## Requirements

- Python 3.x
- Flask
- pandas
- scikit-learn
- nltk

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/movie-recommendation-system.git
   ```
2. Download the required CSV files (tmdb_5000_movies.csv and tmdb_5000_credits.csv) and place them in the same directory as the Python scripts.

## Install the required Python packages:
```bash
pip install -r requirements.txt
```
## Usage
1. Run the Flask server:
```bash
python app.py
```
2. Open a web browser and navigate to http://localhost:5000.
3. Enter the name of a movie in the input field and click the "Recommend" button.
4. The system will display recommended movies based on the input.


