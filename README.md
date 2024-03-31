# Sentiment Analysis App

This is a sentiment analysis application built with Python, Streamlit, Gensim, and SpaCy.

## Description

The application takes a sentence as input and predicts its sentiment as either positive or negative. The prediction is done using a pre-trained model.

## Installation

1. Clone the repository
2. Install the dependencies with `pip install -r requirements.txt`

## Usage

1. Run the Streamlit app with `streamlit run app.py`
2. Enter a sentence in the text box
3. Click "Predict" to get the sentiment of the sentence

## Files

- `app.py`: The main application file
- `models/vectorizer.pkl`: The trained TF-IDF vectorizer
- `models/model.pkl`: The pre-trained sentiment analysis model

## Libraries Used

- Streamlit: For creating the web application
- Gensim: For text preprocessing and creating bigrams and trigrams
- SpaCy: For text lemmatization
- Pickle: For loading the pre-trained model and vectorizer