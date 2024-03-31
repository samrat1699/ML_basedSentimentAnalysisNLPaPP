
import streamlit as st
import pickle
import string
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
import spacy

# Define additional stop words
additional_stopwords = set(['br'])

# Combine Gensim and additional stop words
combined_stopwords = STOPWORDS.union(gensim.parsing.preprocessing.STOPWORDS).union(additional_stopwords)

# Load SpaCy English model for lemmatization
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Function for preprocessing text
def transform_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(' +', ' ', text)
    
    # Tokenize the text
    text = simple_preprocess(text)
    
    # Remove non-alphanumeric characters and apply stop words
    y = []
    for i in text:
        if i.isalnum() and i not in combined_stopwords:
            y.append(i)
    
    # Apply bigrams
    bigram = Phrases(y, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    y = bigram_mod[y]
    
    # Apply trigrams
    trigram = Phrases(bigram_mod[y], min_count=5, threshold=100)
    trigram_mod = Phraser(trigram)
    y = trigram_mod[bigram_mod[y]]
    
    # Lemmatization using SpaCy
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    y = lemmatization([y])[0]
    
    return " ".join(y)

# Load the trained vectorizer and model
tfidf = pickle.load(open('models\\vectorizer.pkl', 'rb'))
model = pickle.load(open('models\\model.pkl', 'rb'))

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    processed_text = transform_text(text)
    
    # Vectorize the processed text
    vectorized_text = tfidf.transform([processed_text])
    
    # Predict sentiment using the model
    prediction = model.predict(vectorized_text)
    
    return prediction[0]

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    # st.write("Enter a sentence to predict its sentiment:")
    
    # User input
    user_input = st.text_area("Enter a sentence:")
    
    # Predict sentiment when the user clicks the button
    if st.button("Predict"):
        # Perform prediction
        prediction = predict_sentiment(user_input)
        
        # Display prediction
        if prediction == 1:
            st.write("Sentiment: Positive")
        else:
            st.write("Sentiment: Negative")

if __name__ == '__main__':
    main()
