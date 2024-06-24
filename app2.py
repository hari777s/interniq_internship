import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download nltk data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to load data
@st.cache
def load_data():
    # Update this to the path where you saved the CSV file
    local_file_path = r"D:\data analyst\intership\task 2 - social media sentiment analysis\Tweets.csv"
    data = pd.read_csv(local_file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word.isalnum()]))
    return data

# Perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Main function to create the Streamlit app
def main():
    st.title("Social Media Sentiment Analysis")
    st.write("This is a web application to perform sentiment analysis on social media posts.")

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)
    
    # Display the raw data
    if st.checkbox("Show raw data"):
        st.write(data.head())

    # Sentiment analysis
    data['sentiment'] = data['text'].apply(analyze_sentiment)

    # Display sentiment counts
    st.write("### Sentiment Counts")
    sentiment_counts = data['sentiment'].value_counts()
    st.write(sentiment_counts)

    # Plot sentiment counts
    st.write("### Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='sentiment', data=data, ax=ax)
    st.pyplot(fig)

    # User input for custom sentiment analysis
    st.write("### Analyze Custom Text")
    user_input = st.text_area("Enter text here:")
    if st.button("Analyze"):
        sentiment = analyze_sentiment(user_input)
        st.write("Sentiment:", sentiment)

# Run the app
if __name__ == "__main__":
    main()
