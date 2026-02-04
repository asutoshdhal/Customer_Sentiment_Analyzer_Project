
# Customer Sentiment Analyzer using TextBlob
# Author: Asutosh Dhal
# Description: Performs sentiment analysis without ML models

import pandas as pd
from textblob import TextBlob

# Load sample customer reviews
data = pd.read_csv('../data/customer_reviews.csv')

# Function to classify sentiment
def classify_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment classification
data['Sentiment'] = data['Review'].apply(classify_sentiment)

# Extract frequent keywords
all_text = ' '.join(data['Review'].astype(str))
blob_all = TextBlob(all_text)
keywords = blob_all.words
keywords_freq = pd.Series(keywords).value_counts().head(10)

# Save results
data.to_csv('../output/sentiment_results.csv', index=False)
keywords_freq.to_csv('../output/frequent_keywords.csv', header=['Frequency'])

print("Sentiment analysis completed. Check 'output/' folder for results.")
