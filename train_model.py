import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)

# ---------------- LOAD DATA ------------------
# Define dataset paths
train_dataset = 'train.csv'
test_dataset = 'test.csv'
twitter_dataset = 'training.1600000.processed.noemoticon.csv'
manual_test_dataset = 'testdata.manual.2009.06.14.csv'
amazon_dataset = 'Dataset-SA.csv'
tech_dataset = 'Sentiment Analysis Dataset.csv'

# Load datasets
try:
    train_df = pd.read_csv(train_dataset, encoding='ISO-8859-1')
    test_df = pd.read_csv(test_dataset, encoding='ISO-8859-1')
    twitter_df = pd.read_csv(twitter_dataset, encoding='ISO-8859-1', names=['polarity', 'id', 'date', 'query', 'user', 'text'], skiprows=1)
    manual_df = pd.read_csv(manual_test_dataset, encoding='ISO-8859-1', names=['polarity', 'id', 'date', 'query', 'user', 'text'])
    amazon_df = pd.read_csv(amazon_dataset, encoding='utf-8')
    tech_df = pd.read_csv(tech_dataset, encoding='utf-8')
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Drop rows with missing values
train_df = train_df.dropna(subset=['text', 'sentiment'])
test_df = test_df.dropna(subset=['text', 'sentiment'])
twitter_df = twitter_df.dropna(subset=['text', 'polarity'])
manual_df = manual_df.dropna(subset=['text', 'polarity'])
amazon_df = amazon_df.dropna(subset=['Summary', 'Sentiment'])
tech_df = tech_df.dropna(subset=['Comment', 'Sentiment'])

# Split datasets into train and test sets
amazon_train_df, amazon_test_df = train_test_split(amazon_df, test_size=0.2, random_state=42, stratify=amazon_df['Sentiment'])
tech_train_df, tech_test_df = train_test_split(tech_df, test_size=0.2, random_state=42, stratify=tech_df['Sentiment'])

# Print sentiment distribution for debugging
print("Amazon train dataset sentiment distribution:")
print(amazon_train_df['Sentiment'].value_counts())
print("\nAmazon test dataset sentiment distribution:")
print(amazon_test_df['Sentiment'].value_counts())
print("\nTech train dataset sentiment distribution:")
print(tech_train_df['Sentiment'].value_counts())
print("\nTech test dataset sentiment distribution:")
print(tech_test_df['Sentiment'].value_counts())
print("\nTrain dataset sentiment distribution:")
print(train_df['sentiment'].value_counts())
print("\nTwitter dataset polarity distribution:")
print(twitter_df['polarity'].value_counts())
print("\nManual test dataset polarity distribution:")
print(manual_df['polarity'].value_counts())

# ---------------- MAP SENTIMENT LABELS ------------------
# Map string sentiments for train.csv, test.csv, and amazon.csv
def map_string_sentiment(sentiment, rate=None):
    sentiment = str(sentiment).lower().strip()
    mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'neg': 0,
        'pos': 2
    }
    # Infer neutral for Rate=3 if sentiment is not explicit
    if rate == 3 and sentiment not in mapping:
        return 1
    return mapping.get(sentiment, None)

# Map sentiment labels for amazon_train_df and amazon_test_df
amazon_train_df['sentiment'] = amazon_train_df.apply(lambda x: map_string_sentiment(x['Sentiment'], x['Rate']), axis=1)
amazon_test_df['sentiment'] = amazon_test_df.apply(lambda x: map_string_sentiment(x['Sentiment'], x['Rate']), axis=1)
amazon_train_df = amazon_train_df.dropna(subset=['sentiment'])
amazon_test_df = amazon_test_df.dropna(subset=['sentiment'])
amazon_train_df['sentiment'] = amazon_train_df['sentiment'].astype(int)
amazon_test_df['sentiment'] = amazon_test_df['sentiment'].astype(int)
amazon_train_df['text'] = amazon_train_df['Summary']
amazon_test_df['text'] = amazon_test_df['Summary']

# Map sentiment labels for tech_train_df and tech_test_df
tech_train_df['sentiment'] = tech_train_df['Sentiment'].astype(int)
tech_test_df['sentiment'] = tech_test_df['Sentiment'].astype(int)
tech_train_df['text'] = tech_train_df['Comment']
tech_test_df['text'] = tech_test_df['Comment']

# Map sentiment labels for train.csv and test.csv
train_df['sentiment'] = train_df['sentiment'].apply(lambda x: map_string_sentiment(x))
test_df['sentiment'] = test_df['sentiment'].apply(lambda x: map_string_sentiment(x))
train_df = train_df.dropna(subset=['sentiment'])
test_df = test_df.dropna(subset=['sentiment'])
train_df['sentiment'] = train_df['sentiment'].astype(int)
test_df['sentiment'] = test_df['sentiment'].astype(int)

# Map sentiment labels for Twitter dataset
def map_twitter_sentiment(polarity):
    try:
        polarity = int(polarity)
        if polarity == 0:
            return 0  # Negative
        elif polarity == 4:
            return 2  # Positive
        return None
    except (ValueError, TypeError):
        return None

twitter_df['sentiment'] = twitter_df['polarity'].apply(map_twitter_sentiment)
twitter_df = twitter_df.dropna(subset=['sentiment'])
twitter_df['sentiment'] = twitter_df['sentiment'].astype(int)

# Map sentiment labels for manual test dataset
def map_manual_sentiment(polarity):
    try:
        polarity = int(polarity)
        if polarity == 0:
            return 0  # Negative
        elif polarity == 2:
            return 1  # Neutral
        elif polarity == 4:
            return 2  # Positive
        return None
    except (ValueError, TypeError):
        return None

manual_df['sentiment'] = manual_df['polarity'].apply(map_manual_sentiment)
manual_df = manual_df.dropna(subset=['sentiment'])
manual_df['sentiment'] = manual_df['sentiment'].astype(int)

# ---------------- TEXT PREPROCESSING ------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english')) - {'not', 'very', 'really', 'no', 'nor'}
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to all datasets
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)
twitter_df['processed_text'] = twitter_df['text'].apply(preprocess_text)
manual_df['processed_text'] = manual_df['text'].apply(preprocess_text)
amazon_train_df['processed_text'] = amazon_train_df['text'].apply(preprocess_text)
amazon_test_df['processed_text'] = amazon_test_df['text'].apply(preprocess_text)
tech_train_df['processed_text'] = tech_train_df['text'].apply(preprocess_text)
tech_test_df['processed_text'] = tech_test_df['text'].apply(preprocess_text)

# Remove empty texts after preprocessing
train_df = train_df[train_df['processed_text'] != '']
test_df = test_df[test_df['processed_text'] != '']
twitter_df = twitter_df[twitter_df['processed_text'] != '']
manual_df = manual_df[manual_df['processed_text'] != '']
amazon_train_df = amazon_train_df[amazon_train_df['processed_text'] != '']
amazon_test_df = amazon_test_df[amazon_test_df['processed_text'] != '']
tech_train_df = tech_train_df[tech_train_df['processed_text'] != '']
tech_test_df = tech_test_df[tech_test_df['processed_text'] != '']

# Combine datasets
train_df = train_df[['processed_text', 'sentiment']]
twitter_df = twitter_df[['processed_text', 'sentiment']]
manual_df = manual_df[['processed_text', 'sentiment']]
test_df = test_df[['processed_text', 'sentiment']]
amazon_train_df = amazon_train_df[['processed_text', 'sentiment']]
amazon_test_df = amazon_test_df[['processed_text', 'sentiment']]
tech_train_df = tech_train_df[['processed_text', 'sentiment']]
tech_test_df = tech_test_df[['processed_text', 'sentiment']]

# Sample Twitter dataset
twitter_df = twitter_df.sample(n=30000, random_state=42)  # Further reduced

# Oversample neutral examples
amazon_neutral = amazon_train_df[amazon_train_df['sentiment'] == 1]
tech_neutral = tech_train_df[tech_train_df['sentiment'] == 1]
if not amazon_neutral.empty:
    amazon_neutral_upsampled = resample(amazon_neutral, replace=True, n_samples=3000, random_state=42)
else:
    amazon_neutral_upsampled = pd.DataFrame(columns=['processed_text', 'sentiment'])
if not tech_neutral.empty:
    tech_neutral_upsampled = resample(tech_neutral, replace=True, n_samples=3000, random_state=42)
else:
    tech_neutral_upsampled = pd.DataFrame(columns=['processed_text', 'sentiment'])

# Combine training datasets
combined_train_df = pd.concat([amazon_train_df, tech_train_df, train_df, twitter_df, amazon_neutral_upsampled, tech_neutral_upsampled], ignore_index=True)

# Handle class imbalance
def balance_classes(df):
    df_neg = df[df['sentiment'] == 0]
    df_neu = df[df['sentiment'] == 1]
    df_pos = df[df['sentiment'] == 2]
    max_size = max(len(df_neg), len(df_neu), len(df_pos))
    df_neg_upsampled = resample(df_neg, replace=True, n_samples=max_size, random_state=42) if len(df_neg) < max_size else df_neg
    df_neu_upsampled = resample(df_neu, replace=True, n_samples=max_size, random_state=42) if len(df_neu) < max_size else df_neu
    df_pos_upsampled = resample(df_pos, replace=True, n_samples=max_size, random_state=42) if len(df_pos) < max_size else df_pos
    return pd.concat([df_neg_upsampled, df_neu_upsampled, df_pos_upsampled], ignore_index=True)

combined_train_df = balance_classes(combined_train_df)

# Combine test datasets
combined_test_df = pd.concat([amazon_test_df, tech_test_df, test_df], ignore_index=True)

# Print combined dataset distributions
print("\nCombined train dataset sentiment distribution:")
print(combined_train_df['sentiment'].value_counts())
print("\nCombined test dataset sentiment distribution:")
print(combined_test_df['sentiment'].value_counts())

# ---------------- VECTORIZE TEXT ------------------
vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
train_TFIDF = vectorizer.fit_transform(combined_train_df['processed_text'])
test_TFIDF = vectorizer.transform(combined_test_df['processed_text'])
manual_test_TFIDF = vectorizer.transform(manual_df['processed_text'])

# ---------------- TRAIN MODEL ------------------
X_train = train_TFIDF
y_train = combined_train_df['sentiment']
X_test = test_TFIDF
y_test = combined_test_df['sentiment']

# Grid search for hyperparameter tuning
param_grid = {'C': [0.1, 0.5, 1.0]}
model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, scoring='accuracy')
model.fit(X_train, y_train)
print(f"\nBest C: {model.best_params_['C']}")

# Cross-validation
cv_scores = cross_val_score(model.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Save the vectorizer and model
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model.best_estimator_, 'NB_model.pkl')

# ---------------- TEST THE MODEL ------------------
predictions = model.predict(X_test)
print(f"\nAccuracy on combined test set: {accuracy_score(y_test, predictions)}")
print("Classification Report (Combined Test Set):")
print(classification_report(y_test, predictions, target_names=['negative', 'neutral', 'positive']))

manual_predictions = model.predict(manual_test_TFIDF)
print("\nAccuracy on Manual Test Dataset:")
print(f"Accuracy: {accuracy_score(manual_df['sentiment'], manual_predictions)}")
print("Classification Report (Manual Test Dataset):")
print(classification_report(manual_df['sentiment'], manual_predictions, target_names=['negative', 'neutral', 'positive']))
