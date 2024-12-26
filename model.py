# Importing Libraries
# Importing essential libraries for data manipulation (pandas), regular expressions (re), natural language processing (nltk, spacy),
# and for loading pre-trained models and vectorizers using pickle (pk).
import pandas as pd
import re, nltk, spacy
import pickle as pk

# Importing stopwords and stemming/lemmatization tools from nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Downloading necessary nltk datasets for text processing
nltk.download('punkt')  # Tokenizer data
nltk.download('punkt_tab')  # Tokenizer data
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')  # WordNet data for lemmatization
nltk.download('omw-1.4')  # Additional data for WordNet

# Loading pre-trained models and vectorizers from pickle files
count_vector = pk.load(open('pickle_file/count_vector.pkl', 'rb'))  # Pre-trained Count Vectorizer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))  # Pre-trained TF-IDF Transformer
model = pk.load(open('pickle_file/model.pkl', 'rb'))  # Pre-trained classification model
recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))  # Pre-trained user-user recommendation matrix

# Loading the spaCy language model with unnecessary components disabled for efficiency
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Reading sample product data
product_df = pd.read_csv('sample30.csv', sep=",")

# Defining Text Preprocessing Functions

def remove_special_characters(text, remove_digits=True):
    """
    Removes special characters from the input text.
    If remove_digits is True, digits are also removed.
    """
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def to_lowercase(words):
    """
    Converts all words in the input list to lowercase.
    """
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    """
    Removes punctuation and special characters from the input list of words.
    """
    return [remove_special_characters(re.sub(r'[^\w\s]', '', word), True) for word in words if word]

stopword_list = stopwords.words('english')  # List of English stopwords

def remove_stopwords(words):
    """
    Removes stopwords from the input list of words.
    """
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    """
    Stems words in the input list using the Lancaster stemmer.
    """
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    """
    Lemmatizes verbs in the input list using WordNet lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    """
    Normalizes the input list of words by converting to lowercase,
    removing punctuation, and filtering out stopwords.
    """
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    """
    Lemmatizes the input list of words.
    """
    return lemmatize_verbs(words)

# Predicting sentiment of product review comments
def model_predict(text):
    """
    Predicts the sentiment of the given text using the pre-trained classification model.
    """
    word_vector = count_vector.transform(text)  # Transform text using Count Vectorizer
    tfidf_vector = tfidf_transformer.transform(word_vector)  # Transform to TF-IDF representation
    return model.predict(tfidf_vector)  # Predict sentiment

def normalize_and_lemmaize(input_text):
    """
    Performs normalization and lemmatization on the input text.
    """
    input_text = remove_special_characters(input_text)  # Remove special characters
    words = nltk.word_tokenize(input_text)  # Tokenize text
    words = normalize(words)  # Normalize tokens
    return ' '.join(lemmatize(words))  # Lemmatize and join tokens into a single string

# Recommending products based on user sentiment
def recommend_products(user_name):
    """
    Generates product recommendations for a given user based on the user-user recommendation matrix
    and sentiment analysis of product reviews.
    """
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))  # Load recommendation matrix
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[:20])  # Top 20 recommendations
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]  # Filter product data
    output_df = product_frame[['name', 'reviews_text']]  # Select relevant columns
    output_df['lemmatized_text'] = output_df['reviews_text'].map(normalize_and_lemmaize)  # Process reviews
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])  # Predict sentiments
    return output_df

def top5_products(df):
    """
    Retrieves the top 5 products with the highest positive sentiment percentage.
    """
    total_product = df.groupby(['name']).agg('count')  # Count total reviews per product
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()  # Count sentiments per product
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')  # Merge with total reviews
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100  # Calculate positive percentage
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')  # Sort by percentage
    return pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])  # Return top 5 products
