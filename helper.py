import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  # Import stopwords library
from nltk.stem import PorterStemmer  # Import PorterStemmer for stemming
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Preprocess text data
def preprocess_text(texts):
  """
  Preprocesses text data by performing the following steps:
    - Lowercase conversion
    - Stopword removal
    - Stemming (optional)
  """
  # Lowercase text
  for text in texts:
    text = text.lower()

  # Remove stopwords
  # stop_words = set(stopwords.words('english'))
  # words = [word for word in text.split() if word not in stop_words]

  # Stem words (optional)
  stemmer = PorterStemmer()
  # Uncomment the following line to enable stemming
  # words = [stemmer.stem(word) for word in words]

  # Join words back into text
  # text = ' '.join(words)
  
  return text

def preprocess_data(data):
  """
  Preprocesses text data for machine learning:

  1. Segment with word pauses, tokenize each data.
  2. Filter out tokens with high/low occurrence rates.
  3. Vectorize filtered tokens using TF-IDF.
  4. Convert to sparse matrix.

  Args:
      data: A pandas DataFrame with 'Sentence' and 'Label' columns.

  Returns:
      A tuple containing the processed features (X) and labels (y).
  """

  # (1) Segment and tokenize (simulation for this example)
  data['Sentence'] = data['Sentence'].apply(lambda x: x.split())

  # Placeholder for token filtering (replace with actual implementation)
  # This part would typically involve calculating occurrence rates and filtering
  # based on your criteria (80% and 2 occurrences)
  filtered_tokens = data['Sentence'].sum()  # Placeholder

  # (3) Vectorize using TF-IDF
  vectorizer = TfidfVectorizer(vocabulary=filtered_tokens)
  X = vectorizer.fit_transform(data['Sentence'])

  # (4) Convert to sparse matrix
  X = X.toarray()  # Convert to dense for simplicity (can use sparse if needed)

  y = data['Label'].values

  return X, y