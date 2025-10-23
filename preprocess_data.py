"""
AI-Powered Email Spam Detection - Data Preprocessing Module
This module handles data loading, cleaning, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SpamDataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the SMS Spam Collection file
        """
        self.data_path = data_path
        self.df = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        
    def load_data(self):
        """Load the SMS Spam Collection dataset"""
        print("Loading dataset...")
        # Load data with tab separator
        self.df = pd.read_csv(
            self.data_path, 
            sep='\t', 
            names=['label', 'message'],
            encoding='utf-8'
        )
        print(f"Dataset loaded: {len(self.df)} messages")
        print(f"Spam messages: {len(self.df[self.df['label'] == 'spam'])}")
        print(f"Ham messages: {len(self.df[self.df['label'] == 'ham'])}")
        return self.df
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize text and apply stemming
        
        Args:
            text: Cleaned text string
            
        Returns:
            Stemmed text string
        """
        # Tokenize
        tokens = text.split()
        
        # Remove stop words and apply stemming
        stemmed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(stemmed_tokens)
    
    def preprocess_data(self):
        """Apply all preprocessing steps to the dataset"""
        print("\nPreprocessing data...")
        
        # Clean text
        print("- Cleaning text (removing URLs, HTML, punctuation, numbers)...")
        self.df['cleaned_message'] = self.df['message'].apply(self.clean_text)
        
        # Tokenize and stem
        print("- Tokenizing and stemming...")
        self.df['processed_message'] = self.df['cleaned_message'].apply(
            self.tokenize_and_stem
        )
        
        # Convert labels to binary (spam=1, ham=0)
        self.df['label_binary'] = self.df['label'].map({'spam': 1, 'ham': 0})
        
        # Remove empty messages after preprocessing
        self.df = self.df[self.df['processed_message'].str.len() > 0]
        
        print(f"Preprocessing complete. Final dataset size: {len(self.df)}")
        
        return self.df
    
    def extract_features(self, max_features=3000):
        """
        Extract TF-IDF features from processed messages
        
        Args:
            max_features: Maximum number of features to extract
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        print(f"\nExtracting TF-IDF features (max_features={max_features})...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Fit and transform
        X = self.tfidf_vectorizer.fit_transform(self.df['processed_message'])
        y = self.df['label_binary'].values
        
        print(f"Feature extraction complete. Shape: {X.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, output_path):
        """Save the TF-IDF vectorizer for future use"""
        print(f"\nSaving TF-IDF vectorizer to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print("Vectorizer saved successfully")
    
    def save_processed_data(self, output_path):
        """Save the preprocessed dataframe"""
        print(f"\nSaving preprocessed data to {output_path}...")
        self.df.to_csv(output_path, index=False)
        print("Preprocessed data saved successfully")


def main():
    """Main execution function"""
    # Set paths
    data_path = '/home/ubuntu/spam_detection/data/SMSSpamCollection'
    output_dir = '/home/ubuntu/spam_detection/data'
    
    # Initialize preprocessor
    preprocessor = SpamDataPreprocessor(data_path)
    
    # Load data
    df = preprocessor.load_data()
    
    # Preprocess data
    df = preprocessor.preprocess_data()
    
    # Extract features
    X, y = preprocessor.extract_features(max_features=3000)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Save preprocessor and data
    preprocessor.save_preprocessor(
        os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    )
    preprocessor.save_processed_data(
        os.path.join(output_dir, 'preprocessed_data.csv')
    )
    
    # Save train/test splits
    print("\nSaving train/test splits...")
    np.savez(
        os.path.join(output_dir, 'train_test_data.npz'),
        X_train=X_train.toarray(),
        X_test=X_test.toarray(),
        y_train=y_train,
        y_test=y_test
    )
    print("Train/test splits saved successfully")
    
    print("\n" + "="*50)
    print("Data preprocessing completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()

