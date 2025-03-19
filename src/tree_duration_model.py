#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class DurationModel:
    """Model for predicting phoneme unit duration"""
    
    def __init__(self):
        self.phone_encoder = LabelEncoder()
        self.word_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.phone_stats = {}  # Store statistics for each phoneme
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data, convert categorical features to numerical features, and add context features"""
        # Encode phonemes and words
        df['phone_encoded'] = self.phone_encoder.fit_transform(df['phone'])
        
        # Add more features: one-hot encoding of phoneme categories (vowel/consonant)
        df['is_vowel'] = df['phone_class'].apply(lambda x: 1 if x == 'V' else 0)
        df['is_consonant'] = df['phone_class'].apply(lambda x: 1 if x == 'C' else 0)
        df['is_silence'] = df['phone_class'].apply(lambda x: 1 if x == 'sil' else 0)
        
        # Phoneme position features in word
        df = self._add_position_features(df)
        
        # Phoneme context features
        df = self._add_context_features(df)
        
        # Add word position features in sentence
        df = self._add_sentence_position_features(df)
        
        # Select features
        features = df[[
            'phone_encoded', 
            'is_vowel', 'is_consonant', 'is_silence',
            'phone_position_first', 'phone_position_last',
            'phone_relative_position', 'word_length',
            'prev_phone_class_V', 'prev_phone_class_C', 'prev_phone_class_sil',
            'next_phone_class_V', 'next_phone_class_C', 'next_phone_class_sil',
            # New sentence position features
            'word_position_in_sentence', 'word_relative_position_in_sentence',
            'is_first_word_in_sentence', 'is_last_word_in_sentence', 'sentence_length'
        ]]
        X = features.values
        y = df['duration'].values
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add phoneme and word position features"""
        print("Starting to process phoneme position features...")
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        # Initialize new features
        df['phone_position_first'] = 0
        df['phone_position_last'] = 0
        df['phone_relative_position'] = 0
        df['word_length'] = 0
        
        total_groups = len(df.groupby(['task_id', 'word']))
        for i, ((task_id, word), group) in enumerate(df.groupby(['task_id', 'word'])):
            if i % 100 == 0:  # Print progress every 100 groups
                print(f"Processing phoneme position features: {i}/{total_groups} ({(i/total_groups*100):.1f}%)")
            
            if word == 'sil' or pd.isna(word):
                continue
                
            # Get group indices
            indices = group.index
            
            # Word length (number of phonemes)
            word_length = len(indices)
            df.loc[indices, 'word_length'] = word_length
            
            # Phoneme position
            if word_length > 0:
                df.loc[indices[0], 'phone_position_first'] = 1
                df.loc[indices[-1], 'phone_position_last'] = 1
                
                # Relative position (between 0 and 1)
                for i, idx in enumerate(indices):
                    df.loc[idx, 'phone_relative_position'] = i / max(1, (word_length - 1))
        
        return df
    
    def _add_sentence_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add word position features in sentence"""
        print("Starting to process sentence position features...")
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        # Initialize new features
        df['word_position_in_sentence'] = 0
        df['word_relative_position_in_sentence'] = 0
        df['is_first_word_in_sentence'] = 0
        df['is_last_word_in_sentence'] = 0
        df['sentence_length'] = 0
        
        total_tasks = len(df['task_id'].unique())
        for i, (task_id, group) in enumerate(df.groupby('task_id')):
            if i % 50 == 0:  # Print progress every 50 tasks
                print(f"Processing sentence position features: {i}/{total_tasks} ({(i/total_tasks*100):.1f}%)")
            
            # Get all unique words in the sentence (in order)
            sentence_words = []
            
            # Sort phonemes by start time and extract unique words in order
            sorted_group = group.sort_values('start')
            for _, row in sorted_group.iterrows():
                word = row['word']
                if word != 'sil' and not pd.isna(word) and (not sentence_words or word != sentence_words[-1]):
                    sentence_words.append(word)
            
            # Sentence length (number of words)
            sentence_length = len(sentence_words)
            
            # Set position information for each word in the sentence
            for i, word in enumerate(sentence_words):
                # Find all phonemes of this word in DataFrame
                word_indices = group[group['word'] == word].index
                
                # Set position features
                df.loc[word_indices, 'word_position_in_sentence'] = i + 1  # 1-based position
                df.loc[word_indices, 'word_relative_position_in_sentence'] = i / max(1, (sentence_length - 1))
                df.loc[word_indices, 'sentence_length'] = sentence_length
                
                # Set first/last word markers
                if i == 0:
                    df.loc[word_indices, 'is_first_word_in_sentence'] = 1
                if i == sentence_length - 1:
                    df.loc[word_indices, 'is_last_word_in_sentence'] = 1
        
        return df
    
    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add phoneme context features"""
        print("Starting to process context features...")
        df = df.copy()
        
        # Initialize previous and next phoneme class features
        df['prev_phone_class_V'] = 0
        df['prev_phone_class_C'] = 0
        df['prev_phone_class_sil'] = 0
        df['next_phone_class_V'] = 0
        df['next_phone_class_C'] = 0
        df['next_phone_class_sil'] = 0
        
        total_tasks = len(df['task_id'].unique())
        for i, (task_id, group) in enumerate(df.groupby('task_id')):
            if i % 50 == 0:  # Print progress every 50 tasks
                print(f"Processing context features: {i}/{total_tasks} ({(i/total_tasks*100):.1f}%)")
            
            indices = group.index
            phone_classes = group['phone_class'].values
            
            # Add previous phoneme class
            for i in range(1, len(indices)):
                prev_class = phone_classes[i-1]
                if prev_class == 'V':
                    df.loc[indices[i], 'prev_phone_class_V'] = 1
                elif prev_class == 'C':
                    df.loc[indices[i], 'prev_phone_class_C'] = 1
                elif prev_class == 'sil':
                    df.loc[indices[i], 'prev_phone_class_sil'] = 1
            
            # Add next phoneme class
            for i in range(len(indices)-1):
                next_class = phone_classes[i+1]
                if next_class == 'V':
                    df.loc[indices[i], 'next_phone_class_V'] = 1
                elif next_class == 'C':
                    df.loc[indices[i], 'next_phone_class_C'] = 1
                elif next_class == 'sil':
                    df.loc[indices[i], 'next_phone_class_sil'] = 1
        
        return df
    
    def fit(self, df: pd.DataFrame):
        """Train the model"""
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model on validation set
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print("Model evaluation results:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R^2): {r2:.4f}")
        
        # Calculate statistics (mean and variance) for each phoneme
        self.calculate_phone_stats(df)
        
        return self
    
    def predict(self, phone_sequence: List[str], word_sequence: List[str] = None) -> np.ndarray:
        """Predict duration for given phoneme sequence"""
        n_phones = len(phone_sequence)
        
        # If word sequence is not provided, use unknown word
        if word_sequence is None:
            word_sequence = ['UNK'] * n_phones
        
        # Build features
        features = []
        for i in range(n_phones):
            phone = phone_sequence[i]
            word = word_sequence[i]
            
            # Check if phoneme and word are in training set
            try:
                phone_encoded = self.phone_encoder.transform([phone])[0]
            except:
                print(f"Warning: Unknown phoneme '{phone}', using default value")
                phone_encoded = 0
                
            
            # Determine phoneme class
            if phone.lower() in ['a', 'e', 'i', 'o', 'u']:
                phone_class = 'V'
                is_vowel, is_consonant, is_silence = 1, 0, 0
            elif phone.lower() == 'sil':
                phone_class = 'sil'
                is_vowel, is_consonant, is_silence = 0, 0, 1
            else:
                phone_class = 'C'
                is_vowel, is_consonant, is_silence = 0, 1, 0
            
            # Position features
            phone_position_first = 1 if i == 0 else 0
            phone_position_last = 1 if i == n_phones - 1 else 0
            phone_relative_position = i / max(1, (n_phones - 1))
            word_length = n_phones  # Simplified, assuming all phonemes are in the same word
            
            # Context features
            prev_phone_class_V = 1 if i > 0 and phone_sequence[i-1].lower() in ['a', 'e', 'i', 'o', 'u'] else 0
            prev_phone_class_C = 1 if i > 0 and phone_sequence[i-1].lower() not in ['a', 'e', 'i', 'o', 'u', 'sil'] else 0
            prev_phone_class_sil = 1 if i > 0 and phone_sequence[i-1].lower() == 'sil' else 0
            
            next_phone_class_V = 1 if i < n_phones-1 and phone_sequence[i+1].lower() in ['a', 'e', 'i', 'o', 'u'] else 0
            next_phone_class_C = 1 if i < n_phones-1 and phone_sequence[i+1].lower() not in ['a', 'e', 'i', 'o', 'u', 'sil'] else 0
            next_phone_class_sil = 1 if i < n_phones-1 and phone_sequence[i+1].lower() == 'sil' else 0
            
            # Word position features in sentence - Since prediction does not have complete sentence information, use default values
            # Here we simplify by assuming all words are in the middle of the sentence
            word_position_in_sentence = 2  # Assuming it's the 2nd word in the sentence
            word_relative_position_in_sentence = 0.5  # Assuming middle of the sentence
            is_first_word_in_sentence = 0  # Not the first word
            is_last_word_in_sentence = 0   # Not the last word
            sentence_length = 3  # Assuming sentence has 3 words
            
            feature_vector = [
                phone_encoded, 
                is_vowel, is_consonant, is_silence,
                phone_position_first, phone_position_last,
                phone_relative_position, word_length,
                prev_phone_class_V, prev_phone_class_C, prev_phone_class_sil,
                next_phone_class_V, next_phone_class_C, next_phone_class_sil,
                # New sentence position features
                word_position_in_sentence, word_relative_position_in_sentence,
                is_first_word_in_sentence, is_last_word_in_sentence, sentence_length
            ]
            
            features.append(feature_vector)
        
        # Convert to numpy array and standardize
        X = self.scaler.transform(np.array(features))
        
        # Predict duration
        durations = self.model.predict(X)
        
        return durations
    
    def calculate_phone_stats(self, df: pd.DataFrame):
        """Calculate statistics (mean and variance) for each phoneme"""
        phone_groups = df.groupby('phone')['duration']
        
        for phone, group in phone_groups:
            durations = group.values
            self.phone_stats[phone] = {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'count': len(durations)
            }
    
    def compare_with_native(self, l2_df: pd.DataFrame, native_df: pd.DataFrame = None):
        """Compare duration between non-native speaker and native speaker"""
        if native_df is None:
            # Use local speaker statistics calculated during training
            native_stats = self.phone_stats
        else:
            # Temporary calculation of native speaker statistics
            native_stats = {}
            phone_groups = native_df.groupby('phone')['duration']
            for phone, group in phone_groups:
                durations = group.values
                native_stats[phone] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'count': len(durations)
                }
        
        # Calculate non-native speaker statistics
        l2_stats = {}
        phone_groups = l2_df.groupby('phone')['duration']
        for phone, group in phone_groups:
            durations = group.values
            l2_stats[phone] = {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'count': len(durations)
            }
        
        # Compare common phonemes
        common_phones = set(native_stats.keys()) & set(l2_stats.keys())
        comparison = {}
        
        for phone in common_phones:
            native_mean = native_stats[phone]['mean']
            native_std = native_stats[phone]['std']
            l2_mean = l2_stats[phone]['mean']
            l2_std = l2_stats[phone]['std']
            
            # Calculate z-score, representing the difference in average duration between L2 speaker and native speaker (in standard deviation units)
            if native_std > 0:
                z_score = (l2_mean - native_mean) / native_std
            else:
                z_score = 0
            
            comparison[phone] = {
                'native_mean': native_mean,
                'native_std': native_std,
                'l2_mean': l2_mean,
                'l2_std': l2_std,
                'z_score': z_score,
                'native_count': native_stats[phone]['count'],
                'l2_count': l2_stats[phone]['count']
            }
        
        return comparison
    
    def visualize_comparison(self, comparison: Dict, output_file: str = 'phone_comparison.png'):
        """Visualize comparison between native and non-native speaker"""
        # Prepare data
        phones = []
        native_means = []
        l2_means = []
        z_scores = []
        
        for phone, stats in comparison.items():
            if stats['native_count'] >= 5 and stats['l2_count'] >= 5:  # Only include phonemes with enough samples
                phones.append(phone)
                native_means.append(stats['native_mean'])
                l2_means.append(stats['l2_mean'])
                z_scores.append(stats['z_score'])
        
        # Sort by absolute z-score
        sorted_indices = np.argsort(np.abs(z_scores))[::-1]
        phones = [phones[i] for i in sorted_indices]
        native_means = [native_means[i] for i in sorted_indices]
        l2_means = [l2_means[i] for i in sorted_indices]
        z_scores = [z_scores[i] for i in sorted_indices]
        
        # Only display top 20 phonemes with largest differences
        n_display = min(20, len(phones))
        phones = phones[:n_display]
        native_means = native_means[:n_display]
        l2_means = l2_means[:n_display]
        z_scores = z_scores[:n_display]
        
        # Create chart
        plt.figure(figsize=(12, 8))
        
        # Draw bar chart, showing average duration between native and non-native speaker
        x = np.arange(len(phones))
        width = 0.35
        
        plt.bar(x - width/2, native_means, width, label='Native speaker')
        plt.bar(x + width/2, l2_means, width, label='Non-native speaker')
        
        plt.xlabel('Phoneme')
        plt.ylabel('Average duration (seconds)')
        plt.title('Comparison between native and non-native speaker')
        plt.xticks(x, phones, rotation=45)
        plt.legend()
        
        # Add second y-axis to display z-score
        ax2 = plt.twinx()
        ax2.plot(x, z_scores, 'r-', marker='o', label='Z score')
        ax2.set_ylabel('Z score (standard deviation units)')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Comparison visualization saved to {output_file}")
    
    def save(self, model_file: str):
        """Save model to file"""
        with open(model_file, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_file}")
    
    def load_preprocessed_data(self, preprocessed_data):
        """Load encoder and feature scaler from preprocessed data"""
        self.phone_encoder = preprocessed_data['phone_encoder']
        self.word_encoder = preprocessed_data['word_encoder']
        self.scaler = preprocessed_data['scaler']
        return self
    
    @classmethod
    def load(cls, model_file: str):
        """Load model from file"""
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from {model_file}")
        return model


if __name__ == "__main__":
    # Example: Load dataset and train model
    american_df = pd.read_csv('american_english_durations.csv')
    other_df = pd.read_csv('other_english_durations.csv')
    
    # Train model (using American English data)
    model = DurationModel()
    model.fit(american_df)
    
    # Save model
    model.save('duration_model.pkl')
    
    # Compare duration between non-native speaker and native speaker
    comparison = model.compare_with_native(other_df, american_df)
    
    # Visualize comparison results
    model.visualize_comparison(comparison, 'phone_duration_comparison.png')
    
    # Output some key differences
    sorted_comparison = sorted(comparison.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)
    print("\nPhonemes with largest differences (represented by Z score):")
    for i, (phone, stats) in enumerate(sorted_comparison[:10]):
        print(f"{i+1}. Phoneme: {phone}, Native mean: {stats['native_mean']:.3f} seconds, Non-native mean: {stats['l2_mean']:.3f} seconds, Z score: {stats['z_score']:.3f}") 