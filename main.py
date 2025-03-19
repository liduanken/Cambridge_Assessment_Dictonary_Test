#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import modules
from src.phone_duration_extractor import extract_phone_durations
from src.tree_duration_model import DurationModel
from src.phone_feature_extractor import extract_phone_features
from src.lstm_model import train_lstm_model
from src.model_evaluator import evaluate_lstm_model

def run_tree_model():
    """Run decision tree-based phoneme duration modeling"""
    print("Running decision tree-based phoneme duration modeling...")
    # Create output directory
    output_dir = os.path.join('output', 'tree')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data extraction
    print("Extracting phoneme durations from ASR results...")
    american_csv = os.path.join(output_dir, 'american_english_durations.csv')
    other_csv = os.path.join(output_dir, 'other_english_durations.csv')
    
    american_df = extract_phone_durations('american_english', american_csv)
    other_df = extract_phone_durations('other_english', other_csv)
    
    # Step 2: Model training and preprocessed data saving
    print("Training phoneme duration prediction model...")
    
    # Create preprocessed data file paths
    preprocessed_data_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    model_path = os.path.join(output_dir, 'duration_model.pkl')
    
    # Check if preprocessed data and model exist
    if os.path.exists(preprocessed_data_path) and os.path.exists(model_path):
        print(f"Found existing preprocessed data file: {preprocessed_data_path} and model file: {model_path}, loading directly...")
        # Load preprocessed data from pkl file
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
            
        # Create new model instance and load preprocessed data
        model = DurationModel()
        model.load_preprocessed_data(preprocessed_data)
        
        # Load trained model
        trained_model = DurationModel.load(model_path)
        
        # Copy trained model parameters to current model
        model.model = trained_model.model
        model.phone_stats = trained_model.phone_stats
    else:
        print("Preprocessing data and training model...")
        # Initialize model
        model = DurationModel()
        # Preprocess data
        X, y = model.preprocess_data(american_df)
        
        # Save preprocessed data
        preprocessed_data = {
            'X': X,
            'y': y,
            'phone_encoder': model.phone_encoder,
            'word_encoder': model.word_encoder,
            'scaler': model.scaler
        }
        
        with open(preprocessed_data_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        print(f"Preprocessed data saved to {preprocessed_data_path}")
        
        # Split training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.model.fit(X_train, y_train)
        
        # Evaluate model on validation set
        y_pred = model.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Model evaluation results:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R^2): {r2:.4f}")
        
        # Calculate statistics for each phoneme
        model.calculate_phone_stats(american_df)
        
        # Save model
        model.save(model_path)
    
    # Step 3: Evaluate differences between non-native and native speakers
    print("Evaluating phoneme duration differences between non-native and native speakers...")
    comparison_file = os.path.join(output_dir, 'phone_duration_comparison.png')
    comparison = model.compare_with_native(other_df, american_df)
    model.visualize_comparison(comparison, comparison_file)
    
    # Output phonemes with largest differences
    sorted_comparison = sorted(comparison.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)
    print("\nPhonemes with largest differences (in Z-scores):")
    for i, (phone, stats) in enumerate(sorted_comparison[:10]):
        print(f"{i+1}. Phoneme: {phone}, Native mean: {stats['native_mean']:.3f}s, Non-native mean: {stats['l2_mean']:.3f}s, Z-score: {stats['z_score']:.3f}")
    
    
    # Statistics and visualization by phoneme category
    american_vowels = american_df[american_df['phone_class'] == 'V']['duration']
    american_consonants = american_df[american_df['phone_class'] == 'C']['duration']
    american_silence = american_df[american_df['phone_class'] == 'sil']['duration']
    
    other_vowels = other_df[other_df['phone_class'] == 'V']['duration']
    other_consonants = other_df[other_df['phone_class'] == 'C']['duration']
    other_silence = other_df[other_df['phone_class'] == 'sil']['duration']
    
    print("\nDifference statistics by phoneme category:")
    print(f"Vowels - Native mean: {american_vowels.mean():.3f}s, Non-native mean: {other_vowels.mean():.3f}s, Difference: {(other_vowels.mean() - american_vowels.mean()):.3f}s")
    print(f"Consonants - Native mean: {american_consonants.mean():.3f}s, Non-native mean: {other_consonants.mean():.3f}s, Difference: {(other_consonants.mean() - american_consonants.mean()):.3f}s")
    print(f"Silence - Native mean: {american_silence.mean():.3f}s, Non-native mean: {other_silence.mean():.3f}s, Difference: {(other_silence.mean() - american_silence.mean()):.3f}s")
    
    # Visualize differences by phoneme category
    categories = ['Vowels (V)', 'Consonants (C)', 'Silence (sil)']
    native_means = [american_vowels.mean(), american_consonants.mean(), american_silence.mean()]
    l2_means = [other_vowels.mean(), other_consonants.mean(), other_silence.mean()]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, native_means, width, label='Native speakers')
    plt.bar(x + width/2, l2_means, width, label='Non-native speakers')
    
    plt.xlabel('Phoneme Category')
    plt.ylabel('Average Duration (seconds)')
    plt.title('Native vs Non-native Speaker Duration Comparison by Phoneme Category')
    plt.xticks(x, categories)
    plt.legend()
    
    category_plot_file = os.path.join(output_dir, 'category_comparison.png')
    plt.tight_layout()
    plt.savefig(category_plot_file)
    plt.close()
    print(f"Phoneme category comparison plot saved to: {category_plot_file}")
    print("Decision tree-based phoneme duration modeling completed!")

def run_lstm_model():
    """Run LSTM-based phoneme duration modeling"""
    print("Running LSTM-based phoneme duration modeling...")
    # Create output directory
    output_dir = os.path.join('output', 'lstm')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Feature extraction
    print("Performing feature extraction...")
    train_features_csv = os.path.join(output_dir, 'train_features.csv')
    test_features_csv = os.path.join(output_dir, 'test_features.csv')
    
    # Check if training set feature file exists
    if os.path.exists(train_features_csv):
        print(f"Found existing training set feature file: {train_features_csv}, loading directly...")
        train_df = pd.read_csv(train_features_csv)
    else:
        print("Extracting training set features...")
        train_features = extract_phone_features('american_english')
        train_df = pd.DataFrame(train_features)
        train_df.to_csv(train_features_csv, index=False)
    
    print(f"Training set feature data, total {len(train_df)} records")
    
    # Check if test set feature file exists
    if os.path.exists(test_features_csv):
        print(f"Found existing test set feature file: {test_features_csv}, loading directly...")
        test_df = pd.read_csv(test_features_csv)
    else:
        print("Extracting test set features...")
        test_features = extract_phone_features('other_english')
        test_df = pd.DataFrame(test_features)
        test_df.to_csv(test_features_csv, index=False)
    
    print(f"Test set feature data, total {len(test_df)} records")
    
    # Step 2: Model training
    print("Executing model training...")
    train_lstm_model()
    
    # Step 3: Model evaluation
    print("Evaluating model performance on test set...")
    evaluate_lstm_model()
    
    print("LSTM-based phoneme duration modeling completed!")

def main():
    parser = argparse.ArgumentParser(description='Phoneme Duration Modeling System')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model for phoneme duration prediction')
    parser.add_argument('--tree', action='store_true', help='Use decision tree model for phoneme duration prediction')
    
    args = parser.parse_args()
    
    if args.lstm:
        run_lstm_model()
    elif args.tree:
        run_tree_model()
    else:
        parser.print_help()
        print("\nPlease select a model type: --lstm or --tree")

if __name__ == "__main__":
    main() 