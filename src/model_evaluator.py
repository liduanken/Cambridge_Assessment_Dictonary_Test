import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Add src directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from lstm_model import PhoneDurationLSTM

def load_model_and_encoder():
    """Load trained model and encoder"""
    # Check if model file exists
    model_path = 'output/lstm/phone_duration_model.pth'
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Please run the training script first.")
    
    # Check if encoder exists
    encoder_path = 'output/lstm/phone_encoder.pkl'
    if not os.path.exists(encoder_path):
        print("Error: Phone encoder not found. Please run the training script first.")
    
    # Load phone encoder
    with open(encoder_path, 'rb') as f:
        phone_encoder = pickle.load(f)
    
    # Create model
    num_phones = len(phone_encoder.classes_)
    num_classes = 3  # Assume 3 phoneme classes (C, V, sil)
    model = PhoneDurationLSTM(num_phones, num_classes)
    
    # Load pretrained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, phone_encoder

def evaluate_test_set():
    """Evaluate test set and save results"""
    print("Loading model...")
    model, phone_encoder = load_model_and_encoder()
    
    print("Loading test set data...")
    test_df = pd.read_csv('output/lstm/test_features.csv')
    
    print(f"Test set samples: {len(test_df)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Group by sentence and word while maintaining original order for batch processing
    grouped = test_df.groupby(['sentence', 'word'])
    
    predictions = []
    true_durations = []
    
    print("Predicting test set...")
    progress_bar = tqdm(total=len(test_df))
    
    # Create phoneme category encoder
    class_encoder = LabelEncoder()
    class_encoder.fit(['C', 'V', 'sil'])
    
    for (sentence, word), group in grouped:
        # Process all phonemes in each word
        phones = group['phone'].tolist()
        phone_classes = group['phone_class'].tolist()
        
        # Encode features
        phone_encoded = torch.tensor(phone_encoder.transform(phones), dtype=torch.long).to(device)
        class_encoded = torch.tensor(class_encoder.transform(phone_classes), dtype=torch.long).to(device)
        
        # Other features
        is_first = torch.tensor(group['is_first_in_word'].astype(float).values, dtype=torch.float).to(device)
        is_last = torch.tensor(group['is_last_in_word'].astype(float).values, dtype=torch.float).to(device)
        word_length = torch.tensor(group['word_length'].astype(float).values, dtype=torch.float).to(device)
        
        # Collect true durations
        true_durations.extend(group['phone_duration'].tolist())
        
        # Predict duration for each phoneme individually
        for i in range(len(phones)):
            features = {
                'phone': phone_encoded[i:i+1],
                'phone_class': class_encoded[i:i+1],
                'is_first': is_first[i:i+1],
                'is_last': is_last[i:i+1],
                'word_length': word_length[i:i+1]
            }
            
            with torch.no_grad():
                duration = model(features)
                predictions.append(duration.item())
                
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate evaluation metrics
    mse = mean_squared_error(true_durations, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_durations, predictions)
    
    print("\nTest set evaluation results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Add predictions to test set data
    test_df['predicted_duration'] = predictions
    test_df['duration_error'] = test_df['phone_duration'] - test_df['predicted_duration']
    
    # Save prediction results
    output_path = 'output/lstm/test_predictions.csv'
    test_df.to_csv(output_path, index=False)
    print(f"\nPrediction results saved to: {output_path}")
    
    # Calculate average error for each phoneme type
    phone_class_errors = test_df.groupby('phone_class')['duration_error'].agg(['mean', 'std']).reset_index()
    print("\nError statistics by phoneme class:")
    print(phone_class_errors)
    
    # Visualize prediction results (requires matplotlib and seaborn)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory
        os.makedirs('output/lstm', exist_ok=True)
        
        # Scatter plot of true vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(test_df['phone_duration'], test_df['predicted_duration'], alpha=0.3)
        plt.plot([0, max(test_df['phone_duration'])], [0, max(test_df['phone_duration'])], 'r--')
        plt.xlabel('True Duration (seconds)')
        plt.ylabel('Predicted Duration (seconds)')
        plt.title('True Duration vs Predicted Duration')
        plt.savefig('output/lstm/true_vs_pred.png')
        
        # Error histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(test_df['duration_error'], kde=True)
        plt.xlabel('Prediction Error (seconds)')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.savefig('output/lstm/error_distribution.png')
        
        # Box plot of errors by phoneme class
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='phone_class', y='duration_error', data=test_df)
        plt.xlabel('Phoneme Class')
        plt.ylabel('Prediction Error (seconds)')
        plt.title('Prediction Error by Phoneme Class')
        plt.savefig('output/lstm/error_by_phone_class.png')
        
        print(f"\nVisualization results saved to: output/lstm/")
    except ImportError:
        print("\nNote: matplotlib or seaborn not installed, skipping visualization steps.")

def evaluate_lstm_model():
    # Ensure output directory exists
    os.makedirs('output/lstm', exist_ok=True)
    
    try:
        evaluate_test_set()
        print("\nTest set evaluation completed!")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_lstm_model() 