import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_phone_features_from_json(result_json_path):
    """Extract phoneme features from a single result.json file"""
    with open(result_json_path, 'r') as f:
        data = json.load(f)
    
    features = []
    
    # Basic validation
    if 'result' not in data or not isinstance(data['result'], dict):
        raise ValueError("JSON data missing 'result' field or incorrect format")
    
    if 'text' not in data['result'] or not isinstance(data['result']['text'], str):
        raise ValueError("Result missing 'text' field or incorrect format")
    
    if 'segments' not in data['result'] or not isinstance(data['result']['segments'], list):
        raise ValueError("Result missing 'segments' field or incorrect format")
    
    # Get sentence level features
    text = data['result']['text']
    
    # Process words in segments
    for segment in data['result']['segments']:
        if not isinstance(segment, dict) or 'words' not in segment:
            continue  # Skip invalid segments
            
        if not isinstance(segment['words'], list):
            continue  # Skip non-list format words field
            
        for word in segment['words']:
            if not isinstance(word, dict):
                continue  # Skip non-dictionary format word
                
            # Validate required fields exist
            if 'phones' not in word or not isinstance(word['phones'], list):
                continue  # Skip words without phoneme list
                
            if 'word' not in word:
                continue  # Skip items without word field
                
            if word['word'] == "":  # Skip silence segments
                continue
                
            word_text = word['word'].strip()
            
            for i, phone in enumerate(word['phones']):
                if not isinstance(phone, dict):
                    continue  # Skip non-dictionary format phone
                    
                # Validate required fields exist
                required_fields = ['phone', 'duration', 'class']
                if not all(field in phone for field in required_fields):
                    continue  # Skip phonemes missing required fields
                
                # Extract current phoneme features
                phone_text = phone['phone']
                phone_duration = phone['duration']
                phone_class = phone['class']
                
                # Get context of previous and next phonemes (if they exist)
                prev_phone = None
                next_phone = None
                
                if i > 0 and i-1 < len(word['phones']) and isinstance(word['phones'][i-1], dict) and 'phone' in word['phones'][i-1]:
                    prev_phone = word['phones'][i-1]['phone']
                
                if i < len(word['phones']) - 1 and i+1 < len(word['phones']) and isinstance(word['phones'][i+1], dict) and 'phone' in word['phones'][i+1]:
                    next_phone = word['phones'][i+1]['phone']
                
                # Get phoneme position features in word
                is_first_in_word = (i == 0)
                is_last_in_word = (i == len(word['phones']) - 1)
                
                # Build feature dictionary
                feature = {
                    'sentence': text,
                    'word': word_text,
                    'phone': phone_text,
                    'phone_class': phone_class,
                    'phone_duration': phone_duration,
                    'prev_phone': prev_phone,
                    'next_phone': next_phone,
                    'is_first_in_word': is_first_in_word,
                    'is_last_in_word': is_last_in_word,
                    'word_length': len(word['phones'])
                }
                
                features.append(feature)
    
    return features

def extract_phone_features(dataset_dir):
    """Process all result.json files in the directory and return feature list"""
    all_features = []
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    
    for subdir in tqdm(subdirs, desc=f"Processing {os.path.basename(dataset_dir)} data"):
        result_path = os.path.join(dataset_dir, subdir, 'result.json')
        
        if os.path.exists(result_path):
            try:
                features = extract_phone_features_from_json(result_path)
                all_features.extend(features)
            except Exception as e:
                print(f"Error processing {result_path}: {str(e)}")
                # Continue processing next file
    
    return all_features

def main():
    # Create data directory (if it doesn't exist)
    os.makedirs('output', exist_ok=True)
    
    print("Extracting training set features...")
    train_features = extract_phone_features('american_english')
    train_df = pd.DataFrame(train_features)
    train_df.to_csv('output/train_features.csv', index=False)
    print(f"Training set features saved, total {len(train_df)} records")
    
    print("Extracting test set features...")
    test_features = extract_phone_features('other_english')
    test_df = pd.DataFrame(test_features)
    test_df.to_csv('output/test_features.csv', index=False)
    print(f"Test set features saved, total {len(test_df)} records")

if __name__ == "__main__":
    main() 