#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, List, Tuple

def load_json_file(file_path: str) -> Dict:
    """Load JSON file and return parsed dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_phone_durations_from_json(json_data: Dict) -> List[Dict]:
    """Extract phonemes and their durations from ASR time alignment data"""
    phone_durations = []
    
    # Check if result contains required data
    if 'result' not in json_data or 'segments' not in json_data['result']:
        return phone_durations
    
    for segment in json_data['result']['segments']:
        if 'words' not in segment:
            continue
            
        for word in segment['words']:
            if 'phones' not in word:
                continue
                
            word_text = word['word_normalized'].strip()
            for phone in word['phones']:
                phone_durations.append({
                    'task_id': json_data.get('task_id', ''),
                    'word': word_text,
                    'phone': phone['phone'],
                    'phone_class': phone.get('class', ''),
                    'duration': phone['duration'],
                    'start': phone['start']
                })
    
    return phone_durations

def extract_phone_durations(dataset_dir: str, output_file: str):
    """Process all time alignment files in the dataset directory and save results to CSV file"""
    all_durations = []
    
    # Get current working directory and project root directory
    current_dir = os.getcwd()
    
    # If current directory is src, move up one level to find dataset directory
    if os.path.basename(current_dir) == 'src':
        root_dir = os.path.dirname(current_dir)
        dataset_path = os.path.join(root_dir, dataset_dir)
    else:
        dataset_path = dataset_dir
    
    print(f"Dataset path: {dataset_path}")
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Directory {dataset_path} does not exist")
        return pd.DataFrame()
    
    # Get all subdirectories
    subdirs = [d for d in glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
    
    print(f"Processing time alignment files in {len(subdirs)} subdirectories...")
    
    for subdir in subdirs:
        json_files = glob(os.path.join(subdir, '*.json'))
        
        for json_file in json_files:
            try:
                data = load_json_file(json_file)
                durations = extract_phone_durations_from_json(data)
                all_durations.extend(durations)
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(all_durations)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_file, index=False)
    print(f"Extracted {len(all_durations)} phoneme duration data points, saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Process American English dataset
    american_df = extract_phone_durations(
        dataset_dir='american_english',
        output_file='american_english_durations.csv'
    )
    
    # Process non-native English dataset
    other_df = extract_phone_durations(
        dataset_dir='other_english',
        output_file='other_english_durations.csv'
    ) 