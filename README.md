# Phoneme Duration Prediction System

This project implements an English phoneme duration prediction system based on LSTM and Decision Tree models. The system can analyze speech recognition results, extract phoneme features, and predict the duration of each phoneme. The system provides two different model implementation approaches that users can choose from based on their needs.

## Project Structure

```
.
├── american_english/  # Training dataset directory
├── other_english/     # Test dataset directory
├── output/            # Output directory
│   ├── lstm/         # LSTM model output (feature CSV, models, etc.)
│   └── tree/         # Decision Tree model output
└── src/               # Source code
    ├── extract_features.py     # Feature extraction script
    ├── lstm_model.py          # LSTM model definition and training
    ├── tree_duration_model.py # Decision Tree model definition and training
    ├── predict.py            # Prediction script
    ├── check_data.py         # Data format checking script
    ├── evaluate.py           # Test set evaluation script
── main.py              # Main entry script
```

## Features

1. **Feature Extraction**: Extract phoneme features from result.json files, including phoneme type, position, context, etc.
2. **Model Training**:
   - LSTM Model: Uses LSTM neural networks to learn phoneme duration patterns
   - Decision Tree Model: Uses decision tree algorithms to learn phoneme duration patterns
3. **Duration Prediction**: Predicts duration for each phoneme in new sentences
4. **Data Checking**: Verifies JSON file format in the dataset to ensure expected structure
5. **Model Evaluation**: Evaluates model performance on test set and generates visualization reports
6. **Native vs Non-native Speaker Analysis**: Analyzes and visualizes differences in phoneme duration between native and non-native speakers

## Dependencies

This project depends on the following Python libraries:

1. Extract the datasets
   ```bash
   # Extract training and testing datasets to the project root directory
   unzip american_english.zip
   unzip other_english.zip
   ```

2. Initialize environment
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. install package
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

### 1. Model Training and Evaluation

Choose between LSTM model or Decision Tree model:

```bash
# Use LSTM model
uv run python main.py --lstm

# Use Decision Tree model
uv run python main.py --tree
```

This will generate the following outputs:
- In `output/lstm/` or `output/tree/` directory:

## Model Details

### LSTM Model
- Uses bidirectional LSTM to capture context dependencies
- Phoneme embedding representation
- Considers phoneme position in words
- Uses phoneme categories as features

### Decision Tree Model
- Decision tree classification based on phoneme features
- Considers phoneme context information
- Supports feature importance analysis
- Provides detailed native vs non-native speaker comparison:
  - Statistical analysis by phoneme category
  - Z-score evaluation of phoneme differences
  - Visualization of phoneme duration distribution

## Details
 please refer to the [report](https://github.com/liduanken/Cambridge_Assessment_Dictonary_Test/blob/main/Report.md) section in this directory

## Notes
- Dataset may contain files with inconsistent formats; error handling mechanisms are in place
- Decision Tree model provides more intuitive feature importance analysis but may be less effective at complex pattern recognition compared to LSTM model

## Future Extensions

- Use more sophisticated text-to-phoneme conversion tools
- Add more features such as syllable structure, stress, etc.
- Try ensemble learning methods (Random Forest, XGBoost, etc.)
- Develop hybrid models combining LSTM and Decision Tree strengths
- Add intonation and prosody prediction
- Improve data cleaning and preprocessing workflow 
