import os
import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PhoneDataset(Dataset):
    def __init__(self, features_df, phone_encoder, transform=None):
        self.features = features_df
        self.phone_encoder = phone_encoder
        self.transform = transform
        
        # Encode categorical features
        self.phone_encoded = torch.tensor(phone_encoder.transform(self.features['phone'].fillna('UNK')), dtype=torch.long)
        self.phone_class_encoded = torch.tensor(LabelEncoder().fit_transform(self.features['phone_class'].fillna('UNK')), dtype=torch.long)
        
        # Process boolean features
        self.is_first = torch.tensor(self.features['is_first_in_word'].astype(int).values, dtype=torch.float)
        self.is_last = torch.tensor(self.features['is_last_in_word'].astype(int).values, dtype=torch.float)
        
        # Numerical features
        self.word_length = torch.tensor(self.features['word_length'].values, dtype=torch.float)
        
        # Target variable
        self.durations = torch.tensor(self.features['phone_duration'].values, dtype=torch.float).view(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return features and target
        return {
            'phone': self.phone_encoded[idx],
            'phone_class': self.phone_class_encoded[idx],
            'is_first': self.is_first[idx],
            'is_last': self.is_last[idx],
            'word_length': self.word_length[idx]
        }, self.durations[idx]

# Define LSTM model
class PhoneDurationLSTM(nn.Module):
    def __init__(self, num_phones, num_classes, embedding_dim=32, hidden_dim=64, num_layers=2, dropout=0.2):
        super(PhoneDurationLSTM, self).__init__()
        
        # Feature embedding layers
        self.phone_embedding = nn.Embedding(num_phones, embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim // 2)
        
        # Total feature dimension
        feature_dim = embedding_dim + embedding_dim // 2 + 3  # phone, class, is_first, is_last, word_length
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features):
        # Get various features
        phone_emb = self.phone_embedding(features['phone'])
        class_emb = self.class_embedding(features['phone_class'])
        
        # Combine all features
        is_first = features['is_first'].unsqueeze(1)
        is_last = features['is_last'].unsqueeze(1)
        word_length = features['word_length'].unsqueeze(1)
        
        x = torch.cat([phone_emb, class_emb, is_first, is_last, word_length], dim=1)
        
        # Add sequence dimension (batch_size, 1, feature_dim)
        x = x.unsqueeze(1)
        
        # Through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Extract output from last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Through fully connected layers
        out = self.fc(lstm_out)
        
        return out

def prepare_data():
    """Prepare training and test data"""
    # Load feature data
    train_df = pd.read_csv('output/lstm/train_features.csv')
    test_df = pd.read_csv('output/lstm/test_features.csv')
    
    # Check if encoder already exists
    encoder_path = 'output/lstm/phone_encoder.pkl'
    if os.path.exists(encoder_path):
        print("Loading existing phone_encoder...")
        with open(encoder_path, 'rb') as f:
            phone_encoder = pickle.load(f)
    else:
        print("Creating new phone_encoder...")
        # Encode phone features
        phone_encoder = LabelEncoder()
        all_phones = pd.concat([train_df['phone'], test_df['phone']]).fillna('UNK').unique()
        phone_encoder.fit(all_phones)
        
        # Save encoder for later use
        with open(encoder_path, 'wb') as f:
            pickle.dump(phone_encoder, f)
    
    # Create datasets
    train_dataset = PhoneDataset(train_df, phone_encoder)
    test_dataset = PhoneDataset(test_df, phone_encoder)
    
    return train_dataset, test_dataset, phone_encoder

def train_model(train_loader, test_loader, model, num_epochs=10, learning_rate=0.001):
    """Train the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            for key in features:
                features[key] = features[key].to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in test_loader:
                # Move data to device
                for key in features:
                    features[key] = features[key].to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'output/lstm/phone_duration_model.pth')
    
    return model

def predict_phone_durations(model, sentence, phone_encoder):
    """Predict phoneme durations for a new sentence"""
    # Note: This is a simplified implementation, actual application needs to convert sentence to phoneme sequence
    # Here we assume the input is already decomposed into phonemes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Build example features - fix dimension issues
    # Process each phoneme as a separate sample
    phones = ['b', 'i', 'g']
    phone_classes = ['C', 'V', 'C']  # Consonant, Vowel, Consonant
    phone_encoded = torch.tensor(phone_encoder.transform(phones), dtype=torch.long).to(device)
    
    # Create correct class encoding for each phoneme
    class_encoder = LabelEncoder()
    class_encoder.fit(['C', 'V', 'sil'])
    class_encoded = torch.tensor(class_encoder.transform(phone_classes), dtype=torch.long).to(device)
    
    # Create position and length features
    is_first = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float).to(device)
    is_last = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).to(device)
    word_length = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float).to(device)
    
    # Build batch
    batch_size = len(phones)
    results = []
    
    # Process each phoneme individually
    for i in range(batch_size):
        features = {
            'phone': phone_encoded[i:i+1],
            'phone_class': class_encoded[i:i+1],
            'is_first': is_first[i:i+1],
            'is_last': is_last[i:i+1],
            'word_length': word_length[i:i+1]
        }
        
        with torch.no_grad():
            duration = model(features)
            results.append(duration.item())
    
    return np.array(results)

def train_lstm_model():
    # Prepare data
    train_dataset, test_dataset, phone_encoder = prepare_data()
    
    # No need to save phone_encoder again as it's handled in prepare_data
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model parameters
    num_phones = len(phone_encoder.classes_)
    num_classes = 3  # Assume 3 phoneme classes (C, V, sil)
    
    # Create model
    model = PhoneDurationLSTM(num_phones, num_classes)
    
    # Train model
    trained_model = train_model(train_loader, test_loader, model)
    
    # Example prediction
    sample_sentence = "big in"
    predicted_durations = predict_phone_durations(trained_model, sample_sentence, phone_encoder)
    print(f"Predicted phoneme durations: {predicted_durations}")

if __name__ == "__main__":
    train_lstm_model() 