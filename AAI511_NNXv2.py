#!/usr/bin/env python
# coding: utf-8

# # AAI-511 Final Assignment - Musician Prediction using MIDI

# Name: `Sunil Prasath`

# **Note:** This project took several iterations of dataset processing, model architecture optimizations and parameter fine-tuning to get to this level of performance accross all the artists.

# - This project was built using the assistance by AI Agents that harnessed the capabilites of few SOTA Large Language Models for rapid prototyping, performance analysis and optimizations with my instructions.
# - LLM Models used for the development: Anthropic's Opus 4.1 (Closed Source, Claude) for coding assistance, Locally managed Llama 3.3 and 4 for performance analysis and improvement. 

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pretty_midi
from IPython.display import Audio, display, HTML
from collections import defaultdict, Counter
import random
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')


# In[2]:


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


# In[3]:


class SimpleMIDIProcessor:    
    def __init__(self, velocity_threshold=10):  # Very low threshold to capture soft notes
        self.velocity_threshold = velocity_threshold
        
    def process_midi(self, midi_path, max_seq_len=1000):
        # Extract simple but comprehensive features from MIDI
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Collect all notes (including very soft ones)
            all_notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Include even very soft notes
                        if note.velocity >= self.velocity_threshold:
                            all_notes.append({
                                'pitch': note.pitch,
                                'velocity': note.velocity,
                                'start': note.start,
                                'duration': note.end - note.start
                            })
            
            if len(all_notes) == 0:
                # Return empty features if no notes
                return self._get_empty_features(max_seq_len)
            
            # Sort by start time
            all_notes.sort(key=lambda x: x['start'])
            
            # Create piano roll representation (more robust than sequences)
            piano_roll = self._create_piano_roll(midi_data)
            
            # Extract simple statistical features
            stats_features = self._extract_statistics(all_notes, midi_data)
            
            # Create note sequence (simplified)
            note_sequence = self._create_sequence(all_notes, max_seq_len)
            
            return {
                'piano_roll': piano_roll,
                'stats': stats_features,
                'sequence': note_sequence,
                'num_notes': len(all_notes)
            }
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return self._get_empty_features(max_seq_len)
    
    def _create_piano_roll(self, midi_data, fs=10):
        # Create a simplified piano roll representation
        try:
            piano_roll = midi_data.get_piano_roll(fs=fs)
            
            # Reduce to 88 piano keys (A0 to C8)
            piano_start = 21
            piano_end = 109
            piano_roll = piano_roll[piano_start:piano_end, :]
            
            # Normalize velocities (but preserve relative differences)
            if piano_roll.max() > 0:
                piano_roll = piano_roll / 127.0  # Keep in 0-1 range
            
            # Reduce time dimension if too long
            max_time_steps = 500
            if piano_roll.shape[1] > max_time_steps:
                # Downsample time
                indices = np.linspace(0, piano_roll.shape[1]-1, max_time_steps, dtype=int)
                piano_roll = piano_roll[:, indices]
            elif piano_roll.shape[1] < max_time_steps:
                # Pad with zeros
                padding = max_time_steps - piano_roll.shape[1]
                piano_roll = np.pad(piano_roll, ((0, 0), (0, padding)), mode='constant')
            
            return piano_roll.astype(np.float32)
            
        except:
            return np.zeros((88, 500), dtype=np.float32)
    
    def _extract_statistics(self, notes, midi_data):
        # Extract simple statistical features - exactly 30 features
        if len(notes) == 0:
            return np.zeros(30, dtype=np.float32)
        
        features = []
        
        # Pitch features (7 features)
        pitches = [n['pitch'] for n in notes]
        features.append(np.mean(pitches))
        features.append(np.std(pitches) if len(pitches) > 1 else 0)
        features.append(np.min(pitches))
        features.append(np.max(pitches))
        features.append(np.median(pitches))
        features.append(np.percentile(pitches, 25))
        features.append(np.percentile(pitches, 75))
        
        # Velocity features (7 features)
        velocities = [n['velocity'] for n in notes]
        features.append(np.mean(velocities))
        features.append(np.std(velocities) if len(velocities) > 1 else 0)
        features.append(np.min(velocities))
        features.append(np.max(velocities))
        features.append(np.median(velocities))
        features.append(np.percentile(velocities, 25))
        features.append(np.percentile(velocities, 75))
        
        # Duration features (5 features)
        durations = [n['duration'] for n in notes]
        features.append(np.mean(durations))
        features.append(np.std(durations) if len(durations) > 1 else 0)
        features.append(np.min(durations))
        features.append(np.max(durations))
        features.append(np.median(durations))
        
        # Interval features (4 features)
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            features.append(np.mean(np.abs(intervals)))
            features.append(np.std(intervals))
            features.append(np.sum(intervals > 0) / len(intervals))  # Ascending ratio
            features.append(np.sum(intervals < 0) / len(intervals))  # Descending ratio
        else:
            features.extend([0, 0, 0, 0])
        
        # Global features (5 features)
        total_duration = midi_data.get_end_time()
        features.append(len(notes))  # Total notes
        features.append(total_duration)  # Duration
        features.append(len(notes) / (total_duration + 1e-6))  # Note density
        features.append(len(midi_data.instruments))  # Number of instruments
        features.append(len(set(pitches)))  # Number of unique pitches
        
        # Pitch class features (2 features)
        pitch_classes = [p % 12 for p in pitches]
        pitch_class_counts = Counter(pitch_classes)
        if len(pitch_class_counts) > 0:
            most_common = pitch_class_counts.most_common(2)
            features.append(most_common[0][1] / len(pitches))  # Most common pitch class ratio
            if len(most_common) > 1:
                features.append(most_common[1][1] / len(pitches))  # Second most common
            else:
                features.append(0)
        else:
            features.extend([0, 0])
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Ensure exactly 30 features
        assert len(features) == 30, f"Expected 30 features, got {len(features)}"
        
        return features
        # Velocity features (important for style)
        velocities = [n['velocity'] for n in notes]
        features.extend([
            np.mean(velocities),
            np.std(velocities),
            np.min(velocities),
            np.max(velocities),
            np.median(velocities),
            np.percentile(velocities, 25),
            np.percentile(velocities, 75)
        ])
        
        # Duration features
        durations = [n['duration'] for n in notes]
        features.extend([
            np.mean(durations),
            np.std(durations),
            np.min(durations),
            np.max(durations),
            np.median(durations)
        ])
        
        # Interval features
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            features.extend([
                np.mean(np.abs(intervals)),
                np.std(intervals),
                np.sum(intervals > 0) / len(intervals),  # Ascending ratio
                np.sum(intervals < 0) / len(intervals),  # Descending ratio
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Global features
        features.extend([
            len(notes),  # Total notes
            midi_data.get_end_time(),  # Duration
            len(notes) / (midi_data.get_end_time() + 1e-6),  # Note density
            len(midi_data.instruments),  # Number of instruments
        ])
        
        # Pitch class histogram (12 semitones)
        pitch_classes = np.array([p % 12 for p in pitches])
        pitch_hist = np.histogram(pitch_classes, bins=12, range=(0, 12))[0]
        pitch_hist = pitch_hist / (np.sum(pitch_hist) + 1e-6)
        
        # Add top 2 most common pitch classes
        top_2_indices = np.argsort(pitch_hist)[-2:]
        features.extend([pitch_hist[top_2_indices[0]], pitch_hist[top_2_indices[1]]])
        
        return np.array(features[:30], dtype=np.float32)  # Ensure exactly 30 features
    
    def _create_sequence(self, notes, max_len):
        # Create simple note sequence
        if len(notes) == 0:
            return np.zeros((max_len, 4), dtype=np.float32)
        
        sequence = []
        for note in notes[:max_len]:
            sequence.append([
                note['pitch'] / 127.0,
                note['velocity'] / 127.0,
                np.tanh(note['duration']),  # Compress long durations
                note['pitch'] % 12 / 12.0  # Pitch class
            ])
        
        sequence = np.array(sequence)
        
        # Pad if needed
        if len(sequence) < max_len:
            padding = np.zeros((max_len - len(sequence), 4))
            sequence = np.vstack([sequence, padding])
        
        return sequence.astype(np.float32)
    
    def _get_empty_features(self, max_seq_len):
        # Return empty features
        return {
            'piano_roll': np.zeros((88, 500), dtype=np.float32),
            'stats': np.zeros(30, dtype=np.float32),
            'sequence': np.zeros((max_seq_len, 4), dtype=np.float32),
            'num_notes': 0
        }


# In[4]:


class RobustMIDIDataset(Dataset):
    # Robust dataset with aggressive augmentation
    
    def __init__(self, data_dir, split='train', augment=True):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.processor = SimpleMIDIProcessor(velocity_threshold=10)
        self.label_encoder = LabelEncoder()
        
        self.samples = []
        self.labels = []
        self.file_paths = []
        
        # Cache directory
        self.cache_dir = f"cache_{split}_simple"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_data()
        
        if self.augment:
            self._balance_dataset()
    
    def _load_data(self):
        # Load all data ensuring every composer is captured
        split_dir = os.path.join(self.data_dir, self.split)
        
        print(f"\nLoading {self.split} data...")
        composers = sorted([d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))])
        
        print(f"Found {len(composers)} composers: {composers}")
        
        composer_samples = defaultdict(list)
        
        for composer in composers:
            composer_dir = os.path.join(split_dir, composer)
            midi_files = [f for f in os.listdir(composer_dir) 
                         if f.lower().endswith(('.mid', '.midi'))]
            
            print(f"  {composer}: {len(midi_files)} files found")
            
            for midi_file in midi_files:
                midi_path = os.path.join(composer_dir, midi_file)
                
                # Try cache first
                cache_file = os.path.join(self.cache_dir, 
                                         f"{composer}_{midi_file}.pkl")
                
                features = None
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            features = pickle.load(f)
                            # Validate cached features
                            if features and 'stats' in features and len(features['stats']) != 30:
                                print(f"    Warning: Cached {midi_file} has {len(features['stats'])} stat features, reprocessing...")
                                features = None
                    except:
                        pass
                
                if features is None:
                    features = self.processor.process_midi(midi_path)
                    
                    # Validate features
                    if features and 'stats' in features:
                        if len(features['stats']) != 30:
                            print(f"    ERROR: {midi_file} generated {len(features['stats'])} stat features instead of 30!")
                            # Fix it here
                            if len(features['stats']) < 30:
                                features['stats'] = np.pad(features['stats'], (0, 30 - len(features['stats'])), 'constant')
                            else:
                                features['stats'] = features['stats'][:30]
                    
                    # Save to cache
                    if features and features['num_notes'] > 0:
                        try:
                            with open(cache_file, 'wb') as f:
                                pickle.dump(features, f)
                        except:
                            pass
                
                if features and features['num_notes'] > 0:  # Only keep files with notes
                    # Final validation
                    if len(features['stats']) != 30:
                        print(f"    Final check failed for {midi_file}: {len(features['stats'])} stats")
                        features['stats'] = np.pad(features['stats'][:30], (0, max(0, 30 - len(features['stats'][:30]))), 'constant')
                    
                    composer_samples[composer].append((features, midi_path))
        
        # Convert to lists
        for composer in composers:
            samples = composer_samples[composer]
            if len(samples) > 0:
                for features, path in samples:
                    self.samples.append(features)
                    self.labels.append(composer)
                    self.file_paths.append(path)
                print(f"  {composer}: {len(samples)} valid samples loaded")
            else:
                print(f"  WARNING: {composer} has no valid samples!")
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Final validation of all samples
        for i, sample in enumerate(self.samples):
            if len(sample['stats']) != 30:
                print(f"Sample {i} has {len(sample['stats'])} stats, fixing...")
                if len(sample['stats']) < 30:
                    self.samples[i]['stats'] = np.pad(sample['stats'], (0, 30 - len(sample['stats'])), 'constant')
                else:
                    self.samples[i]['stats'] = sample['stats'][:30]
    
    def _balance_dataset(self):
        # Aggressively balance the dataset
        class_counts = Counter(self.labels)
        max_count = max(class_counts.values())
        
        print("\nBalancing dataset...")
        
        new_samples = []
        new_labels = []
        new_paths = []
        
        for class_id in range(self.num_classes):
            class_indices = [i for i, l in enumerate(self.labels) if l == class_id]
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            
            current_count = len(class_indices)
            target_count = max_count
            
            # Add original samples
            for idx in class_indices:
                new_samples.append(self.samples[idx])
                new_labels.append(class_id)
                new_paths.append(self.file_paths[idx])
            
            # Augment to reach target
            if current_count < target_count:
                print(f"  Augmenting {class_name}: {current_count} -> {target_count}")
                
                for _ in range(target_count - current_count):
                    # Random sample from this class
                    base_idx = random.choice(class_indices)
                    aug_sample = self._augment_sample(self.samples[base_idx])
                    
                    new_samples.append(aug_sample)
                    new_labels.append(class_id)
                    new_paths.append(self.file_paths[base_idx] + "_aug")
        
        self.samples = new_samples
        self.labels = np.array(new_labels)
        self.file_paths = new_paths
        
        print(f"Balanced dataset size: {len(self.samples)}")
    
    def _augment_sample(self, sample):
        # Simple but effective augmentation
        aug_sample = {}
        
        # Copy original
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                aug_sample[key] = sample[key].copy()
            else:
                aug_sample[key] = sample[key]
        
        # Ensure stats is 30 dimensions before augmentation
        if 'stats' in aug_sample and len(aug_sample['stats']) != 30:
            if len(aug_sample['stats']) < 30:
                aug_sample['stats'] = np.pad(aug_sample['stats'], (0, 30 - len(aug_sample['stats'])), 'constant')
            else:
                aug_sample['stats'] = aug_sample['stats'][:30]
        
        # Random augmentations
        if random.random() < 0.5:
            # Pitch shift
            shift = random.uniform(-0.05, 0.05)
            aug_sample['sequence'][:, 0] = np.clip(aug_sample['sequence'][:, 0] + shift, 0, 1)
            # Only modify first 7 pitch stats
            aug_sample['stats'][0:7] = np.clip(aug_sample['stats'][0:7] + shift * 127, 0, 127)
        
        if random.random() < 0.5:
            # Velocity scaling
            scale = random.uniform(0.8, 1.2)
            aug_sample['sequence'][:, 1] *= scale
            aug_sample['sequence'][:, 1] = np.clip(aug_sample['sequence'][:, 1], 0, 1)
            # Only modify velocity stats (indices 7-13)
            aug_sample['stats'][7:14] = np.clip(aug_sample['stats'][7:14] * scale, 0, 127)
        
        if random.random() < 0.3:
            # Add noise
            noise = np.random.normal(0, 0.01, aug_sample['sequence'].shape)
            aug_sample['sequence'] = np.clip(aug_sample['sequence'] + noise, 0, 1)
        
        # Final check to ensure stats is still 30 dimensions
        if len(aug_sample['stats']) != 30:
            print(f"Warning: Augmented stats has {len(aug_sample['stats'])} dimensions")
            if len(aug_sample['stats']) < 30:
                aug_sample['stats'] = np.pad(aug_sample['stats'], (0, 30 - len(aug_sample['stats'])), 'constant')
            else:
                aug_sample['stats'] = aug_sample['stats'][:30]
        
        return aug_sample
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Light augmentation during training
        if self.augment and random.random() < 0.2:
            sample = self._augment_sample(sample)
        
        # Validate dimensions before returning
        piano_roll = sample['piano_roll']
        stats = sample['stats']
        sequence = sample['sequence']
        
        # Ensure stats has exactly 30 features
        if len(stats) != 30:
            print(f"Warning: stats has {len(stats)} features instead of 30, padding/truncating...")
            if len(stats) < 30:
                stats = np.pad(stats, (0, 30 - len(stats)), 'constant')
            else:
                stats = stats[:30]
        
        return {
            'piano_roll': torch.FloatTensor(piano_roll),
            'stats': torch.FloatTensor(stats),
            'sequence': torch.FloatTensor(sequence),
            'label': self.labels[idx]
        }


# In[5]:


class EnhancedComposerModel(nn.Module):
    # Enhanced model with multiple pathways and strong regularization
    
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        
        # Piano roll pathway (CNN)
        self.piano_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Sequence pathway (1D CNN + LSTM)
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.3)
        )
        
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                           dropout=dropout * 0.5, bidirectional=True)
        
        # Statistics pathway (MLP)
        self.stats_mlp = nn.Sequential(
            nn.Linear(30, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Attention mechanism for sequence
        self.attention = nn.MultiheadAttention(512, num_heads=16, dropout=dropout * 0.3)
        
        # Feature fusion and classification
        piano_features = 128 * 4 * 4  # 2048
        lstm_features = 512
        stats_features = 128
        total_features = piano_features + lstm_features + stats_features
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Label smoothing
        self.label_smoothing = 0.1
    
    def forward(self, piano_roll, stats, sequence):
        batch_size = piano_roll.size(0)
        
        # Piano roll pathway
        piano_features = self.piano_cnn(piano_roll.unsqueeze(1))
        piano_features = piano_features.view(batch_size, -1)
        
        # Sequence pathway
        seq_transposed = sequence.transpose(1, 2)
        seq_cnn_out = self.seq_cnn(seq_transposed)
        seq_cnn_out = seq_cnn_out.transpose(1, 2)
        
        lstm_out, _ = self.lstm(seq_cnn_out)
        
        # Apply attention
        lstm_out_t = lstm_out.transpose(0, 1)
        attended, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attended = attended.transpose(0, 1)
        
        # Global pooling
        lstm_features = torch.mean(attended, dim=1)
        
        # Statistics pathway
        stats_features = self.stats_mlp(stats)
        
        # Combine all features
        combined = torch.cat([piano_features, lstm_features, stats_features], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output


# In[6]:


class ImprovedTrainer:
    # Trainer with proper loss handling and model saving
    
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Metrics
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move to device
            piano_roll = batch['piano_roll'].to(self.device)
            stats = batch['stats'].to(self.device)
            sequence = batch['sequence'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Debug check
            if stats.shape[1] != 30:
                print(f"ERROR: Batch {batch_idx} has stats shape {stats.shape}, expected (batch_size, 30)")
                print(f"Skipping this batch...")
                continue
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.model(piano_roll, stats, sequence)
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Stats shape: {stats.shape}")
                print(f"Piano roll shape: {piano_roll.shape}")
                print(f"Sequence shape: {sequence.shape}")
                continue
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Check for NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, label_encoder):
        self.model.eval()
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                piano_roll = batch['piano_roll'].to(self.device)
                stats = batch['stats'].to(self.device)
                sequence = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(piano_roll, stats, sequence)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        # Calculate metrics
        overall_acc = 100 * correct / total if total > 0 else 0
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        class_accs = []
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
            else:
                acc = 0
            class_accs.append(acc)
            composer = label_encoder.inverse_transform([i])[0]
            print(f"  {composer}: {acc:.1f}% ({class_total[i]} samples)")
        
        return overall_acc, class_accs, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, label_encoder, epochs=50):
        best_acc = 0
        patience_counter = 0
        max_patience = 25
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation
            val_acc, class_accs, _, _ = self.validate(val_loader, label_encoder)
            
            # Calculate balanced accuracy
            min_class_acc = min(class_accs) if class_accs else 0
            balanced_acc = (val_acc + min_class_acc) / 2
            
            print(f"Val Acc: {val_acc:.2f}%, Min Class: {min_class_acc:.2f}%, Balanced: {balanced_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(balanced_acc)
            
            # Save best model (with weights_only=False fix)
            if balanced_acc > best_acc:
                best_acc = balanced_acc
                patience_counter = 0
                
                # Save with weights_only=False compatible format
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'class_accs': class_accs
                }, 'best_model.pth')
                
                print(f"✓ New best model saved! (Balanced Acc: {best_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
        
        return best_acc


# In[7]:


def run_simple_robust_classification():
    # Main training pipeline
    
    # Configuration
    DATA_DIR = "NN_midi_files_extended"
    BATCH_SIZE = 16
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("🎼 Simple Robust MIDI Composer Classification")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load datasets
    train_dataset = RobustMIDIDataset(DATA_DIR, split='train', augment=True)
    val_dataset = RobustMIDIDataset(DATA_DIR, split='dev', augment=False)
    test_dataset = RobustMIDIDataset(DATA_DIR, split='test', augment=False)
    
    # Sync label encoders
    val_dataset.label_encoder = train_dataset.label_encoder
    test_dataset.label_encoder = train_dataset.label_encoder
    val_dataset.num_classes = train_dataset.num_classes
    test_dataset.num_classes = train_dataset.num_classes
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = EnhancedComposerModel(num_classes=train_dataset.num_classes, dropout=0.5).to(DEVICE)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ImprovedTrainer(model, DEVICE, train_dataset.num_classes)
    
    # Train
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    best_acc = trainer.train(train_loader, val_loader, train_dataset.label_encoder, epochs=EPOCHS)
    
    # Load best model for testing (with weights_only=False)
    print("\n" + "="*60)
    print("Loading best model for testing...")
    checkpoint = torch.load('best_model.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("\nFinal Test Evaluation:")
    print("="*60)
    test_acc, test_class_accs, test_preds, test_labels = trainer.validate(
        test_loader, train_dataset.label_encoder
    )
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Min Class Accuracy: {min(test_class_accs):.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.label_encoder.classes_,
                yticklabels=train_dataset.label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'trainer': trainer,
        'label_encoder': train_dataset.label_encoder,
        'device': DEVICE,
        'test_accuracy': test_acc,
        'test_class_accuracies': test_class_accs
    }


# In[8]:


class MIDIPredictor:
    # Simple predictor for inference
    
    def __init__(self, model, label_encoder, device):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.processor = SimpleMIDIProcessor(velocity_threshold=10)
        
    def predict(self, midi_path):
        # Predict composer for a single MIDI file
        self.model.eval()
        
        # Process MIDI
        features = self.processor.process_midi(midi_path)
        
        if features['num_notes'] == 0:
            return None, 0, []
        
        # Prepare inputs
        piano_roll = torch.FloatTensor(features['piano_roll']).unsqueeze(0).to(self.device)
        stats = torch.FloatTensor(features['stats']).unsqueeze(0).to(self.device)
        sequence = torch.FloatTensor(features['sequence']).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(piano_roll, stats, sequence)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            # Get top 3 predictions
            top3_probs, top3_idx = torch.topk(probs, min(3, len(self.label_encoder.classes_)), dim=1)
            top3_composers = self.label_encoder.inverse_transform(top3_idx.cpu().numpy()[0])
            
        predicted_composer = self.label_encoder.inverse_transform([predicted.item()])[0]
        
        return predicted_composer, confidence.item(), list(zip(top3_composers, top3_probs[0].cpu().numpy()))
    
    def predict_and_display(self, midi_path):
        # Predict and display results nicely
        prediction, confidence, top3 = self.predict(midi_path)
        
        if prediction is None:
            print(f"Could not process {midi_path}")
            return
        
        print(f"\n{'='*50}")
        print(f"File: {os.path.basename(midi_path)}")
        print(f"Predicted Composer: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nTop 3 Predictions:")
        for composer, prob in top3:
            print(f"  {composer}: {prob:.2%}")
        print('='*50)
        
        # Try to create audio
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            audio = midi_data.synthesize(fs=22050)
            
            # Normalize audio
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                return Audio(audio, rate=22050, autoplay=False)
        except:
            pass
        
        return None


# In[9]:


def batch_predict(model, directory, label_encoder, device):
    # Batch prediction on a directory
    predictor = MIDIPredictor(model, label_encoder, device)
    results = []
    
    # Find all MIDI files
    midi_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Process each file
    for midi_path in tqdm(midi_files, desc="Processing"):
        prediction, confidence, top3 = predictor.predict(midi_path)
        
        if prediction is not None:
            results.append({
                'file': os.path.basename(midi_path),
                'path': midi_path,
                'predicted': prediction,
                'confidence': confidence,
                'top2': top3[1][0] if len(top3) > 1 else None,
                'top2_conf': top3[1][1] if len(top3) > 1 else 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} files successfully")
        print(f"Average confidence: {df['confidence'].mean():.2%}")
        
        # Show distribution
        print("\nPrediction distribution:")
        print(df['predicted'].value_counts())
        
        # Save results
        df.to_csv('batch_predictions.csv', index=False)
        print("\nResults saved to batch_predictions.csv")
    
    return df


# In[10]:


def plot_training_history(trainer):
    # Plot training history
    if len(trainer.train_losses) == 0:
        print("No training history to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(trainer.val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# In[11]:


def analyze_predictions(predictions, labels, label_encoder):
    # Analyze prediction results
    from sklearn.metrics import classification_report
    
    print("\nDetailed Classification Report:")
    print("="*60)
    print(classification_report(labels, predictions, 
                              target_names=label_encoder.classes_,
                              digits=3))
    
    # Per-class analysis
    correct = np.array(predictions) == np.array(labels)
    
    for i in range(len(label_encoder.classes_)):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(correct[class_mask])
            composer = label_encoder.inverse_transform([i])[0]
            print(f"{composer}: {class_acc:.2%} ({np.sum(class_mask)} samples)")


# In[12]:


results = run_simple_robust_classification()


# In[13]:


predictor = MIDIPredictor(
                results['model'],
                results['label_encoder'],
                results['device']
            )


# In[14]:


predictor.predict_and_display('NN_midi_files_extended/test/handel/handel137.mid')


# In[17]:


predictor.predict_and_display('NN_midi_files_extended/test/schumann/schumann240.mid')


# In[18]:


predictor.predict_and_display('NN_midi_files_extended/test/chopin/chopin086.mid')


# In[20]:


plot_training_history(results['trainer'])

