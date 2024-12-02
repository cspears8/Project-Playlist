import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

class GenreNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(GenreNN, self).__init__()
        # Fully define fully connected layers with drop out and batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def flatten_array_columns(df, length=10):
    for column in df.columns:
        if isinstance(df[column].iloc[0], (list, np.ndarray)):
            # Flatten each array to have 'length' number of elements (or pad with NaN)
            df[column] = df[column].apply(lambda x: x[:length] if isinstance(x, (list, np.ndarray)) else x)
            
            # If the array is shorter than the specified length, pad with NaN (or zeros if preferred)
            df[column] = df[column].apply(lambda x: x + [np.nan] * (length - len(x)) if len(x) < length else x)

            # Ensure the column is now a list of numbers with fixed length
            df[column] = df[column].apply(lambda x: np.array(x))
    
    return df

# combined_df = pd.read_csv('combined_data.csv')

# training_df = combined_df.drop(columns=['mbid', 'genre'])
# training_features = list(training_df.columns)
# joblib.dump(training_features, 'features.pkl')

# label_encoder = LabelEncoder()
# combined_df["genre"] = label_encoder.fit_transform(combined_df['genre'])
# print("Unique genres after encoding:", combined_df['genre'].nunique())

# combined_df = flatten_array_columns(combined_df)
# combined_df = combined_df.apply(pd.to_numeric, errors='coerce')
# combined_df.fillna(0, inplace=True)

# print(combined_df.dtypes)

# features_df = combined_df.drop(columns=['mbid', 'genre'])

# features = features_df.to_numpy()
# labels = combined_df['genre']

# # 70% Training, 15% Testing, 15% Validation split
# X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# print(X_train[:3])
# print(y_train[:3])

# # smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
# # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# scaler = StandardScaler()
# X_train_res = scaler.fit_transform(X_train)
# X_val = scaler.fit_transform(X_val)
# X_test = scaler.fit_transform(X_test)
# joblib.dump(scaler, 'scaler.pkl')

# # Flatten resampled training data into tensors
# X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# # Flatten testing and validation data into tensors
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# # Shows the split by data
# print(f"Training set size: {X_train_tensor.shape}, {y_train_tensor.shape}")
# print(f"Validation set size: {X_val_tensor.shape}, {y_val_tensor.shape}")
# print(f"Test set size: {X_test_tensor.shape}, {y_test_tensor.shape}")


# input_size = X_train.shape[1]
# hidden_size = 512
# output_size = len(label_encoder.classes_)

# model = GenreNN(input_size, hidden_size, output_size)

# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# # Use a scheduler to prevent overfitting and improve convergence
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)
# test_loader = DataLoader(test_dataset, batch_size=64)

# num_epochs = 50
# for epoch in range(num_epochs):
#     # Training step
#     model.train()
#     for inputs, labels in train_loader:
#         labels = labels.long()
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     # Validation step
#     model.eval()
#     val_loss = 0.0
#     val_accuracy = 0.0
#     all_val_labels = []
#     all_val_predictions = []
    
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             labels = labels.long()
#             outputs = model(inputs)
#             loss = loss_function(outputs, labels)
#             val_loss += loss.item()
            
#             # Apply softmax and get most likely
#             probabilities = torch.softmax(outputs, dim=1)
#             _, predicted_indices = torch.max(probabilities, dim=1)

#             all_val_labels.extend(labels.cpu().numpy())
#             all_val_predictions.extend(predicted_indices.cpu().numpy())

#             # Was our prediction correct? How many did we get right?
#             correct = (predicted_indices == labels).float().sum()
#             total = labels.size(0)
#             val_accuracy += correct / total

#     val_loss /= len(val_loader)
#     val_accuracy /= len(val_loader)

#     report = classification_report(all_val_labels, all_val_predictions, zero_division=0)

#     # Print the results
#     if epoch % 10 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}, "
#             f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")
#         print(report)
#     scheduler.step(val_loss)

# # Test step
# model.eval()
# test_loss = 0.0
# test_accuracy = 0.0
# with torch.no_grad():
#     for X_test_batch, y_test_batch in test_loader:
#         y_test_batch = y_test_batch.long()
#         test_outputs = model(X_test_batch)
#         loss = loss_function(test_outputs, y_test_batch)
#         test_loss += loss.item()

#         # Apply argmax and get most likely
#         test_predictions = torch.argmax(test_outputs, dim=1)
        
#         correct = (test_predictions == y_test_batch).float().sum()
#         total = y_test_batch.size(0)
#         test_accuracy += correct / total

# # Average loss and accuracy over all batches in the test set
# test_loss /= len(test_loader)
# test_accuracy /= len(test_loader)

# precision, recall, f1, _ = precision_recall_fscore_support(all_val_labels, all_val_predictions, average='macro')

# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
# torch.save(model.state_dict(), "GenreClassifier.pth")