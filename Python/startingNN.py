import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support

# Load features and the genres for each song
all_song_features = np.load("all_songs_data.npy", encoding="latin1")
all_song_labels = np.load("all_song_labels.npy", encoding="latin1")

y_all_labels = np.argmax(all_song_labels, axis=1)

# 70% Training, 15% Testing, 15% Validation split
X_train, X_temp, y_train, y_temp = train_test_split(all_song_features, y_all_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Flatten data into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Shows the split by data
print(f"Training set size: {X_train_tensor.shape}, {y_train_tensor.shape}")
print(f"Validation set size: {X_val_tensor.shape}, {y_val_tensor.shape}")
print(f"Test set size: {X_test_tensor.shape}, {y_test_tensor.shape}")

class GenreNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(GenreNN, self).__init__()
        # Fully define fully connected layers with drop out and batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
        )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

input_size = X_train.shape[1]
hidden_size = 768
output_size = 50

model = GenreNN(input_size, hidden_size, output_size)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use a scheduler to prevent overfitting and improve convergence
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

num_epochs = 100
for epoch in range(num_epochs):
    # Training step
    model.train()
    for inputs, labels in train_loader:
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation step
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    all_val_labels = []
    all_val_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.long()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            
            # Apply softmax and get most likely
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_indices = torch.max(probabilities, dim=1)

            all_val_labels.extend(labels.cpu().numpy())
            all_val_predictions.extend(predicted_indices.cpu().numpy())

            # Was our prediction correct? How many did we get right?
            correct = (predicted_indices == labels).float().sum()
            total = labels.size(0)
            val_accuracy += correct / total

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    report = classification_report(all_val_labels, all_val_predictions, zero_division=0)

    # Print the results
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")
        print(report)

# Test step
model.eval()
test_loss = 0.0
test_accuracy = 0.0
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        y_test_batch = y_test_batch.long()
        test_outputs = model(X_test_batch)
        loss = loss_function(test_outputs, y_test_batch)
        test_loss += loss.item()

        # Apply argmax and get most likely
        test_predictions = torch.argmax(test_outputs, dim=1)
        
        correct = (test_predictions == y_test_batch).float().sum()
        total = y_test_batch.size(0)
        test_accuracy += correct / total

# Average loss and accuracy over all batches in the test set
test_loss /= len(test_loader)
test_accuracy /= len(test_loader)

precision, recall, f1, _ = precision_recall_fscore_support(all_val_labels, all_val_predictions, average='macro')

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
torch.save(model.state_dict(), "GenreClassifier.pth")