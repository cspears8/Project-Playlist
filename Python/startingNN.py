import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

all_song_features = np.load("all_songs_data.npy", encoding="latin1")
all_song_labels = np.load("all_song_labels.npy", encoding="latin1")

X_train, X_test, y_train, y_test = train_test_split(all_song_features, all_song_labels)

# Might not need this
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class GenreNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenreNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
hidden_size = 64
output_size = y_train.shape[1]

model = GenreNN(input_size, hidden_size, output_size)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs= 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    outputs = model(X_train_tensor)
    loss = loss_function(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
accuracy = (y_pred_binary == y_test_tensor).float().mean()
print(f"Accuracy: {accuracy.item():.4f}")