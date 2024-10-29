import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SongReccomenderNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SongReccomenderNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

input_features = np.load("processed_data.npy")

input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

input_size = input_features.shape[0]
hidden_size = 64
output_size = 8

model = SongReccomenderNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#TODO: Define our target data and what that should look like for training
example_target = torch.tensor([np.random.randint(0, output_size)], dtype=torch.long)

num_epochs= 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = model(input_tensor)

    loss = criterion(outputs, example_target)
    
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
print("Training complete.")