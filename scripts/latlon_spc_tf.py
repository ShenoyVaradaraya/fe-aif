import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load data from the Excel file
file_path = 'Portland_66th_GE_Points.xlsx'

# Read the data into a pandas DataFrame
data = pd.read_excel(file_path)

# Display the first few rows of the data to ensure it's loaded correctly
print(data.head())

# Step 2: Extract features (latitude, longitude) and target values (SPCNorth_relpole, SPCEast_relpole)
X = data[['latitude', 'longitude']].values  # Latitude and longitude columns
y = data[['SPCNorth_relpole', 'SPCEast_relpole']].values  # SPCNorth_relpole and SPCEast_relpole columns

# Normalize the latitude and longitude values
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize the target values (SPC values)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 4: Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 5: Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# Step 6: Define the neural network model
class SPCNet(nn.Module):
    def __init__(self):
        super(SPCNet, self).__init__()
        self.layer1 = nn.Linear(2, 64)  # Input layer (2 input features)
        self.layer2 = nn.Linear(64, 32) # Hidden layer
        self.layer3 = nn.Linear(32, 2)  # Output layer (2 output values: SPCNorth_relpole, SPCEast_relpole)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Apply ReLU activation after the first layer
        x = self.relu(self.layer2(x))  # Apply ReLU activation after the second layer
        return self.layer3(x)  # Output layer

# Instantiate the model
model = SPCNet()

# Step 7: Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Train the model
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

# Step 9: Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to compute gradients for the test set
    test_loss = 0.0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")

# Step 10: Making predictions
def predict_spc(latitude, longitude):
    # Normalize the input
    input_data = scaler_X.transform([[latitude, longitude]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predicted_scaled = model(input_tensor)

    # Reverse the normalization to get the original values
    predicted = scaler_y.inverse_transform(predicted_scaled.numpy())
    return predicted[0]

# # Example prediction
# latitude = 44.883604
# longitude = -93.268128
# predicted_spc = predict_spc(latitude, longitude)
# print(f"Predicted SPCNorth_relpole: {predicted_spc[0]}, SPCEast_relpole: {predicted_spc[1]}")
