cat > test_torch_cuda.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Simple neural network
class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create a model and move it to the GPU if available
model = SimpleNN().to(device)

# Create dummy data
inputs = torch.randn(100, 10).to(device)  # 100 samples, 10 features each
targets = torch.randn(100, 1).to(device)  # 100 target values

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Check if CUDA is being utilized
if device.type == 'cuda':
    print("CUDA is working!")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)} bytes")
else:
    print("CUDA is not available. Using CPU.")
EOF
