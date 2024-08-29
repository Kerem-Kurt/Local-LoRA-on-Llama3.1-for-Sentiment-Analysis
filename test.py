import torch
import bitsandbytes as bnb

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your CUDA installation and GPU setup.")
else:
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")

# Create a simple model using standard PyTorch layers
model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32)
).cuda()

# Create a simple optimizer using bitsandbytes
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)

# Create a random input tensor and target tensor
input_tensor = torch.randn(32, 128).cuda()
target_tensor = torch.randn(32, 32).cuda()

# Loss function
criterion = torch.nn.MSELoss()

# Forward pass
output = model(input_tensor)

# Compute loss
loss = criterion(output, target_tensor)

# Backward pass and optimization step
loss.backward()
optimizer.step()

print("bitsandbytes 8-bit optimizer successfully ran on GPU!")
