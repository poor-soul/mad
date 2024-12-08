import time
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the predefined model
print("Loading model...")
model = maskrcnn_resnet50_fpn(weights="MaskRCNN_ResNet50_FPN_Weights.COCO_V1")
model.eval()  # Set model to evaluation mode
print("Model loaded successfully.")

# Set the device (CPU in your case)
device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")

# Define dummy input data (simulate a batch of 4 images, 3x800x800 each)
batch_size = 4
dummy_input = torch.rand(batch_size, 3, 800, 800).to(device)
print("Dummy input tensor created.")

# Measure inference time
num_iterations = 50  # Number of iterations to average over
total_time = 0

print(f"Running {num_iterations} iterations to measure FPS...")
for iteration in range(num_iterations):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(dummy_input)
    end_time = time.time()
    total_time += end_time - start_time
    print(f"Iteration {iteration + 1}/{num_iterations} completed.")

# Calculate FPS
average_time_per_iteration = total_time / num_iterations
fps = batch_size / average_time_per_iteration

print(f"\nAverage inference time per batch: {average_time_per_iteration:.4f} seconds")
print(f"FPS: {fps:.2f}")
