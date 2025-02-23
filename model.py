# Import necessary packages
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import json
import time

# Paths
data_folder = "/mnt/mydisk/yogesh/ConvBmrk/animals"
labels_path = "imagenet-simple-labels.json"

# Load ResNet-50 model
# model = models.resnet50(pretrained=True)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()  

# Load labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Batch processing
bs = [16, 32, 64, 128, 256]
batch_size = bs[3]
image_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder) if img.endswith(('.jpg', '.png'))]
batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

total_time = 0.0
total_images = 0

print("<-----------Started Batch Inferencing...")
for batch in batches:
    images = []
    for img_path in batch:
        image = Image.open(img_path).convert("RGB")
        images.append(transform(image))
    
    # Create batch tensor
    batch_tensor = torch.stack(images).to('cpu')  
    batch_size_actual = len(batch)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted_indices = torch.max(outputs, 1)
    end_time = time.time()

    total_images += batch_size_actual
    total_time += (end_time - start_time)
    
    # Print predictions
    # predictions = [labels[idx] for idx in predicted_indices]
    # for img, pred in zip(batch, predictions):
    #     print(f"Image: {os.path.basename(img)} -> Predicted: {pred}")
print("<-----------Done Batch Inferencing...")

print(f"<-----------Tot Img: {total_images}")
print(f"<-----------Tot Time: {total_time}")
print(f"<-----------Batch_size: {batch_size}")
print(f"<-----------Latency: {(total_time/total_images) * 1000}ms")
print(f"<-----------Throughput: {(total_images / total_time)* 1000}ips")


# print(f"<-----------Latency: {(total_time/total_images) * 1000 :.4f}ms")
# print(f"<-----------Throughput: {(total_images / total_time)* 1000}ips")