import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

# Define a transform to convert the images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the SVHN dataset
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

# Load the entire dataset into memory
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Select 10 random images
random_indices = np.random.choice(images.shape[0], 10, replace=False)
random_images = images[random_indices]
random_labels = labels[random_indices]

# Create a directory to save the images if it doesn't exist
output_dir = 'random_svhn_images'
os.makedirs(output_dir, exist_ok=True)

# Function to unnormalize and save an image
def save_image(img, label, index):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'image_{index}.png'))
    plt.close()

# Save the images
for i in range(10):
    save_image(random_images[i], random_labels[i].item(), i)

logging.info(f"Images saved to '{output_dir}' directory.")
