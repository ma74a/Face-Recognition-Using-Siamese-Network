import torch

import matplotlib.pyplot as plt


def plot_metrics(train_losses, val_losses):
        plt.figure(figsize=(8, 6))
    
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='s')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()



def show_images(data_loader):

    img1_batch, img2_batch, labels = next(iter(data_loader))
    
    # Show fewer samples but make them larger
    num_samples = min(8, len(img1_batch))  # Show max 4 pairs for better visibility
    
    # Set up the plot with much larger figure size
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, num_samples*6))
    if num_samples == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D even for single sample
    
    # Denormalize function to show images properly
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        return tensor.clamp(0, 1)
    
    for i in range(num_samples):
        # Get the image pair and label
        img1 = denormalize(img1_batch[i])
        img2 = denormalize(img2_batch[i])
        label = labels[i].item()
        
        # Convert to numpy and change channel order for matplotlib
        img1 = img1.numpy().transpose((1, 2, 0))
        img2 = img2.numpy().transpose((1, 2, 0))
        
        # Plot the images with no borders
        axes[i, 0].imshow(img1)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2)
        axes[i, 1].axis('off')
        
        # Add label centered above the pair
        fig.text(0.5, 0.95 - (i * 0.95/num_samples), 
                f'{"SAME PERSON" if label == 0 else "DIFFERENT PEOPLE"}', 
                ha='center', va='center', 
                fontsize=16, weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
    
    # Adjust layout to maximize image size
    plt.subplots_adjust(top=0.94, 
                       bottom=0.01, 
                       left=0.01, 
                       right=0.99, 
                       hspace=0.1, 
                       wspace=0.05)
    plt.show()
