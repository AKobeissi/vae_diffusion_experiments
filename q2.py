import torch
import torch.utils.data
from torch import nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np

# Import your modules
from ddpm_utils.args import args
from ddpm_utils.dataset import MNISTDataset
from ddpm_utils.unet import UNet, load_weights
from q2_ddpm import DenoiseDiffusion
from q2_trainer_ddpm import Trainer

# Create output directory for images
os.makedirs('report_images', exist_ok=True)

# Set a fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Initialize the models and trainer
eps_model = UNet(c_in=1, c_out=1)
eps_model = load_weights(eps_model, args.MODEL_PATH)

diffusion_model = DenoiseDiffusion(
    eps_model=eps_model,
    n_steps=args.n_steps,
    device=args.device,
)

trainer = Trainer(args, eps_model, diffusion_model)

# 2. Load dataset
dataloader = torch.utils.data.DataLoader(
    MNISTDataset(),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

# 3. Function to train for a specific number of epochs and save images
def train_and_save_images(num_epochs=20, save_every=1):
    # Modify args to match our requirements
    original_epochs = args.epochs
    original_show_every = args.show_every_n_epochs
    
    args.epochs = num_epochs
    args.show_every_n_epochs = save_every
    
    # Train the model
    trainer.train(dataloader)
    
    # Restore original args
    args.epochs = original_epochs
    args.show_every_n_epochs = original_show_every

# 4. Generate and save intermediate samples at different timesteps
def generate_and_save_intermediate_samples():
    # Define steps to visualize (from T to 0)
    steps_to_show = [0, 100, 200, 400, 600, 800, 900, 950, 990, 999]
    
    # Generate samples
    images = trainer.generate_intermediate_samples(
        n_samples=6, 
        img_size=args.image_size,
        steps_to_show=steps_to_show,
        set_seed=True
    )
    
    # Plot and save the results
    plt.figure(figsize=(20, 12))
    
    # Get number of samples and steps
    n_samples = images[0].shape[0]
    n_steps = len(images)
    
    # Create a subplot grid
    fig, axs = plt.subplots(n_samples, n_steps, figsize=(n_steps*2, n_samples*2))
    
    # Plot each image
    for sample_idx in range(n_samples):
        for step_idx, img in enumerate(images):
            ax = axs[sample_idx, step_idx]
            ax.imshow(img[sample_idx, 0], cmap='gray')
            
            step = steps_to_show[step_idx] if step_idx < len(steps_to_show) else args.n_steps
            timestep = args.n_steps - step - 1 if step_idx > 0 else args.n_steps - 1
            
            ax.set_title(f'Sample {sample_idx+1}\nt={timestep}', fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('report_images/intermediate_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Generate loss curve
def plot_loss_curve():
    if hasattr(trainer, 'loss_per_iter') and len(trainer.loss_per_iter) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.loss_per_iter)
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('report_images/loss_curve.png', dpi=300)
        plt.close()

# ----- Execute the report generation -----

# Option 1: If you already have a trained model, just generate samples
try:
    # Check if model exists
    checkpoint = torch.load(args.MODEL_PATH)
    print("Found existing model. Generating samples from it...")
    
    # Generate intermediate samples
    generate_and_save_intermediate_samples()
    
except (FileNotFoundError, RuntimeError):
    print("No trained model found. Training a new model...")
    
    # Train for 20 epochs (or fewer for testing)
    train_and_save_images(num_epochs=20, save_every=1)
    
    # Plot loss curve
    plot_loss_curve()
    
    # Generate intermediate samples after training
    generate_and_save_intermediate_samples()

print("Report generation complete. Check the 'report_images' directory for results.")