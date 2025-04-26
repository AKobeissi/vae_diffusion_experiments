import torch
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from q1_train_vae import VAE
from q1_vae import log_likelihood_bernoulli, kl_gaussian_gaussian_analytic

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
batch_size = 128
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False
)

# Model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    x_flat = x.view(x.size(0), -1)
    recon_loss = -log_likelihood_bernoulli(recon_x, x_flat).sum()
    kl = kl_gaussian_gaussian_analytic(
        mu, logvar,
        torch.zeros_like(mu), torch.zeros_like(logvar)
    ).sum()
    return recon_loss + kl

# Training & Validation
num_epochs = 20
train_losses = []
val_losses = []
for epoch in range(1, num_epochs+1):
    # Training
    model.train()
    running_train = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        running_train += loss.item()
        optimizer.step()
    avg_train = running_train / len(train_loader.dataset)
    train_losses.append(avg_train)
    
    # Validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            running_val += loss_function(recon, data, mu, logvar).item()
    avg_val = running_val / len(val_loader.dataset)
    val_losses.append(avg_val)
    
    print(f"Epoch {epoch:2d}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

# Save model
torch.save(model, 'model.pt')

# Plot 1: Learning curves
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.title('Learning Curves')
plt.savefig('learning_curves.png', bbox_inches='tight')
plt.close()

# Plot 2: Samples from prior
with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    samples = model.decode(z).cpu().view(64, 1, 28, 28)
grid = make_grid(samples, nrow=8, padding=2)
grid_img = grid[0].numpy()
plt.figure(figsize=(4,4))
plt.imshow(grid_img, cmap='gray')
plt.axis('off')
plt.title('Samples from VAE Prior')
plt.savefig('vae_samples.png', bbox_inches='tight')
plt.close()

# Plot 3: Latent traversals
eps_vals = torch.linspace(-3, 3, 5)
fig, axes = plt.subplots(20, 5, figsize=(10, 40))
with torch.no_grad():
    base_z = torch.randn(1, 20).to(device)
    for i in range(20):
        for j, eps in enumerate(eps_vals):
            z2 = base_z.clone()
            z2[0, i] += eps
            img = model.decode(z2).cpu().view(28, 28)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
plt.suptitle('Latent Traversals')
plt.tight_layout()
plt.savefig('latent_traversals.png', bbox_inches='tight')
plt.close()

# Plot 4: Interpolations
alphas = torch.linspace(0, 1, 11)
with torch.no_grad():
    z0 = torch.randn(1, 20).to(device)
    z1 = torch.randn(1, 20).to(device)
    latent_imgs = [model.decode(alpha*z0 + (1-alpha)*z1).cpu().view(28, 28) for alpha in alphas]
    x0 = model.decode(z0).cpu().view(28, 28)
    x1 = model.decode(z1).cpu().view(28, 28)
    data_imgs = [(alpha*x0 + (1-alpha)*x1) for alpha in alphas]
fig, axes = plt.subplots(2, len(alphas), figsize=(22, 4))
for idx in range(len(alphas)):
    axes[0, idx].imshow(latent_imgs[idx], cmap='gray'); axes[0, idx].axis('off')
    axes[1, idx].imshow(data_imgs[idx], cmap='gray'); axes[1, idx].axis('off')
axes[0, 0].set_ylabel('Latent Interpolation')
axes[1, 0].set_ylabel('Data Interpolation')
plt.tight_layout()
plt.savefig('interpolations.png', bbox_inches='tight')
plt.close()

print('Saved: model.pt, learning_curves.png, vae_samples.png, latent_traversals.png, interpolations.png')
