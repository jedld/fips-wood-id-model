import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, labels, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    # Generate random epsilon
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)
    
    # Interpolate between real and fake images
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    
    # Use the same labels for interpolated images
    interpolated_labels = labels

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, interpolated_labels)

    # Take the gradient of the scores with respect to the interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten the gradients
    gradient = gradient.view(BATCH_SIZE, -1)

    # Compute the L2 norm of the gradients for each sample in the batch
    gradient_norm = gradient.norm(2, dim=1)

    # Compute the gradient penalty
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
