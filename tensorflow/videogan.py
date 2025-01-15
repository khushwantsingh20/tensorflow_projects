import torch
import numpy as np
from moviepy.editor import ImageSequenceClip
from torchvision.utils import save_image

# Assuming you have a pretrained MoCoGAN generator
class PretrainedGenerator(torch.nn.Module):
    def __init__(self):
        super(PretrainedGenerator, self).__init__()
        # Define or load your pretrained generator here

    def forward(self, z, motion_noise):
        # Implement forward pass to generate frames
        pass

# Load pretrained generator
generator = PretrainedGenerator()
generator.load_state_dict(torch.load("pretrained_generator.pth"))
generator.eval()

# Generate a sequence of frames
def generate_frames(generator, num_frames=30, z_dim=100, motion_dim=10):
    z = torch.randn(1, z_dim)  # Latent vector for content
    motion_noise = torch.randn(num_frames, motion_dim)  # Motion noise for frames

    frames = []
    with torch.no_grad():
        for t in range(num_frames):
            frame = generator(z, motion_noise[t:t+1])  # Generate one frame
            frames.append(frame.squeeze(0))
    return frames

# Generate 1800 frames for a 60-second video at 30 FPS
fps = 30
num_frames = 60 * fps
frames = generate_frames(generator, num_frames=num_frames)

# Save frames as images (optional, for inspection)
for i, frame in enumerate(frames):
    save_image(frame, f"frame_{i:04d}.png")

# Convert frames to a video
def frames_to_video(frames, output_file="output_video.mp4", fps=30):
    images = [np.array(frame.permute(1, 2, 0).mul(255).byte().cpu().numpy()) for frame in frames]
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_file, codec="libx264")

frames_to_video(frames)
print("Video saved as 'output_video.mp4'")
