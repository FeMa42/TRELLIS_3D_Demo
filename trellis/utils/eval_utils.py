import os
import copy
import pandas as pd
from PIL import Image
from subprocess import call, DEVNULL
import json
import torch
from functools import partial
import numpy as np
from dataset_toolkits.utils import sphere_hammersley_sequence, generate_views_from_angles
from trellis.utils import render_utils
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import ImageReward as RM
import utils3d  # Add utils3d import for proper transformations


BLENDER_PATH = "/home/damian/Projects/Diffus3D/blender-3.2.2-linux-x64/blender"

class ImageBasedPromptEvaluator:
    """
    A class to evaluate image-based prompts using CLIP and ImageReward.
    Uses ImageReward from: https://github.com/THUDM/ImageReward
    Args:
        model_name_or_path (str): Path to the CLIP model or model name from Hugging Face Hub.
    """
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch16"):
        self.model_name_or_path = model_name_or_path
        self.clip_score_fn = partial(clip_score, model_name_or_path=self.model_name_or_path)
        self.reward_model = RM.load("ImageReward-v1.0")

    def calculate_clip_score(self, images, prompt):
        """
        Calculate the CLIP score for a batch of images and a single prompt.
        Args:
            images (list): List of PIL.Image images to evaluate.  
            prompts (str): Single prompt corresponding to the images.
        Returns:
            float: The average CLIP score for the batch.
        """
        np_images = np.array(images, dtype=np.uint8)
        prompts = [prompt] * np_images.shape[0]
        with torch.no_grad():
            prompt_clip_score = self.clip_score_fn(torch.from_numpy(np_images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(prompt_clip_score), 4)

    def calculate_reward(self, images, prompt):
        """
        Calculate the reward for a batch of images and prompts using the ImageReward model.
        Args:
            images (list): List of PIL.Image images to evaluate. 
            prompts (str): Single prompt corresponding to the images. 
        Returns:
            torch.Tensor: The average reward for the batch.
        """
        with torch.no_grad():
            rewards = self.reward_model.score(prompt, images)
            # Convert to numpy array
            rewards = np.array(rewards)
            # mean over the batch
            rewards = np.mean(rewards)
        return round(float(rewards), 4)

    def evaluate(self, images, prompt):
        """
        Evaluate a batch of images using CLIP and ImageReward. It expects a batch of images and a single prompt since it is used for renders of a 3D model.
        Args:
            images (list): List of PIL.Image images to evaluate. 
            prompts (str): Single prompt corresponding to the images.
        Returns:
            dict: Dictionary containing the CLIP score and ImageReward score.
        """
        clip_score = self.calculate_clip_score(images, prompt)
        reward = self.calculate_reward(images, prompt)
        return {
            "clip_score": clip_score,
            "image_reward": reward
        }


def render_glb_frames(file_path, output_dir, num_frames=12, elevations=None, azimuths=None):
    """
    Render frames for a GLB file using Blender.

    Args:
        file_path (str): Path to the GLB file.
        output_dir (str): Directory to save the rendered images.
        num_frames (int): Number of frames to render.
        elevations (list): List of elevation angles for each frame.
    """
    output_folder = os.path.join(output_dir, 'renders')

    #azimuths =  [ 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    if azimuths is None:
        azimuths =  torch.linspace(0, 360, num_frames + 1)
        azimuths = azimuths.tolist()[:-1]
    elif isinstance(azimuths, (int, float)):
        print(f"Warning: Azimuths is a single value, generating frames from {azimuths} to 360+{azimuths}")
        # If azimuths is a single value, generate frames from that value to 360
        azimuths =  torch.linspace(azimuths, 360+azimuths, num_frames + 1)
        azimuths = azimuths.tolist()[:-1]
    elif len(azimuths) != num_frames:
        raise ValueError("Azimuths must be a list of length num_frames or a single value")
    if elevations is None:
        elevations =  [70] * len(azimuths)
    # check if elevations is a list or a single value
    elif isinstance(elevations, (int, float)):
        elevations = [elevations] * len(azimuths)
    elif len(elevations) != len(azimuths):
        raise ValueError("Azimuths and elevations must have the same length")

    views = generate_views_from_angles(azimuths, elevations, fixed_radius=1.5)
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join('/mnt/damian/Projects/TRELLIS/dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--output_folder', os.path.expanduser(output_folder),
        '--resolution', '1024',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'cond_rendered': True}

def get_images_from_glb(glb_path, output_dir, num_frames=12, elevations=None, azimuths=None):
    """
    Process a GLB file and render it 
    
    Args:
        glb_path (str): Path to the GLB file.
        output_dir (str): Directory to save the rendered images.
        num_frames (int): Number of frames to render.
        
        Returns:
        pil_images (list): List of rendered images.
    """
    # render the glb file
    render_glb_frames(glb_path, output_dir, num_frames=num_frames, elevations=elevations, azimuths=azimuths)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    render_output_dir = os.path.join(output_dir, 'renders')
    pil_images = []
    files_in_output_dir = os.listdir(render_output_dir)
    for file_name in files_in_output_dir:
        if file_name.endswith('.png'):
            image_path = os.path.join(render_output_dir, file_name)
            image = Image.open(image_path)
            pil_images.append(image.convert("RGB"))
    return pil_images

def get_images_from_gaussians(gaussian, num_frames=12, radius=2, fov=40, pitch=3.1415/12):
    """
    Process a Gaussian and render it using TRELLIS utilities.

    Args:
        gaussian: TRELLIS Gaussian object to render.
        num_frames (int): Number of frames to render.
        radius (float): Radius for the camera.
        fov (float): Field of view for the camera.
        pitch (float): Pitch angle for the camera.

    Returns:
        pil_images (list): List of rendered images.
    """
    # Use TRELLIS built-in camera utilities instead of manual transformations
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    yaws = yaws.tolist()
    pitchs = [pitch] * num_frames
    
    # Use TRELLIS camera transformation utilities
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, radius, fov)
    
    # Use TRELLIS rendering utilities instead of manual implementation
    frames = render_utils.render_frames(gaussian, extrinsics=extrinsics, intrinsics=intrinsics)['color']
    pil_images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    return pil_images

def apply_transform_to_gaussian(gaussian, transform_matrix):
    """
    Apply a transformation to a Gaussian using TRELLIS built-in utilities.
    
    Args:
        gaussian: TRELLIS Gaussian object
        transform_matrix: 3x3 or 4x4 transformation matrix
    
    Returns:
        transformed Gaussian object
    """
    # Use TRELLIS built-in transformation support
    # This is much safer than manual quaternion manipulation
    import tempfile
    import os
    
    # Save to temporary PLY with transformation
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Use built-in transformation in save_ply
        gaussian.save_ply(tmp_path, transform=transform_matrix[:3, :3].tolist())
        
        # Create new Gaussian and load transformed data
        new_gaussian = gaussian.__class__(**gaussian.init_params)
        new_gaussian.load_ply(tmp_path, transform=None)  # Already transformed
        
        return new_gaussian
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def sparse_structure_chamfer_distance(pred_bin, gt_bin):
    # Lower is better (0 = perfect match)
    # pred_bin, gt_bin: [B, 1, 64, 64, 64], binary
    batch_size = pred_bin.shape[0]
    chamfer = []
    for b in range(batch_size):
        pred_coords = pred_bin[b,0].nonzero(as_tuple=False).float()  # [N1, 3]
        gt_coords = gt_bin[b,0].nonzero(as_tuple=False).float()      # [N2, 3]
        if pred_coords.numel() == 0 or gt_coords.numel() == 0:
            chamfer.append(float('inf'))
            continue
        # pred to gt
        dist_pred_gt = torch.cdist(pred_coords, gt_coords).min(dim=1)[0].mean()
        # gt to pred
        dist_gt_pred = torch.cdist(gt_coords, pred_coords).min(dim=1)[0].mean()
        chamfer.append((dist_pred_gt + dist_gt_pred) / 2)
    return sum(chamfer) / len(chamfer)