import torch
from functools import partial
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import ImageReward as RM

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