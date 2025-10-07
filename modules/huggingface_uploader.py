"""
Hugging Face dataset uploader module for TRELLIS Streamlit application.

This module handles uploading generated 3D models and associated files to a 
Hugging Face dataset for public access and 3D printing workflow.

Usage:
    from modules.huggingface_uploader import HuggingFaceUploader
    
    uploader = HuggingFaceUploader()
    uploader.upload_generation(print_number, prompt, stl_path, glb_path, image)
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, upload_folder, upload_file
    HF_HUB_AVAILABLE = True
except ImportError:
    logger.warning("huggingface_hub not installed. Upload functionality will be disabled.")
    logger.warning("Install with: pip install huggingface_hub")
    HF_HUB_AVAILABLE = False


class HuggingFaceUploader:
    """
    Manages uploading generated 3D models to Hugging Face datasets.
    """
    
    def __init__(self,
                 dataset_id: str = "DamianBoborzi/gamescom2025",
                 local_sync_dir: str = "hf_dataset_sync",
                 auto_upload: bool = True):
        """
        Initialize the Hugging Face uploader.
        
        Args:
            dataset_id: Hugging Face dataset repository ID
            local_sync_dir: Local directory for syncing with HF dataset
            auto_upload: Whether to automatically upload on save
        """
        self.dataset_id = dataset_id
        self.local_sync_dir = Path(local_sync_dir)
        self.auto_upload = auto_upload
        self.api = None
        
        # Create local sync directory structure
        self._setup_directories()
        
        # Initialize HF API if available
        if HF_HUB_AVAILABLE:
            self._init_hf_api()
    
    def _setup_directories(self) -> None:
        """Create necessary directory structure for local sync."""
        # Main directories
        directories = [
            self.local_sync_dir,
            self.local_sync_dir / "stl_files",
            self.local_sync_dir / "glb_files", 
            self.local_sync_dir / "preview_images",
            self.local_sync_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create README if it doesn't exist
        readme_path = self.local_sync_dir / "README.md"
        if not readme_path.exists():
            self._create_readme(readme_path)
    
    def _create_readme(self, readme_path: Path) -> None:
        """Create a README file for the dataset."""
        readme_content = """# Gamescom 2025 - Text-to-3D Generations

This dataset contains 3D models generated at Gamescom 2025 using AI text-to-3D technology.

## Contents

- **stl_files/**: Ready-to-print STL files with numbered bases
- **glb_files/**: Original GLB 3D models  
- **preview_images/**: Preview images of the generated objects
- **metadata/**: JSON files with generation details

## File Naming Convention

Files are numbered sequentially (0001, 0002, etc.) with the print number engraved on the base for easy identification.

## Usage

You can download your generated model using the print number you received at the event.

### Finding Your Model

1. Note your print number (e.g., 0001)
2. Download the corresponding STL file from `stl_files/` for 3D printing
3. Or download the GLB file from `glb_files/` for viewing in 3D software

## License

These models were generated at a public event and are provided as-is for personal use.

---
*Generated with FLUX + TRELLIS AI models at Gamescom 2025*
"""
        readme_path.write_text(readme_content)
    
    def _init_hf_api(self) -> None:
        """Initialize Hugging Face API client."""
        try:
            # HF token should be set via environment variable (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
            self.api = HfApi()
            
            # Verify we can access the dataset
            try:
                self.api.dataset_info(self.dataset_id)
                logger.info(f"✅ Connected to Hugging Face dataset: {self.dataset_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not access dataset {self.dataset_id}: {e}")
                logger.warning("Files will be saved locally only. Check your HF token and dataset permissions.")
                self.auto_upload = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hugging Face API: {e}")
            self.auto_upload = False
    
    def save_generation_locally(self,
                               print_number: str,
                               prompt: str,
                               stl_path: Optional[str] = None,
                               glb_path: Optional[str] = None,
                               image: Optional[Image.Image] = None,
                               video_path: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Save generation files to local sync directory.
        
        Args:
            print_number: Unique print number (e.g., "0001")
            prompt: Text prompt used for generation
            stl_path: Path to STL file
            glb_path: Path to GLB file
            image: PIL Image preview
            video_path: Path to video file (optional)
            metadata: Additional metadata
            
        Returns:
            Dictionary of saved file paths
        """
        saved_paths = {}
        
        # Copy STL file
        if stl_path and os.path.exists(stl_path):
            dest_path = self.local_sync_dir / "stl_files" / f"{print_number}.stl"
            shutil.copy2(stl_path, dest_path)
            saved_paths["stl"] = dest_path
            logger.info(f"📁 Saved STL: {dest_path}")
        
        # Copy GLB file
        if glb_path and os.path.exists(glb_path):
            dest_path = self.local_sync_dir / "glb_files" / f"{print_number}.glb"
            shutil.copy2(glb_path, dest_path)
            saved_paths["glb"] = dest_path
            logger.info(f"📁 Saved GLB: {dest_path}")
        
        # Save preview image
        if image:
            dest_path = self.local_sync_dir / "preview_images" / f"{print_number}.png"
            image.save(dest_path, "PNG")
            saved_paths["preview"] = dest_path
            logger.info(f"📁 Saved preview: {dest_path}")
        
        # Save metadata
        metadata_dict = {
            "print_number": print_number,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "files": {
                "stl": f"stl_files/{print_number}.stl" if "stl" in saved_paths else None,
                "glb": f"glb_files/{print_number}.glb" if "glb" in saved_paths else None,
                "preview": f"preview_images/{print_number}.png" if "preview" in saved_paths else None
            }
        }
        
        if metadata:
            metadata_dict.update(metadata)
        
        metadata_path = self.local_sync_dir / "metadata" / f"{print_number}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        saved_paths["metadata"] = metadata_path
        logger.info(f"📁 Saved metadata: {metadata_path}")
        
        # Update index file
        self._update_index(print_number, prompt, metadata_dict)
        
        return saved_paths
    
    def _update_index(self, print_number: str, prompt: str, metadata: Dict[str, Any]) -> None:
        """Update the index.json file with new entry."""
        index_path = self.local_sync_dir / "index.json"
        
        # Load existing index or create new
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {
                "description": "Gamescom 2025 Text-to-3D Generations",
                "total_count": 0,
                "last_updated": None,
                "generations": []
            }
        
        # Add new entry
        entry = {
            "print_number": print_number,
            "prompt": prompt[:100],  # Truncate long prompts
            "timestamp": metadata["timestamp"],
            "has_stl": metadata["files"]["stl"] is not None,
            "has_glb": metadata["files"]["glb"] is not None,
            "has_preview": metadata["files"]["preview"] is not None
        }
        
        # Check if entry already exists (avoid duplicates)
        existing_numbers = [g["print_number"] for g in index["generations"]]
        if print_number not in existing_numbers:
            index["generations"].append(entry)
            index["total_count"] = len(index["generations"])
        else:
            # Update existing entry
            for i, gen in enumerate(index["generations"]):
                if gen["print_number"] == print_number:
                    index["generations"][i] = entry
                    break
        
        index["last_updated"] = datetime.now().isoformat()
        
        # Sort by print number
        index["generations"].sort(key=lambda x: x["print_number"])
        
        # Save updated index
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    
    def upload_to_huggingface(self, 
                             print_number: str,
                             paths: Optional[Dict[str, Path]] = None) -> bool:
        """
        Upload files to Hugging Face dataset.
        
        Args:
            print_number: Print number for the generation
            paths: Optional dict of paths to upload (if None, uploads entire folder)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not HF_HUB_AVAILABLE or not self.api:
            logger.warning("⚠️ Hugging Face upload not available")
            return False
        
        try:
            if paths:
                # Upload specific files
                for file_type, file_path in paths.items():
                    if file_path and file_path.exists():
                        # Determine path in repository
                        relative_path = file_path.relative_to(self.local_sync_dir)
                        
                        logger.info(f"⬆️ Uploading {file_type}: {relative_path}")
                        
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(relative_path),
                            repo_id=self.dataset_id,
                            repo_type="dataset",
                            commit_message=f"Add generation {print_number}"
                        )
            
            # Always upload the updated index
            index_path = self.local_sync_dir / "index.json"
            if index_path.exists():
                upload_file(
                    path_or_fileobj=str(index_path),
                    path_in_repo="index.json",
                    repo_id=self.dataset_id,
                    repo_type="dataset",
                    commit_message=f"Update index for {print_number}"
                )
            
            logger.info(f"✅ Successfully uploaded generation {print_number} to Hugging Face")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to Hugging Face: {e}")
            return False
    
    def upload_generation(self,
                         print_number: str,
                         prompt: str,
                         stl_path: Optional[str] = None,
                         glb_path: Optional[str] = None,
                         image: Optional[Image.Image] = None,
                         video_path: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save generation locally and optionally upload to Hugging Face.
        
        Args:
            print_number: Unique print number
            prompt: Text prompt used
            stl_path: Path to STL file
            glb_path: Path to GLB file
            image: Preview image
            video_path: Path to video file
            metadata: Additional metadata
            
        Returns:
            bool: True if successful (local save + optional upload)
        """
        # Always save locally first
        saved_paths = self.save_generation_locally(
            print_number, prompt, stl_path, glb_path, image, video_path, metadata
        )
        
        # Upload if auto_upload is enabled
        if self.auto_upload:
            return self.upload_to_huggingface(print_number, saved_paths)
        
        return True
    
    def sync_all_to_huggingface(self) -> bool:
        """
        Sync entire local directory to Hugging Face dataset.
        Useful for manual batch uploads.
        
        Returns:
            bool: True if successful
        """
        if not HF_HUB_AVAILABLE or not self.api:
            logger.warning("⚠️ Hugging Face upload not available")
            return False
        
        try:
            logger.info(f"📤 Syncing entire folder to Hugging Face dataset: {self.dataset_id}")
            
            upload_folder(
                folder_path=str(self.local_sync_dir),
                repo_id=self.dataset_id,
                repo_type="dataset",
                commit_message="Sync all local files"
            )
            
            logger.info("✅ Successfully synced all files to Hugging Face")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync to Hugging Face: {e}")
            return False
    
    def get_upload_status(self) -> Dict[str, Any]:
        """
        Get current upload status and statistics.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "hf_available": HF_HUB_AVAILABLE,
            "auto_upload_enabled": self.auto_upload,
            "dataset_id": self.dataset_id,
            "local_sync_dir": str(self.local_sync_dir)
        }
        
        # Count local files
        if self.local_sync_dir.exists():
            status["local_files"] = {
                "stl": len(list((self.local_sync_dir / "stl_files").glob("*.stl"))),
                "glb": len(list((self.local_sync_dir / "glb_files").glob("*.glb"))),
                "previews": len(list((self.local_sync_dir / "preview_images").glob("*.png"))),
                "metadata": len(list((self.local_sync_dir / "metadata").glob("*.json")))
            }
        
        return status


# Global instance for easy access
_global_uploader = None


def get_huggingface_uploader(**kwargs) -> HuggingFaceUploader:
    """
    Get the global HuggingFaceUploader instance.
    
    Args:
        **kwargs: Optional arguments to pass to HuggingFaceUploader constructor
        
    Returns:
        HuggingFaceUploader: The global uploader instance
    """
    global _global_uploader
    if _global_uploader is None:
        _global_uploader = HuggingFaceUploader(**kwargs)
    return _global_uploader