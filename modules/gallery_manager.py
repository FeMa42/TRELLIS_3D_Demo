"""
Gallery management module for TRELLIS Streamlit application.

This module handles all gallery-related operations including:
- SQLite database management for gallery items
- File management for 3D models, images, and videos  
- Print number counter management
- Gallery item CRUD operations

Usage:
    from modules.gallery_manager import GalleryManager
    
    gallery = GalleryManager()
    gallery.save_item(print_number, prompt, image, stl_path, glb_path, video_path)
    items = gallery.load_items(search_query="dragon")
"""

import os
import sqlite3
import shutil
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image


class GalleryManager:
    """
    Manages gallery items, database operations, and file handling.
    """
    
    def __init__(self, 
                 gallery_dir: str = "gallery",
                 db_file: str = "gallery.db", 
                 counter_file: str = "file_counter.txt",
                 output_dir: str = "output"):
        """
        Initialize the GalleryManager.
        
        Args:
            gallery_dir: Directory to store gallery files
            db_file: SQLite database file path
            counter_file: File to store the print counter
            output_dir: Directory for output files
        """
        self.gallery_dir = gallery_dir
        self.db_file = db_file
        self.counter_file = counter_file
        self.output_dir = output_dir
        
        # Create necessary directories
        self._ensure_directories()
        
        # Initialize database
        self._init_database()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.gallery_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for gallery."""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                print_number TEXT UNIQUE NOT NULL,
                prompt TEXT NOT NULL,
                image_path TEXT NOT NULL,
                stl_path TEXT,
                glb_path TEXT,
                video_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Gallery database initialized")
    
    def save_item(self, 
                  print_number: str, 
                  prompt: str, 
                  image: Image.Image, 
                  stl_path: Optional[str] = None, 
                  glb_path: Optional[str] = None, 
                  video_path: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a creation to the gallery.
        
        Args:
            print_number: Unique identifier for the print
            prompt: Text prompt used to generate the item
            image: PIL Image to save as preview
            stl_path: Path to STL file (optional)
            glb_path: Path to GLB file (optional) 
            video_path: Path to video file (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Save preview image
            image_filename = f"{print_number}_preview.png"
            image_filepath = os.path.join(self.gallery_dir, image_filename)
            image.save(image_filepath)
            
            # Copy files to gallery if they exist
            gallery_stl_path = self._copy_file_to_gallery(stl_path, print_number, ".stl")
            gallery_glb_path = self._copy_file_to_gallery(glb_path, print_number, ".glb")
            gallery_video_path = self._copy_file_to_gallery(video_path, print_number, ".mp4")
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata["timestamp"] = datetime.now().isoformat()
            
            # Save to database
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            try:
                c.execute('''
                    INSERT OR REPLACE INTO gallery 
                    (print_number, prompt, image_path, stl_path, glb_path, video_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    print_number, 
                    prompt, 
                    image_filepath,
                    gallery_stl_path,
                    gallery_glb_path,
                    gallery_video_path,
                    json.dumps(metadata)
                ))
                conn.commit()
                print(f"✅ Gallery item #{print_number} saved successfully")
                return True
                
            except Exception as e:
                print(f"❌ Error saving to gallery database: {e}")
                return False
            finally:
                conn.close()
                
        except Exception as e:
            print(f"❌ Error saving gallery item: {e}")
            return False
    
    def _copy_file_to_gallery(self, source_path: Optional[str], print_number: str, extension: str) -> Optional[str]:
        """
        Copy a file to the gallery directory.
        
        Args:
            source_path: Path to source file
            print_number: Print number for naming
            extension: File extension (e.g., ".stl", ".glb", ".mp4")
            
        Returns:
            str: Path to copied file, or None if source doesn't exist
        """
        if source_path and os.path.exists(source_path):
            gallery_path = os.path.join(self.gallery_dir, f"{print_number}{extension}")
            shutil.copy2(source_path, gallery_path)
            return gallery_path
        return None
    
    def load_items(self, limit: int = 50, search_query: Optional[str] = None) -> List[Tuple]:
        """
        Load gallery items from database.
        
        Args:
            limit: Maximum number of items to return
            search_query: Optional search query to filter by print_number or prompt
            
        Returns:
            List of tuples containing gallery item data
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            if search_query:
                c.execute('''
                    SELECT * FROM gallery 
                    WHERE print_number LIKE ? OR prompt LIKE ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (f'%{search_query}%', f'%{search_query}%', limit))
            else:
                c.execute('''
                    SELECT * FROM gallery 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
            
            items = c.fetchall()
            return items
            
        except Exception as e:
            print(f"❌ Error loading gallery items: {e}")
            return []
        finally:
            conn.close()
    
    def get_item_by_number(self, print_number: str) -> Optional[Tuple]:
        """
        Get a specific gallery item by print number.
        
        Args:
            print_number: Print number to search for
            
        Returns:
            Tuple containing gallery item data, or None if not found
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            c.execute('SELECT * FROM gallery WHERE print_number = ?', (print_number,))
            item = c.fetchone()
            return item
        except Exception as e:
            print(f"❌ Error getting gallery item: {e}")
            return None
        finally:
            conn.close()
    
    def delete_item(self, print_number: str) -> bool:
        """
        Delete a gallery item and its associated files.
        
        Args:
            print_number: Print number of item to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        # Get item info before deleting
        item = self.get_item_by_number(print_number)
        if not item:
            return False
        
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            # Delete from database
            c.execute('DELETE FROM gallery WHERE print_number = ?', (print_number,))
            conn.commit()
            
            # Delete associated files
            for file_path in [item[3], item[4], item[5], item[6]]:  # image, stl, glb, video paths
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not delete file {file_path}: {e}")
            
            print(f"✅ Gallery item #{print_number} deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting gallery item: {e}")
            return False
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get gallery statistics.
        
        Returns:
            Dictionary with statistics about the gallery
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            # Total items
            c.execute('SELECT COUNT(*) FROM gallery')
            total_items = c.fetchone()[0]
            
            # Items with STL files
            c.execute('SELECT COUNT(*) FROM gallery WHERE stl_path IS NOT NULL')
            items_with_stl = c.fetchone()[0]
            
            # Items with GLB files
            c.execute('SELECT COUNT(*) FROM gallery WHERE glb_path IS NOT NULL')
            items_with_glb = c.fetchone()[0]
            
            # Items with videos
            c.execute('SELECT COUNT(*) FROM gallery WHERE video_path IS NOT NULL')
            items_with_video = c.fetchone()[0]
            
            return {
                "total_items": total_items,
                "items_with_stl": items_with_stl,
                "items_with_glb": items_with_glb,
                "items_with_video": items_with_video
            }
            
        except Exception as e:
            print(f"❌ Error getting gallery stats: {e}")
            return {"total_items": 0}
        finally:
            conn.close()
    
    # Print Counter Management
    def load_counter(self) -> int:
        """
        Load the current print counter value.
        
        Returns:
            int: Current counter value
        """
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, "r") as f:
                    return int(f.read().strip())
            except Exception as e:
                print(f"⚠️ Warning: Could not read counter file: {e}")
                return 1
        return 1
    
    def save_counter(self, value: int) -> None:
        """
        Save the print counter value.
        
        Args:
            value: Counter value to save
        """
        try:
            with open(self.counter_file, "w") as f:
                f.write(str(value))
        except Exception as e:
            print(f"❌ Error saving counter: {e}")
    
    def get_next_print_number(self) -> str:
        """
        Get the next print number without incrementing the counter.
        
        Returns:
            str: Next print number formatted as 4-digit string
        """
        count = self.load_counter()
        return f"{count:04d}"
    
    def increment_counter(self) -> str:
        """
        Increment the print counter and return the new number.
        
        Returns:
            str: New print number formatted as 4-digit string
        """
        count = self.load_counter()
        new_count = count + 1
        self.save_counter(new_count)
        return f"{count:04d}"  # Return the number that was just used


# Global instance for easy access
_global_gallery_manager = None


def get_gallery_manager(**kwargs) -> GalleryManager:
    """
    Get the global GalleryManager instance.
    
    Args:
        **kwargs: Optional arguments to pass to GalleryManager constructor
        
    Returns:
        GalleryManager: The global gallery manager instance
    """
    global _global_gallery_manager
    if _global_gallery_manager is None:
        _global_gallery_manager = GalleryManager(**kwargs)
    return _global_gallery_manager


# Convenience functions for backward compatibility
def init_database():
    """Legacy function for database initialization."""
    manager = get_gallery_manager()
    # Database is already initialized in __init__
    return True


def save_to_gallery(print_number: str, prompt: str, image: Image.Image, 
                   stl_path: Optional[str] = None, glb_path: Optional[str] = None, 
                   video_path: Optional[str] = None):
    """Legacy function for saving to gallery."""
    manager = get_gallery_manager()
    return manager.save_item(print_number, prompt, image, stl_path, glb_path, video_path)


def load_gallery(limit: int = 50, search_query: Optional[str] = None):
    """Legacy function for loading gallery items."""
    manager = get_gallery_manager()
    return manager.load_items(limit, search_query)


def get_gallery_item_by_number(print_number: str):
    """Legacy function for getting gallery item by number."""
    manager = get_gallery_manager()
    return manager.get_item_by_number(print_number)


def load_counter():
    """Legacy function for loading counter."""
    manager = get_gallery_manager()
    return manager.load_counter()


def save_counter(value: int):
    """Legacy function for saving counter."""
    manager = get_gallery_manager()
    manager.save_counter(value)


def get_next_file_number():
    """Legacy function for getting next file number."""
    manager = get_gallery_manager()
    return manager.get_next_print_number()