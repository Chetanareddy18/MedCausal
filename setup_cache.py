"""
Setup cache for dashboard - downloads pre-computed outputs.
"""

import os
from pathlib import Path
import urllib.request
import zipfile


def ensure_cache(cache_dir: str = "outputs") -> bool:
    """Ensure cache exists, download if necessary."""
    cache_path = Path(cache_dir)
    cache_subdir = cache_path / "_cache"
    
    # If cache already exists, we're good
    if cache_subdir.exists() and any(cache_subdir.iterdir()):
        print(f"✅ Cache exists at {cache_subdir}")
        return True
    
    print("⏳ Downloading pre-computed outputs... (first load only)")
    
    try:
        # Create outputs directory if it doesn't exist
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Use GitHub releases as download source
        CACHE_ZIP_URL = "https://github.com/Chetanareddy18/MedCausal/releases/download/v1.0-outputs/outputs.zip"
        ZIP_FILE = cache_path / "outputs_temp.zip"
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                print(f"  📥 {percent:.0f}%", end="\r")
        
        print(f"  📥 Fetching from GitHub releases...")
        urllib.request.urlretrieve(CACHE_ZIP_URL, str(ZIP_FILE), download_progress)
        
        # Extract
        print(f"\n  📦 Extracting...")
        with zipfile.ZipFile(str(ZIP_FILE), 'r') as zf:
            zf.extractall(str(cache_path.parent))
        
        # Cleanup
        ZIP_FILE.unlink()
        print(f"✅ Cache downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download cache: {type(e).__name__}: {e}")
        print(f"⚠️  Dashboard may not work properly without cached outputs.")
        print(f"   Try visiting: https://github.com/Chetanareddy18/MedCausal/releases")
        return False


if __name__ == "__main__":
    ensure_cache()
