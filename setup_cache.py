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
        
        # Try multiple sources for robustness
        urls = [
            "https://github.com/Chetanareddy18/MedCausal/releases/download/v1.0-outputs/outputs.zip",
            "https://huggingface.co/spaces/ChetanaReddy18/MedCausal/resolve/main/outputs.zip",
        ]
        
        ZIP_FILE = cache_path / "outputs_temp.zip"
        success = False
        
        for url_idx, url in enumerate(urls, 1):
            try:
                print(f"  📥 Attempting download ({url_idx}/{len(urls)})...")
                urllib.request.urlretrieve(url, str(ZIP_FILE), _download_progress)
                print(f"\n  ✅ Downloaded!")
                success = True
                break
            except Exception as e:
                print(f"\n  ⚠️  Failed: {type(e).__name__}")
                if url_idx < len(urls):
                    print(f"     Trying next source...")
        
        if not success:
            raise Exception("All download sources exhausted")
        
        # Extract
        print(f"  📦 Extracting...")
        with zipfile.ZipFile(str(ZIP_FILE), 'r') as zf:
            zf.extractall(str(cache_path.parent))
        
        # Cleanup
        ZIP_FILE.unlink(missing_ok=True)
        print(f"✅ Cache ready!")
        return True
        
    except Exception as e:
        print(f"❌ Cache download failed: {type(e).__name__}: {e}")
        print(f"   ⚠️  Dashboard may not display properly without outputs.")
        print(f"   📖 Please check: https://github.com/Chetanareddy18/MedCausal")
        return False


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    if total_size > 0:
        downloaded = min(block_num * block_size, total_size)
        percent = (downloaded / total_size) * 100
        print(f"  📥 {percent:.0f}%", end="\r")


if __name__ == "__main__":
    ensure_cache()

