"""
Model Download Script
Downloads the sentence transformer model to avoid SSL issues during runtime
"""

import os
import ssl
import urllib3
from sentence_transformers import SentenceTransformer

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables to handle SSL issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

def download_model():
    """Download the sentence transformer model"""
    try:
        print("Downloading sentence transformer model...")
        
        # Create model cache directory
        cache_dir = os.path.join(os.getcwd(), 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download model
        model = SentenceTransformer('all-MiniLM-L6-v2', 
                                  cache_folder=cache_dir,
                                  trust_remote_code=True)
        
        print(f"Model downloaded successfully to: {cache_dir}")
        print("You can now run the main application.")
        
        return True
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("You may need to download the model manually or check your internet connection.")
        return False

if __name__ == "__main__":
    download_model()
