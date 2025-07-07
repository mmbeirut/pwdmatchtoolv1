"""
Model Download Script
Downloads the sentence transformer model to avoid SSL issues during runtime
"""

import os
import ssl
import urllib3
import requests
from sentence_transformers import SentenceTransformer

def setup_ssl_bypass():
    """Setup comprehensive SSL bypass"""
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Set environment variables to handle SSL issues
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_VERIFY'] = 'false'
    
    # Monkey patch requests to disable SSL verification globally
    original_request = requests.Session.request
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    requests.Session.request = patched_request
    
    # Patch sentence transformers utilities
    import sentence_transformers.util
    original_http_get = sentence_transformers.util.http_get
    def patched_http_get(url, **kwargs):
        kwargs['verify'] = False
        return original_http_get(url, **kwargs)
    sentence_transformers.util.http_get = patched_http_get

def download_model():
    """Download the sentence transformer model"""
    try:
        print("Setting up SSL bypass...")
        setup_ssl_bypass()
        
        print("Downloading sentence transformer model...")
        
        # Create model cache directory
        cache_dir = os.path.join(os.getcwd(), 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache directory: {cache_dir}")
        
        # Try to download model with SSL bypass
        model = SentenceTransformer('all-MiniLM-L6-v2', 
                                  cache_folder=cache_dir,
                                  trust_remote_code=True,
                                  device='cpu')
        
        print(f"Model downloaded successfully to: {cache_dir}")
        print("Testing model...")
        
        # Test the model
        test_sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_sentences)
        print(f"Model test successful. Embedding shape: {embeddings.shape}")
        
        print("You can now run the main application.")
        return True
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("\nTrying alternative approach...")
        
        try:
            # Try with different model identifier
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                      cache_folder=cache_dir,
                                      trust_remote_code=True,
                                      device='cpu')
            print("Alternative download method successful!")
            return True
            
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("\nPlease check your internet connection or try running the app anyway.")
            print("The app will fall back to basic text matching if the model isn't available.")
            return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n✓ Model download completed successfully!")
    else:
        print("\n✗ Model download failed, but the app can still run with basic matching.")
