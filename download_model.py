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
    
    # IMPORTANT: Disable offline mode for downloading
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
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
        
        # Try to download model with SSL bypass and explicit online mode
        model = SentenceTransformer('all-MiniLM-L6-v2', 
                                  cache_folder=cache_dir,
                                  trust_remote_code=True,
                                  device='cpu',
                                  local_files_only=False)
        
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
                                      device='cpu',
                                      local_files_only=False)
            print("Alternative download method successful!")
            return True
            
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("\n" + "="*60)
            print("MANUAL DOWNLOAD INSTRUCTIONS:")
            print("="*60)
            print("1. Go to: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
            print("2. Click 'Files and versions' tab")
            print("3. Download these files to a 'local_model' folder:")
            print("   - config.json")
            print("   - pytorch_model.bin")
            print("   - tokenizer.json")
            print("   - tokenizer_config.json")
            print("   - vocab.txt")
            print("   - modules.json")
            print("   - sentence_bert_config.json")
            print("4. Create folder structure:")
            print(f"   {os.getcwd()}\\local_model\\")
            print("5. Place all downloaded files in the local_model folder")
            print("6. Run the application - it will automatically detect the local model")
            print("="*60)
            return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\nModel download completed successfully!")
    else:
        print("\nModel download failed, but the app can still run with basic matching.")
