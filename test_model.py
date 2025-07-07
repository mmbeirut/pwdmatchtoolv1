"""
Test script to verify sentence transformer model is working
"""

import os
import sys

def test_model():
    """Test if the sentence transformer model is working"""
    try:
        # Set offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        
        from sentence_transformers import SentenceTransformer
        
        cache_dir = os.path.join(os.getcwd(), 'model_cache')
        
        print("Testing sentence transformer model...")
        print(f"Cache directory: {cache_dir}")
        print(f"Offline mode: {os.environ.get('TRANSFORMERS_OFFLINE', 'not set')}")
        
        # Try to load model from multiple locations
        model = None
        model_loaded = False
        
        # 1. Try loading from local model directory
        local_model_path = os.path.join(os.getcwd(), 'local_model')
        if os.path.exists(local_model_path):
            try:
                model = SentenceTransformer(local_model_path, device='cpu')
                print(f"Model loaded from local directory: {local_model_path}")
                model_loaded = True
            except Exception as local_error:
                print(f"Failed to load from local directory: {local_error}")
        
        # 2. Try loading from cache
        if not model_loaded:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', 
                                          cache_folder=cache_dir,
                                          device='cpu',
                                          local_files_only=True)
                print("Model loaded from cache")
                model_loaded = True
            except Exception as cache_error:
                print(f"Failed to load from cache: {cache_error}")
        
        if not model_loaded:
            raise Exception("No model found. Please download manually or run download_model.py")
        
        print("Model loaded successfully!")
        
        # Test encoding
        test_sentences = [
            "Software engineer with Python experience",
            "Data scientist with machine learning skills"
        ]
        
        embeddings = model.encode(test_sentences)
        print(f"Model encoding test successful! Shape: {embeddings.shape}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"Similarity calculation successful! Score: {similarity:.4f}")
        
        print("\nAll tests passed! The model is ready for use.")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nTry running: python download_model.py")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
