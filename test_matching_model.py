"""
Unit test to verify which matching model the application is using
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Add the project root to the path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestMatchingModel(unittest.TestCase):
    """Test which matching model is being used by the application"""
    
    def setUp(self):
        """Set up test environment"""
        # Import after setting up the path
        from app import model, pwd_matcher, SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = model
        self.pwd_matcher = pwd_matcher
        self.sentence_transformers_available = SENTENCE_TRANSFORMERS_AVAILABLE
    
    def test_model_type_detection(self):
        """Test what type of model is loaded"""
        print("\n" + "="*60)
        print("MATCHING MODEL DETECTION TEST")
        print("="*60)
        
        print(f"Sentence Transformers Available: {self.sentence_transformers_available}")
        
        if self.model is None:
            print("❌ NO MODEL LOADED - Using basic text matching fallback")
            self.assertIsNone(self.model)
            return
        
        # Check model type
        model_type = type(self.model).__name__
        model_module = type(self.model).__module__
        
        print(f"Model Type: {model_type}")
        print(f"Model Module: {model_module}")
        
        if hasattr(self.model, '__class__'):
            print(f"Model Class: {self.model.__class__}")
        
        # Test if it's a real SentenceTransformer
        if 'sentence_transformers' in model_module.lower():
            print("✅ USING REAL SENTENCE TRANSFORMERS MODEL")
            self.assertTrue('sentence_transformers' in model_module.lower())
            
            # Test model capabilities
            try:
                test_text = ["This is a test sentence"]
                embeddings = self.model.encode(test_text)
                print(f"✅ Model encoding test successful - Shape: {embeddings.shape}")
                print(f"✅ Embedding dimensions: {embeddings.shape[1]}")
                
                # Check if it's the expected model
                if hasattr(self.model, '_modules'):
                    print(f"✅ Model has modules: {list(self.model._modules.keys())}")
                
            except Exception as e:
                print(f"❌ Model encoding test failed: {e}")
                self.fail(f"Model encoding failed: {e}")
                
        elif model_type == 'BasicTransformer':
            print("⚠️  USING BASIC TRANSFORMER FALLBACK")
            self.assertEqual(model_type, 'BasicTransformer')
            
            # Test basic transformer
            try:
                test_text = ["This is a test sentence"]
                embeddings = self.model.encode(test_text)
                print(f"✅ Basic transformer encoding test successful - Shape: {embeddings.shape}")
                print(f"✅ Embedding dimensions: {embeddings.shape[1]}")
            except Exception as e:
                print(f"❌ Basic transformer encoding test failed: {e}")
                self.fail(f"Basic transformer encoding failed: {e}")
        else:
            print(f"❓ UNKNOWN MODEL TYPE: {model_type}")
    
    def test_pwd_matcher_functionality(self):
        """Test PWD matcher functionality"""
        print("\n" + "="*60)
        print("PWD MATCHER FUNCTIONALITY TEST")
        print("="*60)
        
        if self.pwd_matcher is None:
            print("❌ PWD Matcher not initialized")
            self.assertIsNone(self.pwd_matcher)
            return
        
        print("✅ PWD Matcher initialized")
        
        # Create test data
        test_job_data = {
            'job_title': 'Software Engineer',
            'job_description': 'Develop software applications using Python',
            'education_level': 'Bachelors',
            'experience_required': '3 years',
            'skills': 'Python, SQL',
            'location': 'New York, NY',
            'company': 'Test Company',
            'salary_range': '$70,000 - $90,000'
        }
        
        # Create test PWD records
        test_pwd_data = {
            'PWD Case Number': ['PWD-001', 'PWD-002'],
            'C.1': ['Company A', 'Company B'],
            'F.a.1': ['Software Developer', 'Data Analyst'],
            'F.a.2': ['Develop web applications', 'Analyze business data'],
            'F.e.1': ['New York, NY', 'California, CA'],
            'Case Status': ['Determination Issued', 'Pending Determination - Unassigned'],
            'F.b.4.a': ['2 years', '5 years']
        }
        
        test_pwd_records = pd.DataFrame(test_pwd_data)
        
        try:
            # Test similarity calculation
            results = self.pwd_matcher.calculate_similarity(test_job_data, test_pwd_records)
            
            print(f"✅ Similarity calculation successful")
            print(f"✅ Number of results: {len(results)}")
            
            if results:
                first_result = results[0]
                print(f"✅ First result similarity score: {first_result.get('similarity_score', 'N/A')}")
                print(f"✅ First result match strength: {first_result.get('match_strength', 'N/A')}")
                
                # Check which method was used
                if hasattr(self.pwd_matcher, 'model') and self.pwd_matcher.model:
                    if 'sentence_transformers' in str(type(self.pwd_matcher.model)):
                        print("✅ USING SEMANTIC SIMILARITY (Sentence Transformers)")
                    else:
                        print("⚠️  USING BASIC TRANSFORMER FALLBACK")
                else:
                    print("⚠️  USING JACCARD SIMILARITY (Basic Text Matching)")
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            self.fail(f"Similarity calculation failed: {e}")
    
    def test_model_loading_path(self):
        """Test which model loading path was used"""
        print("\n" + "="*60)
        print("MODEL LOADING PATH TEST")
        print("="*60)
        
        # Check for local model directory
        local_model_path = os.path.join(os.getcwd(), 'local_model')
        cache_model_path = os.path.join(os.getcwd(), 'model_cache')
        
        print(f"Local model directory exists: {os.path.exists(local_model_path)}")
        print(f"Cache model directory exists: {os.path.exists(cache_model_path)}")
        
        if os.path.exists(local_model_path):
            print(f"✅ Local model directory found: {local_model_path}")
            # Check for required files
            required_files = [
                'config.json', 'pytorch_model.bin', 'modules.json',
                'sentence_bert_config.json', 'tokenizer.json'
            ]
            
            for file in required_files:
                file_path = os.path.join(local_model_path, file)
                exists = os.path.exists(file_path)
                status = "✅" if exists else "❌"
                print(f"{status} {file}: {exists}")
            
            # Check for pooling config
            pooling_config = os.path.join(local_model_path, '1_Pooling', 'config.json')
            exists = os.path.exists(pooling_config)
            status = "✅" if exists else "❌"
            print(f"{status} 1_Pooling/config.json: {exists}")
        
        # Check environment variables
        print(f"\nEnvironment Variables:")
        print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'not set')}")
        print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
        print(f"HF_HUB_DISABLE_TELEMETRY: {os.environ.get('HF_HUB_DISABLE_TELEMETRY', 'not set')}")

def run_tests():
    """Run all tests and display results"""
    unittest.main(verbosity=2, exit=False)

if __name__ == "__main__":
    print("Starting Matching Model Detection Tests...")
    run_tests()
