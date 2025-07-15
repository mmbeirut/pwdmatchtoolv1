"""
PWD Match Tool v5
Immigration Law Firm - Prevailing Wage Determination Matching Application

This Flask application allows users to compare job descriptions with existing
Prevailing Wage Determinations (PWDs) stored in a SQL Server database.
"""

from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available - using basic text matching")

# Database configuration
SERVER_NAME = "agd-vtanc-2016"
DATABASE_NAME = "ImmApps"
TABLE_NAME = "DOL_9141_form_20260731_allClients"

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Initialize sentence transformer model
model = None
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        import ssl
        import urllib3
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set environment variables to handle SSL and proxy issues
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_VERIFY'] = 'false'
        
        # Monkey patch requests to disable SSL verification globally
        original_request = requests.Session.request
        def patched_request(self, method, url, **kwargs):
            kwargs['verify'] = False
            return original_request(self, method, url, **kwargs)
        requests.Session.request = patched_request
        
        # Also patch the global requests functions
        requests.packages.urllib3.disable_warnings()
        
        # Monkey patch sentence transformers utilities
        import sentence_transformers.util
        original_http_get = sentence_transformers.util.http_get
        def patched_http_get(url, **kwargs):
            kwargs['verify'] = False
            return original_http_get(url, **kwargs)
        sentence_transformers.util.http_get = patched_http_get
        
        # Try to load the model with comprehensive SSL bypass
        cache_dir = os.path.join(os.getcwd(), 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set HuggingFace environment variables for offline mode
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Use offline mode to avoid SSL issues
        os.environ['HF_HUB_OFFLINE'] = '1'  # Also set hub offline mode
        
        logger.info("Attempting to load sentence transformer model in offline mode...")
        
        try:
            # Try multiple model loading approaches
            model_loaded = False
            
            # 1. Try loading from local model directory
            local_model_path = os.path.join(os.getcwd(), 'local_model')
            if os.path.exists(local_model_path):
                try:
                    model = SentenceTransformer(local_model_path, device='cpu')
                    logger.info(f"Successfully loaded model from local directory: {local_model_path}")
                    model_loaded = True
                except Exception as local_error:
                    logger.warning(f"Failed to load from local directory: {local_error}")
            
            # 2. Try loading from cache in offline mode
            if not model_loaded:
                try:
                    model = SentenceTransformer('all-MiniLM-L6-v2', 
                                              cache_folder=cache_dir,
                                              device='cpu',
                                              local_files_only=True)
                    logger.info("Successfully loaded sentence transformer model from cache: all-MiniLM-L6-v2")
                    model_loaded = True
                except Exception as cache_error:
                    logger.warning(f"Cache loading failed: {cache_error}")
            
            # 3. Try loading from Hugging Face cache directory
            if not model_loaded:
                hf_cache_path = os.path.expanduser("~/.cache/huggingface/transformers")
                if os.path.exists(hf_cache_path):
                    try:
                        # Look for cached model directories
                        for item in os.listdir(hf_cache_path):
                            if 'all-minilm-l6-v2' in item.lower():
                                cached_model_path = os.path.join(hf_cache_path, item)
                                model = SentenceTransformer(cached_model_path, device='cpu')
                                logger.info(f"Successfully loaded model from HF cache: {cached_model_path}")
                                model_loaded = True
                                break
                    except Exception as hf_error:
                        logger.warning(f"HF cache loading failed: {hf_error}")
            
            if not model_loaded:
                raise Exception("No local model found. Please download model manually.")
                
        except Exception as e:
            logger.error(f"All sentence transformer loading attempts failed: {e}")
            logger.info("Creating basic transformer fallback...")
        
            # Create a minimal fallback that mimics sentence transformer interface
            class BasicTransformer:
                def encode(self, texts):
                    """Basic encoding using simple text features"""
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Simple feature extraction: word count, character count, etc.
                    features = []
                    for text in texts:
                        words = text.lower().split()
                        feature_vector = [
                            len(words),                    # word count
                            len(text),                     # character count
                            len(set(words)),               # unique words
                            sum(len(word) for word in words) / max(len(words), 1),  # avg word length
                            text.count('.'),               # sentence count approximation
                        ]
                        # Pad to make it 384 dimensional like all-MiniLM-L6-v2
                        feature_vector.extend([0.0] * (384 - len(feature_vector)))
                        features.append(feature_vector)
                    
                    return np.array(features)
            
            model = BasicTransformer()
            logger.info("Using basic transformer fallback")
        
    except Exception as e:
        logger.error(f"All sentence transformer loading attempts failed: {e}")
        logger.info("Using basic text matching (sentence transformers not available)")
        model = None
        
else:
    logger.info("Using basic text matching (sentence transformers not available)")

class DatabaseManager:
    """Handles database connections and queries"""
    
    def __init__(self):
        self.connection_string = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        self.engine = None
        
    def get_engine(self):
        """Create and return database engine"""
        if not self.engine:
            try:
                self.engine = create_engine(self.connection_string)
                logger.info("Database engine created successfully")
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise
        return self.engine
    
    def get_pwds(self, filters=None):
        """Retrieve PWDs from database with optional filters"""
        try:
            engine = self.get_engine()
            
            # Base query - no case status filter
            query = f"SELECT * FROM [{TABLE_NAME}]"
            
            # Log the base query
            logger.info(f"Base query: {query}")
            
            # Add filters if provided
            if filters:
                filter_conditions = []
                
                if filters.get('companies'):
                    company_list = "', '".join(filters['companies'])
                    filter_conditions.append(f"[C.1] IN ('{company_list}')")
                
                if filters.get('locations'):
                    location_list = "', '".join(filters['locations'])
                    filter_conditions.append(f"CONCAT([F.e.3], ', ', [F.e.4]) IN ('{location_list}')")
                
                if filters.get('job_titles'):
                    title_list = "', '".join(filters['job_titles'])
                    filter_conditions.append(f"[F.a.1] IN ('{title_list}')")
                
                if filters.get('case_statuses'):
                    status_list = "', '".join(filters['case_statuses'])
                    filter_conditions.append(f"[Case Status] IN ('{status_list}')")
                
                if filter_conditions:
                    query += " WHERE " + " AND ".join(filter_conditions)
                    logger.info(f"Query with filters: {query}")
            
            # Execute query and log results
            df = pd.read_sql(query, engine)
            logger.info(f"Query executed successfully. Retrieved {len(df)} PWD records")
            
            # Log some sample data for debugging
            if len(df) > 0:
                logger.info(f"Sample companies: {df['C.1'].head().tolist()}")
                logger.info(f"Sample job titles: {df['F.a.1'].head().tolist()}")
                logger.info(f"Sample case statuses: {df['Case Status'].head().tolist()}")
            else:
                logger.warning("No records returned from query")
            
            return df
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            logger.error(f"Query was: {query}")
            return pd.DataFrame()
    
    def get_filter_options(self):
        """Get unique values for filter dropdowns"""
        try:
            engine = self.get_engine()
            
            # Get unique companies
            companies_query = f"SELECT DISTINCT [C.1] as company FROM [{TABLE_NAME}] WHERE [C.1] IS NOT NULL ORDER BY [C.1]"
            companies = pd.read_sql(companies_query, engine)['company'].tolist()
            
            # Get unique locations using F.e.3 and F.e.4
            locations_query = f"SELECT DISTINCT CONCAT([F.e.3], ', ', [F.e.4]) as location FROM [{TABLE_NAME}] WHERE [F.e.3] IS NOT NULL AND [F.e.4] IS NOT NULL ORDER BY location"
            locations = pd.read_sql(locations_query, engine)['location'].tolist()
            
            # Get unique job titles
            titles_query = f"SELECT DISTINCT [F.a.1] as title FROM [{TABLE_NAME}] WHERE [F.a.1] IS NOT NULL ORDER BY [F.a.1]"
            titles = pd.read_sql(titles_query, engine)['title'].tolist()
            
            # Get unique case statuses
            statuses_query = f"SELECT DISTINCT [Case Status] as status FROM [{TABLE_NAME}] WHERE [Case Status] IS NOT NULL ORDER BY [Case Status]"
            statuses = pd.read_sql(statuses_query, engine)['status'].tolist()
            
            return {
                'companies': companies,
                'locations': locations,
                'job_titles': titles,
                'case_statuses': statuses
            }
            
        except Exception as e:
            logger.error(f"Failed to get filter options: {e}")
            return {'companies': [], 'locations': [], 'job_titles': []}

class PWDMatcher:
    """Handles PWD matching logic using sentence transformers"""

    def __init__(self, model):
        self.model = model

    def calculate_similarity(self, job_data, pwd_records):
        """Calculate similarity scores between job data and PWD records"""
        if pwd_records.empty:
            return []

        # Stage 1: Exact company match if company specified
        if job_data.get('company'):
            company_name = job_data['company'].strip().lower()
            logger.info(f"Searching for company: '{company_name}'")
            logger.info(f"Number of records before company filter: {len(pwd_records)}")
            logger.info(f"Sample of company names in records: {pwd_records['C.1'].head().tolist()}")

            pwd_records = pwd_records[pwd_records['C.1'].str.strip().str.lower().str.contains(company_name, na=False)]

            logger.info(f"Number of records after company filter: {len(pwd_records)}")

            if pwd_records.empty:
                logger.info(f"No exact matches found for company: {job_data['company']}")
                return []

        # Stage 2: Calculate similarity on remaining records
        try:
            if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
                return self._calculate_semantic_similarity(job_data, pwd_records)
            else:
                return self._calculate_basic_similarity(job_data, pwd_records)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return []

    def _calculate_semantic_similarity(self, job_data, pwd_records):
            """Calculate similarity using sentence transformers"""
            # Create job description text
            job_text = self._create_job_text(job_data)
            job_skills_text = self._create_job_skills_text(job_data)

            # Create PWD texts
            pwd_texts = []
            pwd_skills_texts = []
            for _, pwd in pwd_records.iterrows():
                pwd_text = self._create_pwd_text(pwd)
                pwd_texts.append(pwd_text)
                pwd_skills_text = self._create_pwd_skills_text(pwd)
                pwd_skills_texts.append(pwd_skills_text)

            if not pwd_texts:
                return []

            # Calculate embeddings
            job_embedding = self.model.encode([job_text])
            job_skills_embedding = self.model.encode([job_skills_text]) if job_skills_text else None
            pwd_embeddings = self.model.encode(pwd_texts)
            logger.info(f"pwd_skills_texts before encoding: {pwd_skills_texts}")
            for i, text in enumerate(pwd_skills_texts):
                logger.info(f"  pwd_skills_texts[{i}] type: {type(text)}, value: '{text}'")
            # Create embeddings only for non-empty texts
            pwd_skills_embeddings = []
            for text in pwd_skills_texts:
                if text and isinstance(text, str) and text.strip():
                    # Encode valid text
                    embedding = self.model.encode([text])[0]
                else:
                    # Create zero vector for empty/invalid text
                    if not pwd_skills_embeddings:
                        # Get dimensions from a sample encoding
                        sample_encoding = self.model.encode(["sample text"])
                        zero_vector = np.zeros(sample_encoding.shape[1])
                    else:
                        zero_vector = np.zeros_like(pwd_skills_embeddings[0])
                    embedding = zero_vector
                pwd_skills_embeddings.append(embedding)
            pwd_skills_embeddings = np.array(pwd_skills_embeddings)
            
            logger.info(f"pwd_skills_embeddings after encoding: {pwd_skills_embeddings}")

            # Calculate cosine similarities
            job_similarities = cosine_similarity(job_embedding, pwd_embeddings)[0]

            # Create results with similarity scores
            results = []
            for i, (_, pwd) in enumerate(pwd_records.iterrows()):
                job_similarity_score = float(job_similarities[i])

                # Calculate skills similarity
                skills_similarity_score = 0.0
                
                # Only attempt skills similarity if we have valid embeddings
                if (job_skills_embedding is not None and 
                    isinstance(pwd_skills_embeddings[i], np.ndarray) and 
                    pwd_skills_embeddings[i].size > 0 and 
                    not np.all(pwd_skills_embeddings[i] == 0)):  # Skip zero vectors
                    try:
                        similarities = cosine_similarity(job_skills_embedding, [pwd_skills_embeddings[i]])[0]
                        if isinstance(similarities, np.ndarray) and similarities.size > 0:
                            skills_similarity_score = float(similarities[0])
                    except Exception as e:
                        logger.error(f"Skills similarity calculation failed: {str(e)}")
                        skills_similarity_score = 0.0

                # Calculate combined similarity
                try:
                    # Convert numpy values to Python floats
                    job_sim = float(job_similarity_score)
                    skills_sim = float(skills_similarity_score)
                    
                    # Calculate weighted average
                    combined_similarity = (0.7 * job_sim) + (0.3 * skills_sim)
                    match_strength = self._determine_match_strength(combined_similarity)
                except (ValueError, TypeError) as e:
                    logger.error(f"Combined similarity calculation failed: {e}")
                    combined_similarity = 0.0
                    match_strength = 'Very Weak'

                # Use F.e.3 and F.e.4 for location display
                location_parts = []
                if pwd.get('F.e.3'):
                    location_parts.append(pwd['F.e.3'])
                if pwd.get('F.e.4'):
                    location_parts.append(pwd['F.e.4'])
                job_location = ' '.join(location_parts) if location_parts else pwd.get('F.e.1', '')

                # Get wage info and check for wage issues
                wage_info = self._get_wage_info(pwd)
                wage_issue = False
                if wage_info and job_data.get('salary_range'):
                    try:
                        # Extract only digits and period for salary conversion
                        job_salary = float(''.join(c for c in job_data['salary_range'] if c.isdigit() or c == '.'))
                        if 'amount' in wage_info and wage_info['amount'] is not None:
                            if job_salary > float(wage_info.get('amount', 0)):
                                wage_issue = True
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to parse job salary or compare with wage info: {job_data['salary_range']} vs {wage_info.get('amount', 'N/A')}. Error: {e}")
                        pass  # Keep wage_issue as False

                # Combine job description fields
                job_desc = pwd.get('F.a.2', '')
                if pwd.get('Addendum_F.a.2'):
                    # Ensure concatenation is safe, e.g., if one is None
                    job_desc = f"{job_desc} {pwd['Addendum_F.a.2']}" if job_desc else pwd['Addendum_F.a.2']

                # Combine occupation requirement fields
                occupation_req = pwd.get('F.b.4.b', '')
                if pwd.get('Addendum_F.b.4.b'):
                    # Ensure concatenation is safe
                    occupation_req = f"{occupation_req} {pwd['Addendum_F.b.4.b']}" if occupation_req else pwd[
                        'Addendum_F.b.4.b']

                # Get validity period
                validity_period = ''
                if pwd.get('Validity Period From') and pwd.get('Validity Period To'):
                    validity_period = f"{pwd['Validity Period From']} to {pwd['Validity Period To']}"

                # Get ONET code
                onet_code = ''
                # Check for non-null/non-empty strings before concatenating
                f_d_1 = str(pwd['F.d.1']) if pd.notnull(pwd.get('F.d.1')) else ''
                f_d_1_a = str(pwd['F.d.1.a']) if pd.notnull(pwd.get('F.d.1.a')) else ''
                if f_d_1 and f_d_1_a:
                    onet_code = f"{f_d_1}-{f_d_1_a}"
                elif f_d_1:
                    onet_code = f_d_1
                elif f_d_1_a:
                    onet_code = f_d_1_a

                # Create result dictionary with all fields
                result_dict = {
                    'pwd_case_number': pwd.get('PWD Case Number', ''),
                    'company': pwd.get('C.1', ''),
                    'job_title': pwd.get('F.a.1', ''),
                    'job_location': job_location,
                    'job_description': job_desc.strip(),  # Use strip to clean up leading/trailing spaces
                    'education_required': self._get_education_level(pwd),
                    'experience_requirement': pwd.get('F.b.4.a', ''),
                    'alternate_experience': pwd.get('F.c.4.a', ''),
                    'occupation_requirement': occupation_req.strip(),  # Use strip to clean up
                    'special_skills': pwd.get('Addendum_F.b.5.a(iv)', ''),
                    'alternate_special_skills': pwd.get('Addendum_F.c.5.a(iv)', ''),
                    'similarity_score': combined_similarity,
                    'job_similarity': job_similarity_score,
                    'skills_similarity': skills_similarity_score,
                    'match_strength': match_strength,
                    'wage_info': wage_info,
                    'wage_issue': wage_issue,
                    'case_status': pwd.get('Case Status', ''),
                    'onet_code': onet_code,
                    'validity_period': validity_period
                }

                # Add travel requirement as display-only field (not used in similarity calculation)
                # Use .get() with default False for boolean fields
                result_dict['travel_required'] = 'Yes' if pwd.get('F.d.3.yes', False) is True else 'No'

                results.append(result_dict)

            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results
    
    def _calculate_basic_similarity(self, job_data, pwd_records):
        """Calculate basic text similarity without sentence transformers"""
        job_text = self._create_job_text(job_data).lower()
        job_skills_text = self._create_job_skills_text(job_data).lower()
        job_words = set(job_text.split())
        job_skills_words = set(job_skills_text.split()) if job_skills_text else set()
        
        results = []
        for _, pwd in pwd_records.iterrows():
            pwd_text = self._create_pwd_text(pwd).lower()
            pwd_skills_text = self._create_pwd_skills_text(pwd).lower()
            pwd_words = set(pwd_text.split())
            pwd_skills_words = set(pwd_skills_text.split()) if pwd_skills_text else set()
            
            # Calculate Jaccard similarity for job description (intersection over union)
            if job_words and pwd_words:
                intersection = len(job_words.intersection(pwd_words))
                union = len(job_words.union(pwd_words))
                job_similarity_score = intersection / union if union > 0 else 0.0
            else:
                job_similarity_score = 0.0
            
            # Calculate Jaccard similarity for skills
            skills_similarity_score = 0.0
            if job_skills_words and pwd_skills_words:
                skills_intersection = len(job_skills_words.intersection(pwd_skills_words))
                skills_union = len(job_skills_words.union(pwd_skills_words))
                skills_similarity_score = skills_intersection / skills_union if skills_union > 0 else 0.0
            
            # Combined similarity (weighted average: 70% job description, 30% skills)
            combined_similarity = (0.7 * job_similarity_score) + (0.3 * skills_similarity_score)
            match_strength = self._determine_match_strength(combined_similarity)
            
            # Use F.e.3 and F.e.4 for location display
            location_parts = []
            if pwd.get('F.e.3'):
                location_parts.append(pwd['F.e.3'])
            if pwd.get('F.e.4'):
                location_parts.append(pwd['F.e.4'])
            job_location = ' '.join(location_parts) if location_parts else pwd.get('F.e.1', '')
            
            results.append({
                'pwd_case_number': pwd.get('PWD Case Number', ''),
                'company': pwd.get('C.1', ''),
                'job_title': pwd.get('F.a.1', ''),
                'job_location': job_location,
                'job_description': pwd.get('F.a.2', ''),
                'education_required': self._get_education_level(pwd),
                'experience_required': pwd.get('F.b.4.a', ''),
                'similarity_score': combined_similarity,
                'job_similarity': job_similarity_score,
                'skills_similarity': skills_similarity_score,
                'match_strength': match_strength,
                'wage_info': self._get_wage_info(pwd),
                'case_status': pwd.get('Case Status', '')
            })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def _create_job_text(self, job_data):
        """Create searchable text from job data"""
        text_parts = []
        
        if job_data.get('job_title'):
            text_parts.append(f"Job Title: {job_data['job_title']}")
        if job_data.get('job_description'):
            text_parts.append(f"Description: {job_data['job_description']}")
        if job_data.get('education_level'):
            text_parts.append(f"Education: {job_data['education_level']}")
        if job_data.get('experience_required'):
            text_parts.append(f"Experience: {job_data['experience_required']}")
        if job_data.get('location'):
            text_parts.append(f"Location: {job_data['location']}")
        
        return " ".join(text_parts)
    
    def _create_job_skills_text(self, job_data):
        """Create searchable text from job skills data"""
        if job_data.get('skills'):
            return f"Skills: {job_data['skills']}"
        return ""
    
    def _create_pwd_text(self, pwd):
        """Create searchable text from PWD record"""
        text_parts = []
        
        if pwd.get('F.a.1'):
            text_parts.append(f"Job Title: {pwd['F.a.1']}")
        
        # Combine F.a.2 with Addendum1 for job description
        job_desc_parts = []
        if pwd.get('F.a.2'):
            job_desc_parts.append(pwd['F.a.2'])
        if pwd.get('Addendum1'):
            job_desc_parts.append(pwd['Addendum1'])
        
        if job_desc_parts:
            text_parts.append(f"Description: {' '.join(job_desc_parts)}")
        
        education = self._get_education_level(pwd)
        if education:
            text_parts.append(f"Education: {education}")
        
        if pwd.get('F.b.4.a'):
            text_parts.append(f"Experience: {pwd['F.b.4.a']}")
        
        # Use F.e.3 and F.e.4 for location instead of F.e.1
        location_parts = []
        if pwd.get('F.e.3'):
            location_parts.append(pwd['F.e.3'])
        if pwd.get('F.e.4'):
            location_parts.append(pwd['F.e.4'])
        
        if location_parts:
            text_parts.append(f"Location: {' '.join(location_parts)}")
        
        return " ".join(text_parts)
    
    def _create_pwd_skills_text(self, pwd):
        """Create searchable text from PWD skills data"""
        skills_parts = []
        
        # Safely handle Addendum4
        addendum4 = pwd.get('Addendum4')
        if addendum4 is not None and addendum4 != False:
            skills_parts.append(str(addendum4).strip())
            
        # Safely handle Addendum6
        addendum6 = pwd.get('Addendum6')
        if addendum6 is not None and addendum6 != False:
            skills_parts.append(str(addendum6).strip())
        
        # Only join non-empty strings
        skills_parts = [part for part in skills_parts if part]
        
        if skills_parts:
            return f"Skills: {' '.join(skills_parts)}"
        return ""
    
    def _get_education_level(self, pwd):
        """Extract education level from PWD record"""
        education_fields = {
            'F.b.1.Doctorate': 'Doctorate',
            'F.b.1.Masters': 'Masters',
            'F.b.1.Bachelors': 'Bachelors',
            'F.b.1.Associates': 'Associates',
            'F.b.1.HighSchoolGED': 'High School/GED',
            'F.b.1.None': 'None'
        }
        
        for field, level in education_fields.items():
            if pwd.get(field) == 'Yes':
                return level
        return ''

    def _get_wage_info(self, pwd):
        """Extract wage information from PWD record"""
        wage_info = {}

        # Get G.4 wage amount and period
        g4_amount = None
        g4_period = None
        for period in ['Hour', 'Week', 'BiWeekly', 'Month', 'Year']:
            field = f'G.4.a.{period}'
            val = pwd.get(field)
            if val is None:
                continue  # skip nulls
            try:
                # Some strings may contain commas, so remove them
                g4_amount_candidate = float(str(val).replace(',', '').strip())
                g4_amount = g4_amount_candidate
                g4_period = period
                break
            except (ValueError, TypeError):
                continue  # skip non-numeric values (like 'False', etc.)

        # Get G.5 wage amount if available
        g5_amount = None
        if pwd.get('G.5'):
            g5_value = pwd['G.5']
            if g5_value is None or (isinstance(g5_value, str) and g5_value.lower() in ['n/a', 'na', '', 'false']):
                g5_value = '0'
            try:
                g5_amount = float(str(g5_value).replace(',', '').strip())
            except (ValueError, TypeError):
                g5_amount = 0.0

        # Use the higher of G.4 and G.5
        if g4_amount is not None and g5_amount is not None:
            wage_info['amount'] = max(g4_amount, g5_amount)
            wage_info['period'] = g4_period
        elif g4_amount is not None:
            wage_info['amount'] = g4_amount
            wage_info['period'] = g4_period
        elif g5_amount is not None:
            wage_info['amount'] = g5_amount
            wage_info['period'] = 'Year'  # G.5 is typically annual

        # Get wage source
        wage_sources = {
            'G.4.c.OES_All_Industries': 'OES All Industries',
            'G.4.c.OES_ACWIA': 'OES ACWIA',
            'G.4.c.CBA': 'Collective Bargaining Agreement',
            'G.4.c.DBA': 'Davis-Bacon Act',
            'G.4.c.SCA': 'Service Contract Act'
        }

        for field, source in wage_sources.items():
            if pwd.get(field) == 'Yes':
                wage_info['source'] = source
                break

        return wage_info
    
    def _determine_match_strength(self, similarity_score):
        """Determine match strength based on similarity score"""
        if similarity_score >= 0.8:
            return 'Strong'
        elif similarity_score >= 0.6:
            return 'Moderate'
        elif similarity_score >= 0.4:
            return 'Weak'
        else:
            return 'Very Weak'

# Initialize components
db_manager = DatabaseManager()
pwd_matcher = PWDMatcher(model) if model else None

# Log model information for debugging
if model:
    model_type = type(model).__name__
    model_module = type(model).__module__
    logger.info(f"Initialized with model type: {model_type} from module: {model_module}")
    
    if 'sentence_transformers' in model_module.lower():
        logger.info("Using real SentenceTransformer model for semantic similarity")
    elif model_type == 'BasicTransformer':
        logger.info("Using BasicTransformer fallback with simple features")
    else:
        logger.info(f"Using unknown model type: {model_type}")
else:
    logger.info("No model loaded - will use basic text matching")

@app.route('/')
def index():
    """Main page with job input form"""
    try:
        filter_options = db_manager.get_filter_options()
        return render_template('index.html', filter_options=filter_options)
    except Exception as e:
        logger.error(f"Error loading main page: {e}")
        return render_template('error.html', error="Failed to load application"), 500

@app.route('/search', methods=['POST'])
def search_pwds():
    """Search and match PWDs based on job data"""
    try:
        # Get job data from form
        job_data = {
            'job_title': request.form.get('job_title', ''),
            'job_description': request.form.get('job_description', ''),
            'education_level': request.form.get('education_level', ''),
            'experience_requirement': request.form.get('experience_requirement', ''),
            'alternate_experience': request.form.get('alternate_experience', ''),
            'occupation_requirement': request.form.get('occupation_requirement', ''),
            'special_skills': request.form.get('special_skills', ''),
            'alternate_special_skills': request.form.get('alternate_special_skills', ''),
            'location': request.form.get('location', ''),
            'company': request.form.get('company', ''),
            'salary_range': request.form.get('salary_range', '')
        }
        
        # Log the search parameters for debugging
        logger.info(f"Search request - Job Title: '{job_data['job_title']}', Company: '{job_data['company']}'")
        
        # Get filters
        filters = {
            'companies': request.form.getlist('filter_companies'),
            'locations': request.form.getlist('filter_locations'),
            'job_titles': request.form.getlist('filter_job_titles'),
            'case_statuses': request.form.getlist('filter_case_statuses')
        }
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v}
        logger.info(f"Applied filters: {filters}")
        
        # Get PWD records
        pwd_records = db_manager.get_pwds(filters)
        logger.info(f"Retrieved {len(pwd_records)} PWD records from database")
        
        if pwd_records.empty:
            # Try a broader search without filters to see if data exists
            logger.info("No records found with filters, trying broader search...")
            all_records = db_manager.get_pwds()
            logger.info(f"Total certified PWD records in database: {len(all_records)}")
            
            return jsonify({
                'success': True,
                'results': [],
                'message': f'No PWD records found matching your criteria. Total PWDs in database: {len(all_records)}'
            })
        
        # Calculate similarities
        if pwd_matcher:
            results = pwd_matcher.calculate_similarity(job_data, pwd_records)
        else:
            # Fallback without similarity scoring
            results = []
            for _, pwd in pwd_records.iterrows():
                results.append({
                    'pwd_case_number': pwd.get('PWD Case Number', ''),
                    'company': pwd.get('C.1', ''),
                    'job_title': pwd.get('F.a.1', ''),
                    'job_location': pwd.get('F.e.1', ''),
                    'job_description': pwd.get('F.a.2', ''),
                    'similarity_score': 0.5,
                    'match_strength': 'Unknown',
                    'case_status': pwd.get('Case Status', '')
                })
        
        logger.info(f"Returning {len(results)} results")
        
        return jsonify({
            'success': True,
            'results': results[:50],  # Limit to top 50 results
            'total_found': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Search failed. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        engine = db_manager.get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'model': 'loaded' if model else 'not loaded'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/debug/data')
def debug_data():
    """Debug endpoint to check database data"""
    try:
        engine = db_manager.get_engine()
        
        # Check total records
        total_query = f"SELECT COUNT(*) as total FROM [{TABLE_NAME}]"
        total_result = pd.read_sql(total_query, engine)
        total_records = total_result['total'].iloc[0]
        
        # Check what case status values actually exist
        status_query = f"SELECT DISTINCT [Case Status], COUNT(*) as count FROM [{TABLE_NAME}] GROUP BY [Case Status]"
        status_result = pd.read_sql(status_query, engine)
        
        # Check determination issued records
        determination_query = f"SELECT COUNT(*) as determination FROM [{TABLE_NAME}] WHERE [Case Status] = 'Determination Issued'"
        determination_result = pd.read_sql(determination_query, engine)
        determination_records = determination_result['determination'].iloc[0]
        
        # Check for specific company (without case status filter)
        aecom_query = f"SELECT COUNT(*) as aecom FROM [{TABLE_NAME}] WHERE [C.1] LIKE '%AECOM%'"
        aecom_result = pd.read_sql(aecom_query, engine)
        aecom_records = aecom_result['aecom'].iloc[0]
        
        # Check for specific job title (without case status filter)
        civil_eng_query = f"SELECT COUNT(*) as civil_eng FROM [{TABLE_NAME}] WHERE [F.a.1] LIKE '%Civil Engineering%'"
        civil_eng_result = pd.read_sql(civil_eng_query, engine)
        civil_eng_records = civil_eng_result['civil_eng'].iloc[0]
        
        # Get sample records (without case status filter)
        sample_query = f"SELECT TOP 5 [PWD Case Number], [C.1], [F.a.1], [Case Status] FROM [{TABLE_NAME}]"
        sample_result = pd.read_sql(sample_query, engine)
        
        return jsonify({
            'total_records': int(total_records),
            'determination_records': int(determination_records),
            'aecom_records': int(aecom_records),
            'civil_engineering_records': int(civil_eng_records),
            'case_status_values': status_result.to_dict('records'),
            'sample_records': sample_result.to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/debug/model')
def debug_model():
    """Debug endpoint to check which model is being used"""
    try:
        model_info = {
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'model_loaded': model is not None,
            'pwd_matcher_initialized': pwd_matcher is not None
        }
        
        if model:
            model_info.update({
                'model_type': type(model).__name__,
                'model_module': type(model).__module__,
                'model_class': str(model.__class__)
            })
            
            # Test model encoding
            try:
                test_text = ["Test sentence for model verification"]
                embeddings = model.encode(test_text)
                model_info.update({
                    'encoding_test': 'success',
                    'embedding_shape': embeddings.shape,
                    'embedding_dimensions': embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'
                })
                
                # Determine model type
                if 'sentence_transformers' in model_info['model_module'].lower():
                    model_info['matching_method'] = 'semantic_similarity_sentence_transformers'
                elif model_info['model_type'] == 'BasicTransformer':
                    model_info['matching_method'] = 'basic_transformer_fallback'
                else:
                    model_info['matching_method'] = 'unknown'
                    
            except Exception as e:
                model_info.update({
                    'encoding_test': 'failed',
                    'encoding_error': str(e)
                })
        else:
            model_info['matching_method'] = 'basic_text_matching_jaccard'
        
        # Check model files
        local_model_path = os.path.join(os.getcwd(), 'local_model')
        cache_model_path = os.path.join(os.getcwd(), 'model_cache')
        
        model_info.update({
            'local_model_directory_exists': os.path.exists(local_model_path),
            'cache_model_directory_exists': os.path.exists(cache_model_path),
            'environment_variables': {
                'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE', 'not_set'),
                'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE', 'not_set'),
                'HF_HUB_DISABLE_TELEMETRY': os.environ.get('HF_HUB_DISABLE_TELEMETRY', 'not_set')
            }
        })
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Model debug query failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
