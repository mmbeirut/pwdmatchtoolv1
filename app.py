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


# --- Safe strip helper ---
def safe_strip(value, field_name=None, row_idx=None):
    try:
        if isinstance(value, str):
            return value.strip()
        elif value is None:
            return ""
        else:
            return str(value).strip()
    except Exception as e:
        logger.error(f"safe_strip failed on field '{field_name}', row {row_idx}, value: {repr(value)} - {e}")
        return ""


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
                def encode(self, texts, show_progress_bar=False):
                    """Basic encoding using simple text features"""
                    if isinstance(texts, str):
                        texts = [texts]

                    # Simple feature extraction: word count, character count, etc.
                    features = []
                    for text in texts:
                        words = str(text).lower().split()
                        feature_vector = [
                            len(words),  # word count
                            len(str(text)),  # character count
                            len(set(words)),  # unique words
                            sum(len(word) for word in words) / max(len(words), 1),  # avg word length
                            str(text).count('.'),  # sentence count approximation
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
            company_name = job_data['company']
            if company_name is not None:
                company_name = company_name.strip().lower()
            else:
                company_name = ""
            # PATCH: Use fillna('') to avoid NoneType errors when calling .str methods
            pwd_records = pwd_records[
                pwd_records['C.1'].fillna('').str.strip().str.lower().str.contains(company_name, na=False)
            ]

            if pwd_records.empty:
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
        """Calculate similarity using sentence transformers with multi-factor scoring"""
        results = []
        
        for i, (_, pwd) in enumerate(pwd_records.iterrows()):
            # Calculate individual category similarities
            education_score = self._calculate_education_similarity(job_data, pwd)
            experience_score = self._calculate_experience_similarity(job_data, pwd, i)
            occupation_score = self._calculate_occupation_similarity(job_data, pwd, i)
            skills_score = self._calculate_skills_similarity(job_data, pwd, i)
            job_desc_score = self._calculate_job_description_similarity(job_data, pwd, i)
            location_score = self._calculate_location_similarity(job_data, pwd, i)
            
            # Define base weights
            base_weights = {
                'education': 0.2,
                'experience': 0.2,
                'occupation': 0.2,
                'skills': 0.2,
                'job_description': 0.1,
                'location': 0.1
            }
            
            # Collect available scores and calculate final similarity
            available_scores = {}
            if education_score is not None:
                available_scores['education'] = education_score
            if experience_score is not None:
                available_scores['experience'] = experience_score
            if occupation_score is not None:
                available_scores['occupation'] = occupation_score
            if skills_score is not None:
                available_scores['skills'] = skills_score
            if job_desc_score is not None:
                available_scores['job_description'] = job_desc_score
            if location_score is not None:
                available_scores['location'] = location_score
            
            # Redistribute weights proportionally for missing categories
            if available_scores:
                total_available_weight = sum(base_weights[cat] for cat in available_scores.keys())
                weight_multiplier = 1.0 / total_available_weight if total_available_weight > 0 else 0
                
                combined_similarity = sum(
                    available_scores[cat] * base_weights[cat] * weight_multiplier 
                    for cat in available_scores.keys()
                )
            else:
                combined_similarity = 0.0
            
            match_strength = self._determine_match_strength(combined_similarity)
            
            # For backward compatibility, also calculate individual job and skills scores
            job_similarity_score = job_desc_score if job_desc_score is not None else 0.0
            skills_similarity_score = skills_score if skills_score is not None else 0.0

            # Use F.e.3 and F.e.4 for location display
            location_parts = []
            if pwd.get('F.e.3'):
                location_parts.append(safe_strip(pwd.get('F.e.3'), 'F.e.3', i))
            if pwd.get('F.e.4'):
                location_parts.append(safe_strip(pwd.get('F.e.4'), 'F.e.4', i))
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
                job_desc = f"{safe_strip(job_desc, 'F.a.2', i)} {safe_strip(pwd['Addendum_F.a.2'], 'Addendum_F.a.2', i)}" if job_desc else safe_strip(
                    pwd['Addendum_F.a.2'], 'Addendum_F.a.2', i)

            # Combine occupation requirement fields
            occupation_req = pwd.get('F.b.4.b', '')
            if pwd.get('Addendum_F.b.4.b'):
                occupation_req = f"{safe_strip(occupation_req, 'F.b.4.b', i)} {safe_strip(pwd['Addendum_F.b.4.b'], 'Addendum_F.b.4.b', i)}" if occupation_req else safe_strip(
                    pwd['Addendum_F.b.4.b'], 'Addendum_F.b.4.b', i)

            # Get validity period
            validity_period = ''
            if pwd.get('Validity Period From') and pwd.get('Validity Period To'):
                validity_period = f"{pwd['Validity Period From']} to {pwd['Validity Period To']}"

            # Get ONET code
            onet_code = ''
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
                'job_description': safe_strip(job_desc, 'F.a.2/Addendum_F.a.2', i),
                'required_education': self._get_education_level(pwd, 'required'),
                'alternate_education': self._get_education_level(pwd, 'alternate'),
                'required_experience': pwd.get('F.b.4.a', ''),
                'alternate_experience': pwd.get('F.c.4.a', ''),
                'occupation_requirement': safe_strip(occupation_req, 'F.b.4.b/Addendum_F.b.4.b', i),
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
                'validity_period': validity_period,
                # Individual category scores for detailed breakdown
                'education_score': education_score,
                'experience_score': experience_score,
                'occupation_score': occupation_score,
                'skills_score': skills_score,
                'job_description_score': job_desc_score,
                'location_score': location_score
            }

            # Add travel requirement as display-only field (not used in similarity calculation)
            result_dict['travel_required'] = 'Yes' if pwd.get('F.d.3.yes', False) is True else 'No'

            results.append(result_dict)

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

    def _calculate_basic_similarity(self, job_data, pwd_records):
        """Calculate similarity using basic text matching with multi-factor scoring"""
        results = []
        
        for i, (_, pwd) in enumerate(pwd_records.iterrows()):
            # Calculate individual category similarities using basic text matching
            education_score = self._calculate_education_similarity(job_data, pwd)
            experience_score = self._calculate_experience_similarity_basic(job_data, pwd, i)
            occupation_score = self._calculate_occupation_similarity_basic(job_data, pwd, i)
            skills_score = self._calculate_skills_similarity_basic(job_data, pwd, i)
            job_desc_score = self._calculate_job_description_similarity_basic(job_data, pwd, i)
            location_score = self._calculate_location_similarity_basic(job_data, pwd, i)
            
            # Define base weights
            base_weights = {
                'education': 0.2,
                'experience': 0.2,
                'occupation': 0.2,
                'skills': 0.2,
                'job_description': 0.1,
                'location': 0.1
            }
            
            # Collect available scores and calculate final similarity
            available_scores = {}
            if education_score is not None:
                available_scores['education'] = education_score
            if experience_score is not None:
                available_scores['experience'] = experience_score
            if occupation_score is not None:
                available_scores['occupation'] = occupation_score
            if skills_score is not None:
                available_scores['skills'] = skills_score
            if job_desc_score is not None:
                available_scores['job_description'] = job_desc_score
            if location_score is not None:
                available_scores['location'] = location_score
            
            # Redistribute weights proportionally for missing categories
            if available_scores:
                total_available_weight = sum(base_weights[cat] for cat in available_scores.keys())
                weight_multiplier = 1.0 / total_available_weight if total_available_weight > 0 else 0
                
                combined_similarity = sum(
                    available_scores[cat] * base_weights[cat] * weight_multiplier 
                    for cat in available_scores.keys()
                )
            else:
                combined_similarity = 0.0
            
            match_strength = self._determine_match_strength(combined_similarity)
            
            # For backward compatibility, also calculate individual job and skills scores
            job_similarity_score = job_desc_score if job_desc_score is not None else 0.0
            skills_similarity_score = skills_score if skills_score is not None else 0.0

            # Use F.e.3 and F.e.4 for location display
            location_parts = []
            if pwd.get('F.e.3'):
                location_parts.append(safe_strip(pwd['F.e.3'], 'F.e.3', i))
            if pwd.get('F.e.4'):
                location_parts.append(safe_strip(pwd['F.e.4'], 'F.e.4', i))
            job_location = ' '.join(location_parts) if location_parts else pwd.get('F.e.1', '')

            results.append({
                'pwd_case_number': pwd.get('PWD Case Number', ''),
                'company': pwd.get('C.1', ''),
                'job_title': pwd.get('F.a.1', ''),
                'job_location': job_location,
                'job_description': safe_strip(pwd.get('F.a.2', ''), 'F.a.2', i),
                'required_education': self._get_education_level(pwd, 'required'),
                'alternate_education': self._get_education_level(pwd, 'alternate'),
                'required_experience': pwd.get('F.b.4.a', ''),
                'similarity_score': combined_similarity,
                'job_similarity': job_similarity_score,
                'skills_similarity': skills_similarity_score,
                'match_strength': match_strength,
                'wage_info': self._get_wage_info(pwd),
                'case_status': pwd.get('Case Status', ''),
                # Individual category scores for detailed breakdown
                'education_score': education_score,
                'experience_score': experience_score,
                'occupation_score': occupation_score,
                'skills_score': skills_score,
                'job_description_score': job_desc_score,
                'location_score': location_score
            })

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

    def _create_job_text(self, job_data, row_idx=None):
        """Create searchable text from job data"""
        text_parts = []

        if job_data.get('job_title'):
            text_parts.append(f"Job Title: {safe_strip(job_data['job_title'], 'job_title', row_idx)}")
        if job_data.get('job_description'):
            text_parts.append(f"Description: {safe_strip(job_data['job_description'], 'job_description', row_idx)}")
        if job_data.get('required_education'):
            text_parts.append(
                f"Required Education: {safe_strip(job_data['required_education'], 'required_education', row_idx)}")
        if job_data.get('alternate_education'):
            text_parts.append(
                f"Alternate Education: {safe_strip(job_data['alternate_education'], 'alternate_education', row_idx)}")
        if job_data.get('required_experience'):
            text_parts.append(
                f"Required Experience: {safe_strip(job_data['required_experience'], 'required_experience', row_idx)}")
        if job_data.get('location'):
            text_parts.append(f"Location: {safe_strip(job_data['location'], 'location', row_idx)}")

        return " ".join(text_parts)

    def _create_job_skills_text(self, job_data, row_idx=None):
        """Create searchable text from job skills data"""
        skills_parts = []

        # Get special skills and alternate special skills
        if job_data.get('special_skills'):
            skills_parts.append(safe_strip(job_data['special_skills'], 'special_skills', row_idx))
        if job_data.get('alternate_special_skills'):
            skills_parts.append(safe_strip(job_data['alternate_special_skills'], 'alternate_special_skills', row_idx))

        # Only join non-empty strings
        skills_parts = [part for part in skills_parts if part and part.strip()]

        if skills_parts:
            return f"Skills: {' '.join(skills_parts)}"
        return ""

    def _create_pwd_text(self, pwd, row_idx=None):
        """Create searchable text from PWD record"""
        text_parts = []

        if pwd.get('F.a.1'):
            text_parts.append(f"Job Title: {safe_strip(pwd['F.a.1'], 'F.a.1', row_idx)}")

        # Combine F.a.2 with Addendum1 for job description
        job_desc_parts = []
        if pwd.get('F.a.2'):
            job_desc_parts.append(safe_strip(pwd['F.a.2'], 'F.a.2', row_idx))
        if pwd.get('Addendum1'):
            job_desc_parts.append(safe_strip(pwd['Addendum1'], 'Addendum1', row_idx))

        if job_desc_parts:
            text_parts.append(f"Description: {' '.join(job_desc_parts)}")

        required_education = self._get_education_level(pwd, 'required')
        if required_education:
            text_parts.append(f"Required Education: {required_education}")

        alternate_education = self._get_education_level(pwd, 'alternate')
        if alternate_education:
            text_parts.append(f"Alternate Education: {alternate_education}")

        if pwd.get('F.b.4.a'):
            text_parts.append(f"Required Experience: {safe_strip(pwd['F.b.4.a'], 'F.b.4.a', row_idx)}")

        # Use F.e.3 and F.e.4 for location instead of F.e.1
        location_parts = []
        if pwd.get('F.e.3'):
            location_parts.append(safe_strip(pwd['F.e.3'], 'F.e.3', row_idx))
        if pwd.get('F.e.4'):
            location_parts.append(safe_strip(pwd['F.e.4'], 'F.e.4', row_idx))

        if location_parts:
            text_parts.append(f"Location: {' '.join(location_parts)}")

        return " ".join(text_parts)

    def _create_pwd_skills_text(self, pwd, row_idx=None):
        """Create searchable text from PWD skills data"""
        skills_parts = []

        # Safely handle Addendum4
        addendum4 = pwd.get('Addendum4')
        if addendum4 is not None and addendum4 != False and safe_strip(addendum4, 'Addendum4', row_idx):
            skills_parts.append(safe_strip(addendum4, 'Addendum4', row_idx))

        # Safely handle Addendum6
        addendum6 = pwd.get('Addendum6')
        if addendum6 is not None and addendum6 != False and safe_strip(addendum6, 'Addendum6', row_idx):
            skills_parts.append(safe_strip(addendum6, 'Addendum6', row_idx))

        # Only join non-empty strings
        skills_parts = [part for part in skills_parts if part]

        if skills_parts:
            return f"Skills: {' '.join(skills_parts)}"
        return ""

    def _get_education_level(self, pwd, education_type='required'):
        """Extract education level from PWD record
        
        Args:
            pwd: The PWD record
            education_type: 'required' for F.b.1.* fields or 'alternate' for F.c.2.* fields
        """
        if education_type == 'required':
            education_fields = {
                'F.b.1.Doctorate': 'Doctorate',
                'F.b.1.Masters': 'Masters',
                'F.b.1.Bachelors': 'Bachelors',
                'F.b.1.Associates': 'Associates',
                'F.b.1.HighSchoolGED': 'High School/GED',
                'F.b.1.None': 'None',
                'F.b.1.OtherDegree': 'Other Degree'
            }
        else:  # alternate
            education_fields = {
                'F.c.2.Doctorate': 'Doctorate',
                'F.c.2.Masters': 'Masters',
                'F.c.2.Bachelors': 'Bachelors',
                'F.c.2.Associates': 'Associates',
                'F.c.2.HighSchoolGED': 'High School/GED',
                'F.c.2.None': 'None',
                'F.c.2.OtherDegree': 'Other Degree'
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

    def _calculate_education_similarity(self, job_data, pwd):
        """Calculate education similarity with hierarchical comparison"""
        # Define education hierarchy (higher index = higher education)
        education_hierarchy = {
            'None': 0,
            'High School/GED': 1,
            'Associates': 2,
            'Bachelors': 3,
            'Masters': 4,
            'Doctorate': 5,
            'Other Degree': 2.5  # Place between Associates and Bachelors
        }
        
        def get_education_score(job_edu, pwd_edu):
            if not job_edu or not pwd_edu:
                return None
            
            job_level = education_hierarchy.get(job_edu, 0)
            pwd_level = education_hierarchy.get(pwd_edu, 0)
            
            if job_level == pwd_level:
                return 1.0  # Exact match
            elif pwd_level > job_level:
                return 0.9  # Higher education than required
            else:
                # Lower education than required
                diff = job_level - pwd_level
                if diff == 1:
                    return 0.7
                elif diff == 2:
                    return 0.4
                elif diff == 3:
                    return 0.2
                else:
                    return 0.1
        
        # Get education data
        job_required_edu = job_data.get('required_education', '').strip()
        job_alternate_edu = job_data.get('alternate_education', '').strip()
        pwd_required_edu = self._get_education_level(pwd, 'required')
        pwd_alternate_edu = self._get_education_level(pwd, 'alternate')
        
        # Debug logging
        logger.info(f"Education similarity calculation:")
        logger.info(f"  Job required education: '{job_required_edu}'")
        logger.info(f"  Job alternate education: '{job_alternate_edu}'")
        logger.info(f"  PWD required education: '{pwd_required_edu}'")
        logger.info(f"  PWD alternate education: '{pwd_alternate_edu}'")
        
        scores = []
        
        # Required to Required comparison
        if job_required_edu and pwd_required_edu:
            req_score = get_education_score(job_required_edu, pwd_required_edu)
            if req_score is not None:
                scores.append(req_score)
                logger.info(f"  Required-to-required score: {req_score}")
        
        # Alternate to Alternate comparison (only if both job and PWD have alternate)
        if job_alternate_edu and pwd_alternate_edu:
            alt_score = get_education_score(job_alternate_edu, pwd_alternate_edu)
            if alt_score is not None:
                scores.append(alt_score)
                logger.info(f"  Alternate-to-alternate score: {alt_score}")
        
        if not scores:
            logger.info(f"  No education scores calculated - returning None")
            return None
        
        final_score = sum(scores) / len(scores)
        logger.info(f"  Final education score: {final_score}")
        
        # If we have both required and alternate scores, average them
        # Otherwise use the single available score
        return final_score
    
    def _calculate_experience_similarity(self, job_data, pwd, row_idx):
        """Calculate experience similarity using semantic similarity"""
        job_required_exp = job_data.get('required_experience', '').strip()
        job_alternate_exp = job_data.get('alternate_experience', '').strip()
        pwd_required_exp = safe_strip(pwd.get('F.b.4.a', ''), 'F.b.4.a', row_idx)
        pwd_alternate_exp = safe_strip(pwd.get('F.c.4.a', ''), 'F.c.4.a', row_idx)
        
        scores = []
        
        # Required to Required comparison
        if job_required_exp and pwd_required_exp:
            try:
                job_emb = self.model.encode([job_required_exp], show_progress_bar=False)
                pwd_emb = self.model.encode([pwd_required_exp], show_progress_bar=False)
                similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
                scores.append(float(similarity))
            except Exception as e:
                logger.error(f"Experience similarity calculation failed: {e}")
        
        # Alternate to Alternate comparison
        if job_alternate_exp and pwd_alternate_exp:
            try:
                job_emb = self.model.encode([job_alternate_exp], show_progress_bar=False)
                pwd_emb = self.model.encode([pwd_alternate_exp], show_progress_bar=False)
                similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
                scores.append(float(similarity))
            except Exception as e:
                logger.error(f"Alternate experience similarity calculation failed: {e}")
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def _calculate_occupation_similarity(self, job_data, pwd, row_idx):
        """Calculate occupation requirements similarity using semantic similarity"""
        job_occupation = job_data.get('occupation_requirement', '').strip()
        
        # Combine F.b.4.b with Addendum3 (checking both possible column names)
        pwd_occupation_parts = []
        if pwd.get('F.b.4.b'):
            pwd_occupation_parts.append(safe_strip(pwd['F.b.4.b'], 'F.b.4.b', row_idx))
        
        # Check for Addendum3 or Addendum_F.b.4.b
        addendum_field = pwd.get('Addendum3') or pwd.get('Addendum_F.b.4.b')
        if addendum_field:
            pwd_occupation_parts.append(safe_strip(addendum_field, 'Addendum3/Addendum_F.b.4.b', row_idx))
        
        pwd_occupation = ' '.join(pwd_occupation_parts) if pwd_occupation_parts else ''
        
        if not job_occupation or not pwd_occupation:
            return None
        
        try:
            job_emb = self.model.encode([job_occupation], show_progress_bar=False)
            pwd_emb = self.model.encode([pwd_occupation], show_progress_bar=False)
            similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Occupation similarity calculation failed: {e}")
            return None
    
    def _calculate_skills_similarity(self, job_data, pwd, row_idx):
        """Calculate skills similarity using semantic similarity"""
        job_required_skills = job_data.get('special_skills', '').strip()
        job_alternate_skills = job_data.get('alternate_special_skills', '').strip()
        
        # Get PWD skills from Addendum4 and Addendum6 (matching _create_pwd_skills_text)
        pwd_required_skills = safe_strip(pwd.get('Addendum4', ''), 'Addendum4', row_idx)
        pwd_alternate_skills = safe_strip(pwd.get('Addendum6', ''), 'Addendum6', row_idx)
        
        scores = []
        
        # Required to Required comparison
        if job_required_skills and pwd_required_skills:
            try:
                job_emb = self.model.encode([job_required_skills], show_progress_bar=False)
                pwd_emb = self.model.encode([pwd_required_skills], show_progress_bar=False)
                similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
                scores.append(float(similarity))
            except Exception as e:
                logger.error(f"Skills similarity calculation failed: {e}")
        
        # Alternate to Alternate comparison
        if job_alternate_skills and pwd_alternate_skills:
            try:
                job_emb = self.model.encode([job_alternate_skills], show_progress_bar=False)
                pwd_emb = self.model.encode([pwd_alternate_skills], show_progress_bar=False)
                similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
                scores.append(float(similarity))
            except Exception as e:
                logger.error(f"Alternate skills similarity calculation failed: {e}")
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def _calculate_job_description_similarity(self, job_data, pwd, row_idx):
        """Calculate job description similarity (title + description)"""
        job_parts = []
        if job_data.get('job_title'):
            job_parts.append(job_data['job_title'].strip())
        if job_data.get('job_description'):
            job_parts.append(job_data['job_description'].strip())
        
        pwd_parts = []
        if pwd.get('F.a.1'):
            pwd_parts.append(safe_strip(pwd['F.a.1'], 'F.a.1', row_idx))
        
        # Combine F.a.2 with Addendum_F.a.2
        if pwd.get('F.a.2'):
            pwd_parts.append(safe_strip(pwd['F.a.2'], 'F.a.2', row_idx))
        if pwd.get('Addendum_F.a.2'):
            pwd_parts.append(safe_strip(pwd['Addendum_F.a.2'], 'Addendum_F.a.2', row_idx))
        
        job_text = ' '.join(job_parts) if job_parts else ''
        pwd_text = ' '.join(pwd_parts) if pwd_parts else ''
        
        if not job_text or not pwd_text:
            return None
        
        try:
            job_emb = self.model.encode([job_text], show_progress_bar=False)
            pwd_emb = self.model.encode([pwd_text], show_progress_bar=False)
            similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Job description similarity calculation failed: {e}")
            return None
    
    def _calculate_location_similarity(self, job_data, pwd, row_idx):
        """Calculate location similarity using fuzzy text matching"""
        job_location = job_data.get('location', '').strip().lower()
        
        # Use F.e.3 and F.e.4 for PWD location
        pwd_location_parts = []
        if pwd.get('F.e.3'):
            pwd_location_parts.append(safe_strip(pwd['F.e.3'], 'F.e.3', row_idx))
        if pwd.get('F.e.4'):
            pwd_location_parts.append(safe_strip(pwd['F.e.4'], 'F.e.4', row_idx))
        
        pwd_location = ' '.join(pwd_location_parts).strip().lower() if pwd_location_parts else ''
        
        if not job_location or not pwd_location:
            return None
        
        try:
            job_emb = self.model.encode([job_location], show_progress_bar=False)
            pwd_emb = self.model.encode([pwd_location], show_progress_bar=False)
            similarity = cosine_similarity(job_emb, pwd_emb)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Location similarity calculation failed: {e}")
            return None

    def _calculate_experience_similarity_basic(self, job_data, pwd, row_idx):
        """Calculate experience similarity using basic text matching"""
        job_required_exp = job_data.get('required_experience', '').strip().lower()
        job_alternate_exp = job_data.get('alternate_experience', '').strip().lower()
        pwd_required_exp = safe_strip(pwd.get('F.b.4.a', ''), 'F.b.4.a', row_idx).lower()
        pwd_alternate_exp = safe_strip(pwd.get('F.c.4.a', ''), 'F.c.4.a', row_idx).lower()
        
        def jaccard_similarity(text1, text2):
            if not text1 or not text2:
                return 0.0
            words1 = set(text1.split())
            words2 = set(text2.split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        scores = []
        
        # Required to Required comparison
        if job_required_exp and pwd_required_exp:
            scores.append(jaccard_similarity(job_required_exp, pwd_required_exp))
        
        # Alternate to Alternate comparison
        if job_alternate_exp and pwd_alternate_exp:
            scores.append(jaccard_similarity(job_alternate_exp, pwd_alternate_exp))
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def _calculate_occupation_similarity_basic(self, job_data, pwd, row_idx):
        """Calculate occupation requirements similarity using basic text matching"""
        job_occupation = job_data.get('occupation_requirement', '').strip().lower()
        
        # Combine F.b.4.b with Addendum3
        pwd_occupation_parts = []
        if pwd.get('F.b.4.b'):
            pwd_occupation_parts.append(safe_strip(pwd['F.b.4.b'], 'F.b.4.b', row_idx))
        
        addendum_field = pwd.get('Addendum3') or pwd.get('Addendum_F.b.4.b')
        if addendum_field:
            pwd_occupation_parts.append(safe_strip(addendum_field, 'Addendum3/Addendum_F.b.4.b', row_idx))
        
        pwd_occupation = ' '.join(pwd_occupation_parts).strip().lower() if pwd_occupation_parts else ''
        
        if not job_occupation or not pwd_occupation:
            return None
        
        job_words = set(job_occupation.split())
        pwd_words = set(pwd_occupation.split())
        intersection = len(job_words.intersection(pwd_words))
        union = len(job_words.union(pwd_words))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_skills_similarity_basic(self, job_data, pwd, row_idx):
        """Calculate skills similarity using basic text matching"""
        job_required_skills = job_data.get('special_skills', '').strip().lower()
        job_alternate_skills = job_data.get('alternate_special_skills', '').strip().lower()
        pwd_required_skills = safe_strip(pwd.get('Addendum4', ''), 'Addendum4', row_idx).lower()
        pwd_alternate_skills = safe_strip(pwd.get('Addendum6', ''), 'Addendum6', row_idx).lower()
        
        def jaccard_similarity(text1, text2):
            if not text1 or not text2:
                return 0.0
            words1 = set(text1.split())
            words2 = set(text2.split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        scores = []
        
        # Required to Required comparison
        if job_required_skills and pwd_required_skills:
            scores.append(jaccard_similarity(job_required_skills, pwd_required_skills))
        
        # Alternate to Alternate comparison
        if job_alternate_skills and pwd_alternate_skills:
            scores.append(jaccard_similarity(job_alternate_skills, pwd_alternate_skills))
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def _calculate_job_description_similarity_basic(self, job_data, pwd, row_idx):
        """Calculate job description similarity using basic text matching"""
        job_parts = []
        if job_data.get('job_title'):
            job_parts.append(job_data['job_title'].strip())
        if job_data.get('job_description'):
            job_parts.append(job_data['job_description'].strip())
        
        pwd_parts = []
        if pwd.get('F.a.1'):
            pwd_parts.append(safe_strip(pwd['F.a.1'], 'F.a.1', row_idx))
        if pwd.get('F.a.2'):
            pwd_parts.append(safe_strip(pwd['F.a.2'], 'F.a.2', row_idx))
        if pwd.get('Addendum_F.a.2'):
            pwd_parts.append(safe_strip(pwd['Addendum_F.a.2'], 'Addendum_F.a.2', row_idx))
        
        job_text = ' '.join(job_parts).strip().lower() if job_parts else ''
        pwd_text = ' '.join(pwd_parts).strip().lower() if pwd_parts else ''
        
        if not job_text or not pwd_text:
            return None
        
        job_words = set(job_text.split())
        pwd_words = set(pwd_text.split())
        intersection = len(job_words.intersection(pwd_words))
        union = len(job_words.union(pwd_words))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_location_similarity_basic(self, job_data, pwd, row_idx):
        """Calculate location similarity using basic text matching"""
        job_location = job_data.get('location', '').strip().lower()
        
        pwd_location_parts = []
        if pwd.get('F.e.3'):
            pwd_location_parts.append(safe_strip(pwd['F.e.3'], 'F.e.3', row_idx))
        if pwd.get('F.e.4'):
            pwd_location_parts.append(safe_strip(pwd['F.e.4'], 'F.e.4', row_idx))
        
        pwd_location = ' '.join(pwd_location_parts).strip().lower() if pwd_location_parts else ''
        
        if not job_location or not pwd_location:
            return None
        
        job_words = set(job_location.split())
        pwd_words = set(pwd_location.split())
        intersection = len(job_words.intersection(pwd_words))
        union = len(job_words.union(pwd_words))
        return intersection / union if union > 0 else 0.0

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
            'required_education': request.form.get('required_education', ''),
            'alternate_education': request.form.get('alternate_education', ''),
            'required_experience': request.form.get('required_experience', ''),
            'alternate_experience': request.form.get('alternate_experience', ''),
            'occupation_requirement': request.form.get('occupation_requirement', ''),
            'special_skills': request.form.get('special_skills', ''),
            'alternate_special_skills': request.form.get('alternate_special_skills', ''),
            'location': request.form.get('location', ''),
            'company': request.form.get('company', ''),
            'salary_range': request.form.get('salary_range', '')
        }

        # Log skills data for debugging
        if job_data['special_skills'] or job_data['alternate_special_skills']:
            logger.info(
                f"Skills data provided - Special Skills: '{job_data['special_skills']}', Alternate Skills: '{job_data['alternate_special_skills']}'")

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
                embeddings = model.encode(test_text, show_progress_bar=False)
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
