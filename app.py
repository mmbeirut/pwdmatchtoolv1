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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SERVER_NAME = "agd-vtanc-2016"
DATABASE_NAME = "ImmApps"
TABLE_NAME = "DOL_9141_form_20260731_allClients"

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Initialize sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {e}")
    model = None

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
            
            # Base query
            query = f"SELECT * FROM [{TABLE_NAME}] WHERE [Case Status] = 'Certified'"
            
            # Add filters if provided
            if filters:
                filter_conditions = []
                
                if filters.get('companies'):
                    company_list = "', '".join(filters['companies'])
                    filter_conditions.append(f"[C.1] IN ('{company_list}')")
                
                if filters.get('locations'):
                    location_list = "', '".join(filters['locations'])
                    filter_conditions.append(f"[F.e.1] IN ('{location_list}')")
                
                if filters.get('job_titles'):
                    title_list = "', '".join(filters['job_titles'])
                    filter_conditions.append(f"[F.a.1] IN ('{title_list}')")
                
                if filter_conditions:
                    query += " AND " + " AND ".join(filter_conditions)
            
            df = pd.read_sql(query, engine)
            logger.info(f"Retrieved {len(df)} PWD records")
            return df
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return pd.DataFrame()
    
    def get_filter_options(self):
        """Get unique values for filter dropdowns"""
        try:
            engine = self.get_engine()
            
            # Get unique companies
            companies_query = f"SELECT DISTINCT [C.1] as company FROM [{TABLE_NAME}] WHERE [C.1] IS NOT NULL AND [Case Status] = 'Certified' ORDER BY [C.1]"
            companies = pd.read_sql(companies_query, engine)['company'].tolist()
            
            # Get unique locations
            locations_query = f"SELECT DISTINCT [F.e.1] as location FROM [{TABLE_NAME}] WHERE [F.e.1] IS NOT NULL AND [Case Status] = 'Certified' ORDER BY [F.e.1]"
            locations = pd.read_sql(locations_query, engine)['location'].tolist()
            
            # Get unique job titles
            titles_query = f"SELECT DISTINCT [F.a.1] as title FROM [{TABLE_NAME}] WHERE [F.a.1] IS NOT NULL AND [Case Status] = 'Certified' ORDER BY [F.a.1]"
            titles = pd.read_sql(titles_query, engine)['title'].tolist()
            
            return {
                'companies': companies,
                'locations': locations,
                'job_titles': titles
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
        if not self.model or pwd_records.empty:
            return []
        
        try:
            # Create job description text
            job_text = self._create_job_text(job_data)
            
            # Create PWD texts
            pwd_texts = []
            for _, pwd in pwd_records.iterrows():
                pwd_text = self._create_pwd_text(pwd)
                pwd_texts.append(pwd_text)
            
            if not pwd_texts:
                return []
            
            # Calculate embeddings
            job_embedding = self.model.encode([job_text])
            pwd_embeddings = self.model.encode(pwd_texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(job_embedding, pwd_embeddings)[0]
            
            # Create results with similarity scores
            results = []
            for i, (_, pwd) in enumerate(pwd_records.iterrows()):
                similarity_score = float(similarities[i])
                match_strength = self._determine_match_strength(similarity_score)
                
                results.append({
                    'pwd_case_number': pwd.get('PWD Case Number', ''),
                    'company': pwd.get('C.1', ''),
                    'job_title': pwd.get('F.a.1', ''),
                    'job_location': pwd.get('F.e.1', ''),
                    'job_description': pwd.get('F.a.2', ''),
                    'education_required': self._get_education_level(pwd),
                    'experience_required': pwd.get('F.b.4.a', ''),
                    'similarity_score': similarity_score,
                    'match_strength': match_strength,
                    'wage_info': self._get_wage_info(pwd)
                })
            
            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return []
    
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
        if job_data.get('skills'):
            text_parts.append(f"Skills: {job_data['skills']}")
        if job_data.get('location'):
            text_parts.append(f"Location: {job_data['location']}")
        
        return " ".join(text_parts)
    
    def _create_pwd_text(self, pwd):
        """Create searchable text from PWD record"""
        text_parts = []
        
        if pwd.get('F.a.1'):
            text_parts.append(f"Job Title: {pwd['F.a.1']}")
        if pwd.get('F.a.2'):
            text_parts.append(f"Description: {pwd['F.a.2']}")
        if pwd.get('F.a.2.addendum'):
            text_parts.append(f"Additional Description: {pwd['F.a.2.addendum']}")
        
        education = self._get_education_level(pwd)
        if education:
            text_parts.append(f"Education: {education}")
        
        if pwd.get('F.b.4.a'):
            text_parts.append(f"Experience: {pwd['F.b.4.a']}")
        if pwd.get('F.b.4.b'):
            text_parts.append(f"Skills: {pwd['F.b.4.b']}")
        if pwd.get('F.e.1'):
            text_parts.append(f"Location: {pwd['F.e.1']}")
        
        return " ".join(text_parts)
    
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
        
        # Get wage amount and period
        for period in ['Hour', 'Week', 'BiWeekly', 'Month', 'Year']:
            field = f'G.4.a.{period}'
            if pwd.get(field):
                wage_info['amount'] = pwd[field]
                wage_info['period'] = period
                break
        
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
            'experience_required': request.form.get('experience_required', ''),
            'skills': request.form.get('skills', ''),
            'location': request.form.get('location', ''),
            'company': request.form.get('company', ''),
            'salary_range': request.form.get('salary_range', '')
        }
        
        # Get filters
        filters = {
            'companies': request.form.getlist('filter_companies'),
            'locations': request.form.getlist('filter_locations'),
            'job_titles': request.form.getlist('filter_job_titles')
        }
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v}
        
        # Get PWD records
        pwd_records = db_manager.get_pwds(filters)
        
        if pwd_records.empty:
            return jsonify({
                'success': True,
                'results': [],
                'message': 'No PWD records found matching your criteria.'
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
                    'match_strength': 'Unknown'
                })
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
