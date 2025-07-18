# PWD Match Tool v5 - Application Documentation

## Overview
The PWD Match Tool v5 is a Flask-based web application designed for immigration law firms to compare job descriptions with existing Prevailing Wage Determinations (PWDs) stored in a SQL Server database. The application uses advanced semantic similarity matching to help identify relevant PWD cases for new job positions.

## Core Functionality

### 1. Job Data Input and Comparison
**Primary Function**: Users can input comprehensive job details and receive ranked matches against existing PWD records.

**Input Fields Available**:
- Job Title: Position name/title
- Job Description: Detailed description of job duties and responsibilities
- Required Education: Minimum education level required
- Alternate Education: Alternative acceptable education level
- Required Experience: Minimum work experience requirements
- Alternate Experience: Alternative acceptable experience
- Occupation Requirements: Specific occupational qualifications needed
- Special Skills: Required technical or specialized skills
- Alternate Special Skills: Alternative acceptable skills
- Location: Job location (city, state)
- Company: Employer company name
- Salary Range: Expected compensation range

### 2. Advanced Filtering System
**Filter Options**:
- Companies: Filter by specific employer organizations
- Locations: Filter by geographic locations (city, state combinations)
- Job Titles: Filter by specific position titles
- Case Statuses: Filter by PWD application status (e.g., "Determination Issued")

**Filter Behavior**: Users can apply multiple filters simultaneously to narrow search results before similarity matching begins.

### 3. Multi-Factor Similarity Scoring
**Semantic Similarity Engine**: The application employs a sophisticated scoring system that evaluates six distinct categories:

**Category Weights**:
- Education Similarity: 15% weight
- Experience Similarity: 15% weight  
- Occupation Requirements: 15% weight
- Skills Matching: 20% weight
- Job Description: 15% weight
- Location Matching: 20% weight

**Education Matching Logic**:
- Hierarchical comparison (None < High School/GED < Associates < Bachelors < Masters < Doctorate)
- Exact matches score 1.0
- Higher education than required scores 0.9
- Lower education receives graduated penalties (0.7, 0.4, 0.2, 0.1)
- Compares both required and alternate education paths

**Experience Matching**: Semantic comparison of experience descriptions using sentence transformers or Jaccard similarity fallback.

**Skills Analysis**: Compares required and alternate special skills using advanced text similarity algorithms.

**Location Matching**: Geographic similarity scoring between job location and PWD work locations.

### 4. Intelligent Model Loading System
**Primary Model**: Attempts to load SentenceTransformer 'all-MiniLM-L6-v2' for semantic similarity
**Loading Sequence**:
1. Local model directory check
2. Cache directory search
3. HuggingFace cache directory scan
4. BasicTransformer fallback creation

**Fallback Mechanism**: If sentence transformers fail, creates BasicTransformer using simple text features (word count, character count, unique words, average word length, sentence approximation).

### 5. Database Integration
**Connection**: Connects to SQL Server database "ImmApps" on server "agd-vtanc-2016"
**Table**: Queries "DOL_9141_form_20260731_allClients" table
**Query Optimization**: Dynamic query building with conditional WHERE clauses based on applied filters

### 6. Results Processing and Display
**Match Strength Classification**:
- Strong: ≥80% similarity
- Moderate: 60-79% similarity  
- Weak: 40-59% similarity
- Very Weak: <40% similarity

**Result Limiting**: Returns top 50 matches to prevent performance issues
**Sorting**: Results ranked by combined similarity score (highest first)

### 7. Comprehensive Result Data
**Each Result Contains**:
- PWD Case Number
- Company name
- Job title and location
- Complete job description (including addendums)
- Education requirements (required and alternate)
- Experience requirements (required and alternate)
- Occupation requirements (including addendums)
- Special skills (required and alternate)
- Individual category similarity scores
- Combined similarity score
- Match strength classification
- Wage information and potential wage issues
- Case status
- ONET occupation code
- Validity period
- Travel requirements

### 8. Wage Analysis
**Wage Comparison**: Compares user-provided salary range against PWD wage determinations
**Wage Sources**: Identifies wage source (OES All Industries, OES ACWIA, CBA, DBA, SCA)
**Issue Detection**: Flags potential wage issues when job salary is below PWD minimum

### 9. API Endpoints
**Main Routes**:
- `/` - Main application interface with job input form
- `/search` (POST) - Processes job data and returns similarity matches
- `/health` - Application health check and database connectivity test
- `/debug/data` - Database diagnostic information and record counts
- `/debug/model` - Model loading status and configuration details

### 10. Error Handling and Logging
**Comprehensive Logging**: Tracks model loading, database queries, similarity calculations, and errors
**Graceful Degradation**: Falls back to basic text matching if advanced models fail
**Error Recovery**: Continues operation even with partial data or model failures

### 11. Security and Configuration
**Environment Variables**: Supports configurable secret keys and database connections
**SSL Handling**: Comprehensive SSL bypass for model downloading in corporate environments
**Offline Mode**: Supports offline operation with cached models

### 12. Data Safety Features
**Safe String Processing**: Handles null values and data type inconsistencies gracefully
**Input Validation**: Processes various data formats and handles missing fields
**Memory Management**: Limits result sets and optimizes query performance

## Technical Architecture
**Framework**: Flask web application
**Database**: SQL Server with SQLAlchemy ORM
**ML Libraries**: SentenceTransformers, scikit-learn for similarity calculations
**Frontend**: HTML templates with dynamic filtering and result display
**Deployment**: Configurable for development and production environments

This application serves as a comprehensive tool for immigration law professionals to efficiently identify relevant PWD precedents for new job positions, significantly reducing manual research time while providing quantitative similarity assessments.
