# PWD Match Tool v5

An interactive web application for immigration law firms to compare job descriptions with existing Prevailing Wage Determinations (PWDs) stored in a SQL Server database.

## Features

- **Job Comparison**: Compare job descriptions against existing PWD records
- **AI-Powered Matching**: Uses sentence transformers for semantic similarity analysis
- **Multi-Filter Search**: Filter results by company, location, job title, and skills
- **Match Strength Analysis**: Provides similarity scores and match strength ratings
- **Clean Modern UI**: Bootstrap-based interface with firm branding colors
- **Database Integration**: Direct connection to SQL Server with trusted authentication

## Technology Stack

- **Backend**: Python Flask
- **Database**: SQL Server with SQLAlchemy ORM
- **AI/ML**: Sentence Transformers for semantic matching
- **Frontend**: Bootstrap 5, jQuery
- **Styling**: Custom CSS with #002856 primary color scheme

## Prerequisites

- Python 3.8 or higher
- SQL Server with ODBC Driver 17
- Access to the ImmApps database on agd-vtanc-2016 server
- Windows environment (for trusted connection)

## Installation

1. **Clone or download the application files**

2. **Install Python dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Verify database access**:
   - Ensure you have access to server: `agd-vtanc-2016`
   - Database: `ImmApps`
   - Table: `DOL_9141_form_20260731_allClients`
   - Trusted connection should work with your Windows credentials

4. **Test the installation**:
   ```cmd
   python app.py
   ```

## Running the Application

1. **Start the Flask server**:
   ```cmd
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Check system health** (optional):
   ```
   http://localhost:5000/health
   ```

## Usage

### Basic Job Matching

1. **Enter Job Information**:
   - Job Title (required)
   - Job Description (required)
   - Education Level
   - Experience Required
   - Skills
   - Location
   - Company Name
   - Salary Range

2. **Apply Filters** (optional):
   - Filter by specific companies
   - Filter by job locations
   - Filter by job titles
   - Use Ctrl/Cmd to select multiple options

3. **Search and Review Results**:
   - Results are sorted by similarity score
   - Match strength indicators: Strong, Moderate, Weak, Very Weak
   - Similarity scores shown as percentages
   - PWD case numbers and details provided

### Understanding Results

- **Strong Match (80%+)**: High similarity, likely good PWD match
- **Moderate Match (60-79%)**: Reasonable similarity, review carefully
- **Weak Match (40-59%)**: Low similarity, may need modifications
- **Very Weak Match (<40%)**: Poor similarity, likely not suitable

## Database Schema

The application connects to the `DOL_9141_form_20260731_allClients` table with the following key fields:

- `PWD Case Number`: Unique PWD identifier
- `C.1`: Company name
- `F.a.1`: Job title
- `F.a.2`: Job description
- `F.e.1`: Job location
- `F.b.1.*`: Education requirements
- `F.b.4.*`: Experience and skills requirements
- `G.4.*`: Wage information

## Configuration

### Database Settings
```python
SERVER_NAME = "agd-vtanc-2016"
DATABASE_NAME = "ImmApps"
TABLE_NAME = "DOL_9141_form_20260731_allClients"
```

### AI Model
The application uses the `all-MiniLM-L6-v2` sentence transformer model for semantic similarity analysis. This model is downloaded automatically on first run.

## Troubleshooting

### Database Connection Issues
- Verify SQL Server is accessible
- Check Windows authentication/trusted connection
- Ensure ODBC Driver 17 for SQL Server is installed

### Model Loading Issues
- First run may take time to download the AI model
- Ensure internet connection for initial model download
- Check available disk space (model requires ~100MB)

### Performance Issues
- Large result sets are limited to top 50 matches
- Consider adding more specific filters for better performance
- Database queries are optimized for certified PWDs only

## Development

### Project Structure
```
pwd-match-tool/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template with styling
│   ├── index.html        # Main search interface
│   └── error.html        # Error page template
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Key Components

1. **DatabaseManager**: Handles SQL Server connections and queries
2. **PWDMatcher**: Manages AI-powered similarity calculations
3. **Flask Routes**: Web interface and API endpoints

### Adding Features

- Modify `app.py` for backend functionality
- Update templates for UI changes
- Add new routes for additional features
- Extend database queries in DatabaseManager class

## Security Notes

- Uses trusted Windows authentication (no credentials in code)
- No cloud AI services (complies with firm policy)
- Local sentence transformer model only
- Input validation and SQL injection protection via SQLAlchemy

## Support

For technical issues or questions:
1. Check the system health endpoint: `/health`
2. Review application logs in the console
3. Verify database connectivity
4. Ensure all dependencies are installed correctly

## License

Internal use only - Immigration Law Firm proprietary application.
