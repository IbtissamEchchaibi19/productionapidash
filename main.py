from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional, Union
import pandas as pd
import uvicorn
import json
import hashlib
import traceback
from fastapi.middleware.wsgi import WSGIMiddleware

# Import your existing classes
from extract_production_data import HoneyProductionExtractor  # Your existing extractor
from github_storage import GitHubConfig, GitHubCSVStorage  # Your GitHub storage classes

# DASHBOARD IMPORT - UNCHANGED
try:
    from dashboard import app as dash_app
    DASHBOARD_AVAILABLE = True
    print("✓ Dashboard app imported successfully")
except ImportError as e:
    print(f"✗ Dashboard import error: {e}")
    print("Creating a dummy dashboard app...")
    
    try:
        import dash
        from dash import html
        
        dash_app = dash.Dash(__name__)
        dash_app.layout = html.Div([
            html.H1("Dashboard Not Available"),
            html.P("The dashboard module could not be imported.")
        ])
        DASHBOARD_AVAILABLE = False
    except:
        dash_app = None
        DASHBOARD_AVAILABLE = False
except Exception as e:
    print(f"✗ Dashboard import failed: {e}")
    dash_app = None
    DASHBOARD_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Honey Production Processing API", version="1.0.0")
if dash_app:
    app.mount("/dash_app", WSGIMiddleware(dash_app.server))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic models for request/response
class DeleteReportRequest(BaseModel):
    report_ids: List[str]

class DeleteReportResponse(BaseModel):
    message: str
    deleted_reports: List[dict]
    not_found_reports: List[str]
    total_deleted: int
    remaining_records: int
    errors: Optional[List[str]] = None

# Global variables
REPORTS_DIR = "honey_production_reports"
PROCESSED_FILES_TRACKER = "processed_production_files.json"
extractor = HoneyProductionExtractor()

# Initialize GitHub CSV storage
try:
    github_config = GitHubConfig()
    github_storage = github_config.get_storage_instance()
    print("✓ GitHub CSV storage initialized successfully")
    print(f"✓ Repository: {github_config.repo_owner}/{github_config.repo_name}")
    print(f"✓ CSV file: {github_config.csv_filename}")
except Exception as e:
    print(f"✗ GitHub storage initialization failed: {e}")
    print("Please check your .env file contains:")
    print("- GITHUB_REPO_OWNER")
    print("- GITHUB_REPO_NAME") 
    print("- GITHUB_TOKEN")
    print("- GITHUB_CSV_FILENAME (optional)")
    raise

# Global variables for dashboard
dash_thread = None

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# CSV column structure
CSV_COLUMNS = [
    'batch_number', 'report_year', 'company_name', 'apiary_number', 
    'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
    'beshara_kg', 'production_kg', 'num_production_hives', 
    'production_per_hive_kg', 'num_hive_supers', 'harvest_date',
    'efficiency_ratio', 'waste_percentage', 'extraction_date'
]

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file to track if it's been processed"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error generating hash for {file_path}: {e}")
        return ""

def load_processed_files_tracker() -> dict:
    """Load the tracker of processed files"""
    if os.path.exists(PROCESSED_FILES_TRACKER):
        try:
            with open(PROCESSED_FILES_TRACKER, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processed files tracker: {e}")
            return {}
    return {}

def save_processed_files_tracker(tracker: dict):
    """Save the tracker of processed files"""
    try:
        with open(PROCESSED_FILES_TRACKER, 'w') as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files tracker: {e}")

def is_file_processed(file_path: str) -> bool:
    """Check if a file has already been processed"""
    try:
        if not os.path.exists(file_path):
            return False
            
        tracker = load_processed_files_tracker()
        filename = os.path.basename(file_path)
        
        if filename not in tracker:
            return False
        
        current_hash = get_file_hash(file_path)
        if not current_hash:
            return False
            
        return tracker[filename] == current_hash
        
    except Exception as e:
        print(f"Error checking if file processed {file_path}: {e}")
        return False

def mark_file_as_processed(file_path: str):
    """Mark a file as processed in the tracker"""
    try:
        if not os.path.exists(file_path):
            return
            
        tracker = load_processed_files_tracker()
        filename = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)
        
        if file_hash:
            tracker[filename] = file_hash
            save_processed_files_tracker(tracker)
            print(f"Marked {filename} as processed")
    except Exception as e:
        print(f"Error marking file as processed {file_path}: {e}")

def remove_file_from_tracker(filename: str):
    """Remove a file from the processed files tracker"""
    try:
        tracker = load_processed_files_tracker()
        if filename in tracker:
            del tracker[filename]
            save_processed_files_tracker(tracker)
            print(f"Removed {filename} from processed files tracker")
    except Exception as e:
        print(f"Error removing {filename} from tracker: {e}")

def initialize_github_csv_if_needed():
    """Initialize GitHub CSV file with proper column structure if it doesn't exist"""
    try:
        # Check if CSV exists in GitHub
        content, sha = github_storage.get_file_content()
        
        if content is None:
            print("CSV file not found in GitHub repository, creating new one...")
            # Create empty DataFrame with all expected columns
            df = pd.DataFrame(columns=CSV_COLUMNS)
            csv_content = df.to_csv(index=False)
            
            success = github_storage.upload_csv_content(
                csv_content, 
                commit_message="Initialize honey production CSV with column structure"
            )
            
            if success:
                print(f"✓ Initialized GitHub CSV with {len(CSV_COLUMNS)} columns")
            else:
                raise Exception("Failed to initialize CSV in GitHub")
        else:
            print("✓ GitHub CSV file already exists")
            
    except Exception as e:
        print(f"Error initializing GitHub CSV: {e}")
        raise

def ensure_data_consistency(data_dict: dict) -> dict:
    """Ensure all required columns are present with proper default values"""
    consistent_data = {}
    
    for column in CSV_COLUMNS:
        if column in data_dict and data_dict[column] is not None:
            # Convert to appropriate type and handle NaN
            value = data_dict[column]
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                consistent_data[column] = None
            else:
                consistent_data[column] = value
        else:
            # Set appropriate default for missing columns
            if column in ['batch_number', 'company_name', 'apiary_number', 'location']:
                consistent_data[column] = 'Unknown'
            elif column == 'report_year':
                consistent_data[column] = 2024
            elif column == 'extraction_date':
                consistent_data[column] = pd.Timestamp.now().strftime('%Y-%m-%d')
            else:
                consistent_data[column] = None
    
    return consistent_data

def append_to_github_csv(new_data: List[dict]):
    """Append new data to GitHub CSV with proper column alignment and validation"""
    if not new_data:
        return
    
    try:
        print(f"DEBUG: Appending {len(new_data)} records to GitHub CSV")
        
        # Ensure data consistency for all records
        consistent_data = []
        for record in new_data:
            consistent_record = ensure_data_consistency(record)
            consistent_data.append(consistent_record)
            print(f"DEBUG: Processed record - batch: {consistent_record.get('batch_number')}, location: {consistent_record.get('location')}, production: {consistent_record.get('production_kg')}")
        
        # Use GitHub storage to append data
        success = github_storage.append_data_to_csv(consistent_data)
        
        if success:
            print(f"SUCCESS: Added {len(new_data)} records to GitHub CSV")
        else:
            raise Exception("Failed to append data to GitHub CSV")
            
    except Exception as e:
        print(f"ERROR: Failed to append to GitHub CSV: {e}")
        traceback.print_exc()
        raise

def get_report_records_by_ids(report_ids: List[str]) -> tuple:
    """Get report records by their IDs from GitHub CSV"""
    try:
        df = github_storage.read_csv_as_dataframe()
        
        if df is None or df.empty or 'batch_number' not in df.columns:
            return [], report_ids
        
        # Convert report_ids to strings for comparison
        report_ids_str = [str(id) for id in report_ids]
        
        # Find matching records
        matching_records = df[df['batch_number'].astype(str).isin(report_ids_str)]
        found_batch_numbers = matching_records['batch_number'].astype(str).unique().tolist()
        
        # Determine which IDs were not found
        not_found_ids = [id for id in report_ids_str if id not in found_batch_numbers]
        
        # Convert matching records to list of dictionaries
        found_records = matching_records.to_dict('records')
        
        return found_records, not_found_ids
        
    except Exception as e:
        print(f"Error getting report records from GitHub: {e}")
        return [], report_ids

def delete_records_from_github_csv(report_ids: List[str]) -> int:
    """Delete records from GitHub CSV by batch numbers"""
    try:
        df = github_storage.read_csv_as_dataframe()
        
        if df is None or df.empty:
            return 0
        
        original_count = len(df)
        
        # Filter out records with matching batch numbers
        df_filtered = df[~df['batch_number'].astype(str).isin([str(id) for id in report_ids])]
        deleted_count = original_count - len(df_filtered)
        
        # Update the entire CSV in GitHub
        success = github_storage.update_entire_csv(
            df_filtered, 
            commit_message=f"Delete {deleted_count} production records"
        )
        
        if success:
            print(f"Deleted {deleted_count} records from GitHub CSV")
            return deleted_count
        else:
            raise Exception("Failed to update GitHub CSV after deletion")
        
    except Exception as e:
        print(f"Error deleting records from GitHub CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating GitHub CSV: {str(e)}")

def delete_pdf_files(filenames: List[str]) -> List[str]:
    """Delete PDF files from the reports directory"""
    deleted_files = []
    
    for filename in filenames:
        file_path = os.path.join(REPORTS_DIR, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(filename)
                print(f"Deleted file: {filename}")
                
                # Remove from processed files tracker
                remove_file_from_tracker(filename)
            else:
                print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
    
    return deleted_files

# ROUTES
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Honey Production Processing API with GitHub Storage",
        "endpoints": {
            "upload_reports": "/upload-reports/",
            "delete_reports": "/delete-reports/",
            "get_reports": "/reports/",
            "get_data": "/data/",
            "dashboard": "/dashboard/",
            "health": "/health/"
        },
        "storage": {
            "type": "GitHub",
            "repository": f"{github_config.repo_owner}/{github_config.repo_name}",
            "csv_file": github_config.csv_filename,
            "raw_csv_url": github_storage.get_raw_csv_url()
        }
    }

@app.get("/dashboard/")
async def get_dashboard():
    """Redirect directly to the dashboard"""
    return RedirectResponse(url="")

@app.delete("/delete-reports/")
async def delete_reports(request: DeleteReportRequest):
    """Delete one or multiple production reports by their batch numbers"""
    if not request.report_ids:
        raise HTTPException(status_code=400, detail="No report IDs provided")
    
    try:
        # Get report records that match the IDs from GitHub
        found_records, not_found_ids = get_report_records_by_ids(request.report_ids)
        
        if not found_records:
            raise HTTPException(
                status_code=404, 
                detail=f"No reports found with IDs: {', '.join(request.report_ids)}"
            )
        
        deleted_reports = []
        errors = []
        
        # Delete records from GitHub CSV
        try:
            deleted_count = delete_records_from_github_csv(request.report_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting from GitHub CSV: {str(e)}")
        
        # Extract filenames from the found records and delete PDF files
        filenames_to_delete = []
        for record in found_records:
            batch_number = record.get('batch_number', 'unknown')
            potential_filename = f"{batch_number}.pdf"
            filenames_to_delete.append(potential_filename)
        
        # Delete PDF files
        deleted_files = delete_pdf_files(filenames_to_delete)
        
        # Prepare response data
        for record in found_records:
            batch_number = str(record.get('batch_number', 'unknown'))
            deleted_reports.append({
                "batch_number": batch_number,
                "apiary_number": record.get('apiary_number', 'unknown'),
                "location": record.get('location', 'unknown'),
                "deleted_from_csv": True,
                "deleted_pdf_file": f"{batch_number}.pdf" in deleted_files
            })
        
        # Get remaining record count from GitHub
        remaining_records = 0
        try:
            df = github_storage.read_csv_as_dataframe()
            if df is not None:
                remaining_records = len(df)
        except Exception as e:
            print(f"Error counting remaining records: {e}")
        
        return DeleteReportResponse(
            message=f"Successfully deleted {len(deleted_reports)} report(s) from GitHub",
            deleted_reports=deleted_reports,
            not_found_reports=not_found_ids,
            total_deleted=len(deleted_reports),
            remaining_records=remaining_records,
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting reports: {str(e)}")

@app.get("/reports/")
async def list_reports():
    """List all production reports from GitHub CSV"""
    try:
        df = github_storage.read_csv_as_dataframe()
        
        if df is None or df.empty:
            return {
                "message": "No production report data found in GitHub",
                "reports": [],
                "total_count": 0
            }
        
        # Group by batch_number to get unique reports
        if 'batch_number' not in df.columns:
            return {
                "message": "CSV structure error - missing batch_number column",
                "reports": [],
                "total_count": 0
            }
        
        unique_reports = df.groupby('batch_number').agg({
            'report_year': 'first' if 'report_year' in df.columns else lambda x: 'N/A',
            'company_name': 'first' if 'company_name' in df.columns else lambda x: 'N/A',
            'location': lambda x: ', '.join(x.unique()) if 'location' in df.columns else 'N/A',
            'apiary_number': 'count',
            'production_kg': 'sum' if 'production_kg' in df.columns else lambda x: 0,
            'extraction_date': 'first' if 'extraction_date' in df.columns else lambda x: 'N/A'
        }).reset_index()
        
        reports = []
        for _, row in unique_reports.iterrows():
            report_info = {
                "batch_number": str(row['batch_number']),
                "report_year": row.get('report_year', 'N/A'),
                "company_name": row.get('company_name', 'N/A'),
                "locations": row.get('location', 'N/A'),
                "total_apiaries": row.get('apiary_number', 0),
                "total_production_kg": row.get('production_kg', 0),
                "extraction_date": row.get('extraction_date', 'N/A')
            }
            reports.append(report_info)
        
        return {
            "message": f"Found {len(reports)} unique report(s) in GitHub",
            "reports": reports,
            "total_count": len(reports),
            "total_apiary_records": len(df),
            "github_csv_url": github_storage.get_raw_csv_url()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing reports from GitHub: {str(e)}")
# Add this enhanced debugging to your FastAPI upload endpoint

@app.post("/upload-reports/")
async def upload_reports(files: Union[List[UploadFile], UploadFile] = File(...)):
    """Upload and process honey production report PDFs with enhanced debugging"""
    # Convert single file to list for uniform processing
    if isinstance(files, UploadFile):
        files = [files]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    total_new_records = 0
    errors = []
    all_new_data = []
    skipped_files = []
    debug_info = []  # Add debug information
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            errors.append(f"{file.filename}: Only PDF files are allowed")
            continue
        
        file_path = os.path.join(REPORTS_DIR, file.filename)
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Add file size and validation check
            file_size = os.path.getsize(file_path)
            debug_info.append({
                "filename": file.filename,
                "file_size_bytes": file_size,
                "file_path": file_path,
                "file_exists": os.path.exists(file_path)
            })
            
            print(f"DEBUG: File {file.filename} saved - Size: {file_size} bytes")
            
            # Check if this file has already been processed
            if is_file_processed(file_path):
                skipped_files.append({
                    "filename": file.filename,
                    "reason": "Already processed (no changes detected)"
                })
                continue
            
            # Process the new/changed file with detailed debugging
            print(f"Processing new/changed file: {file.filename}")
            try:
                # Add detailed debugging before extraction
                print(f"DEBUG: About to call extractor.process_report() for {file.filename}")
                print(f"DEBUG: File path exists: {os.path.exists(file_path)}")
                print(f"DEBUG: File is readable: {os.access(file_path, os.R_OK)}")
                
                # Try to validate PDF file
                try:
                    with open(file_path, 'rb') as pdf_file:
                        header = pdf_file.read(10)
                        is_valid_pdf = header.startswith(b'%PDF-')
                        print(f"DEBUG: PDF header validation: {is_valid_pdf}, Header: {header}")
                except Exception as header_error:
                    print(f"DEBUG: Error reading PDF header: {header_error}")
                
                # Call the extractor with error handling
                report_data = extractor.process_report(file_path)
                
                print(f"DEBUG: Extractor returned: {type(report_data)}")
                print(f"DEBUG: Number of records: {len(report_data) if report_data else 0}")
                
                # Debug each record in detail
                if report_data:
                    for i, record in enumerate(report_data):
                        print(f"DEBUG: Record {i+1}: {record}")
                        # Check for None values
                        none_fields = [k for k, v in record.items() if v is None]
                        non_none_fields = [k for k, v in record.items() if v is not None]
                        print(f"DEBUG: Record {i+1} - None fields: {none_fields}")
                        print(f"DEBUG: Record {i+1} - Non-None fields: {non_none_fields}")
                
                if report_data and len(report_data) > 0:
                    print(f"DEBUG: Extractor returned {len(report_data)} records for {file.filename}")
                    
                    # Validate and clean the data before adding
                    valid_records = []
                    for record in report_data:
                        # Debug the record structure
                        print(f"DEBUG: Raw record: {record}")
                        
                        # Check if record has meaningful data
                        meaningful_fields = 0
                        for key, value in record.items():
                            if value is not None and value != '' and str(value).lower() != 'nan':
                                meaningful_fields += 1
                        
                        print(f"DEBUG: Record has {meaningful_fields} meaningful fields out of {len(record)} total fields")
                        
                        # Ensure the record has all required fields
                        clean_record = ensure_data_consistency(record)
                        valid_records.append(clean_record)
                        
                        print(f"DEBUG: Cleaned record - batch: {clean_record.get('batch_number')}, production: {clean_record.get('production_kg')}")
                    
                    all_new_data.extend(valid_records)
                    mark_file_as_processed(file_path)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "records_added": len(valid_records)
                    })
                    total_new_records += len(valid_records)
                    print(f"SUCCESS: Processed {file.filename}: {len(valid_records)} valid records")
                else:
                    errors.append(f"{file.filename}: No data extracted from PDF")
                    print(f"WARNING: No data extracted from {file.filename}")
                    
                    # Add debug info for failed extraction
                    debug_info.append({
                        "filename": file.filename,
                        "extraction_result": "empty",
                        "extractor_response": str(report_data)
                    })
                    
            except Exception as e:
                error_msg = f"{file.filename}: Error processing PDF - {str(e)}"
                errors.append(error_msg)
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                
                # Add detailed error info
                debug_info.append({
                    "filename": file.filename,
                    "extraction_result": "error",
                    "error_message": str(e),
                    "error_traceback": traceback.format_exc()
                })
                
        except Exception as e:
            error_msg = f"{file.filename}: Error saving file - {str(e)}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            
            # Clean up file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    # Append all new data to GitHub CSV
    if all_new_data:
        try:
            print(f"DEBUG: About to append {len(all_new_data)} records to GitHub CSV")
            append_to_github_csv(all_new_data)
            print(f"SUCCESS: Added {len(all_new_data)} total records to GitHub CSV")
        except Exception as e:
            error_msg = f"Error saving data to GitHub CSV: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
    
    # Get current total records from GitHub
    total_records = 0
    try:
        df = github_storage.read_csv_as_dataframe()
        if df is not None:
            total_records = len(df)
            print(f"DEBUG: GitHub CSV now contains {total_records} total records")
    except Exception as e:
        print(f"Error reading GitHub CSV for count: {e}")
    
    return {
        "message": f"Upload completed. Processed {len(processed_files)} new/changed files",
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "total_new_records": total_new_records,
        "total_records": total_records,
        "storage_location": "GitHub",
        "github_csv_url": github_storage.get_raw_csv_url(),
        "errors": errors if errors else None,
        "debug_info": debug_info  # Include debug information in response
    }



@app.get("/data/")
async def get_data():
    """Get all production data from GitHub CSV"""
    try:
        df = github_storage.read_csv_as_dataframe()
        
        if df is None or df.empty:
            return {
                "message": "No data available in GitHub CSV",
                "data": [],
                "total_records": 0
            }
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        return {
            "message": f"Retrieved {len(data)} records from GitHub CSV",
            "data": data,
            "total_records": len(data),
            "unique_batches": df['batch_number'].nunique() if 'batch_number' in df.columns else 0,
            "github_csv_url": github_storage.get_raw_csv_url()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data from GitHub: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    try:
        dashboard_status = "running" if dash_thread and dash_thread.is_alive() else "stopped"
        
        # Get processing status
        pdf_count = 0
        processed_count = 0
        unprocessed_count = 0
        
        try:
            if os.path.exists(REPORTS_DIR):
                pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
                pdf_count = len(pdf_files)
                
                for filename in pdf_files:
                    file_path = os.path.join(REPORTS_DIR, filename)
                    if is_file_processed(file_path):
                        processed_count += 1
                    else:
                        unprocessed_count += 1
                        
        except Exception as e:
            print(f"Warning: Could not get processing status: {e}")
        
        # Check GitHub CSV status
        github_csv_status = "unknown"
        csv_records = 0
        try:
            content, sha = github_storage.get_file_content()
            if content is not None:
                github_csv_status = "exists"
                df = github_storage.read_csv_as_dataframe()
                if df is not None:
                    csv_records = len(df)
            else:
                github_csv_status = "not_found"
        except Exception as e:
            github_csv_status = f"error: {str(e)}"
            print(f"Warning: Could not check GitHub CSV status: {e}")
        
        return {
            "status": "healthy",
            "message": "Honey Production Processing API with GitHub Storage is running",
            "dashboard_status": dashboard_status,
            "dashboard_available": DASHBOARD_AVAILABLE,
            "github_csv_status": github_csv_status,
            "csv_records": csv_records,
            "github_repository": f"{github_config.repo_owner}/{github_config.repo_name}",
            "github_csv_filename": github_config.csv_filename,
            "github_csv_url": github_storage.get_raw_csv_url(),
            "reports_directory_exists": os.path.exists(REPORTS_DIR),
            "total_pdf_files": pdf_count,
            "processed_files": processed_count,
            "unprocessed_files": unprocessed_count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

# Initialize GitHub CSV on startup
@app.on_event("startup")
async def startup_event():
    """Initialize GitHub CSV file on application startup"""
    try:
        initialize_github_csv_if_needed()
        print("✓ GitHub CSV initialization completed")
    except Exception as e:
        print(f"✗ GitHub CSV initialization failed: {e}")
        # Don't raise here - let the app start but log the error

if __name__ == "__main__":
    print("Starting Honey Production Processing API with GitHub Storage...")
    print("API will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("Dashboard will be available at: http://localhost:8051")
    print(f"GitHub Repository: {github_config.repo_owner}/{github_config.repo_name}")
    print(f"GitHub CSV: {github_config.csv_filename}")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)