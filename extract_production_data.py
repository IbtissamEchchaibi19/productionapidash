import os
import re
import pandas as pd
import fitz  # PyMuPDF
from datetime import datetime
import glob
import csv
import camelot
import numpy as np

class HoneyProductionExtractor:
    def __init__(self):
        # Enhanced patterns with more variations and debugging
        self.patterns = {
            'batch_number': [
                r'Batch[:\s]*([A-Z0-9]+)',
                r'Report[:\s]*No[.:]?\s*([A-Z0-9]+)',
                r'SIDR\s*([A-Z0-9]+)',
                r'SDR\s*([A-Z0-9]+)',
                r'([A-Z]{2,4}\d{3,6})',  # Pattern like ABC123, SIDR2024
                r'Batch\s*Number[.:]?\s*([A-Z0-9]+)',
                r'ID[.:]?\s*([A-Z0-9]+)',
            ],
            'report_title': [
                r'SIDR\s+(\d{4})\s*-\s*([^-\n]+)',
                r'SDR\s+(\d{4})\s*-\s*([^-\n]+)',
                r'Report\s+(\d{4})\s*-\s*([^-\n]+)',
                r'Production\s+Report\s+(\d{4})\s*-\s*([^-\n]+)',
            ],
            'company_name': [
                r'([A-Z\s]+LLC)',
                r'([A-Z\s]+L\.L\.C)',
                r'MANAHIL\s+LLC',
                r'Company[:\s]+([A-Z\s]+)',
                r'([A-Z\s]{5,}\s+COMPANY)',
            ],
            'year': [
                r'SIDR\s+(\d{4})',
                r'SDR\s+(\d{4})',
                r'Report\s+(\d{4})',
                r'Year[:\s]+(\d{4})',
                r'(\d{4})',  # Any 4-digit year
            ],
            'report_type': [
                r'APIARIES\s+HONEY\s+PRODUCTION\s+MAP',
                r'HONEY\s+PRODUCTION\s+REPORT',
                r'PRODUCTION\s+SUMMARY',
                r'PRODUCTION\s+REPORT',
            ],
        }
        
        # Enhanced table headers
        self.table_headers = [
            'Apiary Number', 'Gross weight', 'Drum weight', 'Net weight', 
            'Beshara', 'Production', 'No. of Production Hives', 
            'Production per hive', 'No. of Hive Supers', 'Harvest Date',
            'Apiary', 'Gross', 'Drum', 'Net', 'Hives', 'Supers'
        ]
        
        # Enhanced UAE locations
        self.uae_locations = [
            "Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", 
            "Fujairah", "Ras Al Khaimah", "Kalba", "Masafi", "Hatta",
            "Khor Fakkan", "Al farfar", "UAQ", "RAK", "Dibba", "Taweeh",
            "Casablanca", "Rabat", "Marrakech", "Fes", "Tangier", "Agadir", 
            "Oujda", "Meknes", "Taza", "Al Ain", "Liwa", "Ruwais"
        ]
        
        # Output CSV fields
        self.csv_fields = [
            'batch_number', 'report_year', 'company_name', 'apiary_number', 
            'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
            'beshara_kg', 'production_kg', 'num_production_hives', 
            'production_per_hive_kg', 'num_hive_supers', 'harvest_date',
            'efficiency_ratio', 'waste_percentage', 'extraction_date'
        ]

    def extract_text_from_pdf(self, pdf_path):
        """Enhanced text extraction with better error handling"""
        text = ""
        try:
            print(f"DEBUG: Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            print(f"DEBUG: PDF has {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n--- PAGE {page_num + 1} ---\n"
                text += page_text
                print(f"DEBUG: Page {page_num + 1} extracted {len(page_text)} characters")
                
            doc.close()
            print(f"DEBUG: Total extracted text length: {len(text)}")
            return text
        except Exception as e:
            print(f"ERROR: Failed to extract text from {pdf_path}: {e}")
            return ""

    def extract_header_info(self, text):
        """Enhanced header extraction with better debugging"""
        header_info = {}
        
        print(f"DEBUG: Extracting header info...")
        print(f"DEBUG: Text sample (first 1000 chars):\n{text[:1000]}")
        
        for field, patterns in self.patterns.items():
            found = False
            
            for pattern in patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        if field == 'report_title':
                            if len(match.groups()) >= 2:
                                header_info['year'] = match.group(1)
                                header_info['title'] = match.group(2).strip()
                            else:
                                header_info[field] = match.group(0).strip()
                        elif field == 'report_type':
                            header_info[field] = match.group(0).strip()
                        else:
                            if len(match.groups()) >= 1:
                                header_info[field] = match.group(1).strip()
                            else:
                                header_info[field] = match.group(0).strip()
                        
                        print(f"DEBUG: Found {field} = '{header_info[field]}' using pattern '{pattern}'")
                        found = True
                        break
                        
                except Exception as e:
                    print(f"ERROR: Pattern '{pattern}' for field '{field}': {e}")
                    continue
            
            if not found:
                header_info[field] = None
                print(f"DEBUG: Could not find {field}")
                
        # Enhanced fallback extraction
        if not header_info.get('year'):
            year_matches = re.findall(r'20\d{2}', text)
            if year_matches:
                header_info['year'] = year_matches[0]
                print(f"DEBUG: Fallback year found: {header_info['year']}")
        
        if not header_info.get('batch_number'):
            # Try to find any alphanumeric code that looks like a batch
            batch_matches = re.findall(r'\b[A-Z]{2,4}\d{3,6}\b', text)
            if batch_matches:
                header_info['batch_number'] = batch_matches[0]
                print(f"DEBUG: Fallback batch found: {header_info['batch_number']}")
                
        return header_info

    def extract_location_from_text(self, text):
        """Enhanced location extraction"""
        text_lower = text.lower()
        for location in self.uae_locations:
            if location.lower() in text_lower:
                return location
        
        # Try partial matches for compound location names
        for location in self.uae_locations:
            parts = location.lower().split()
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 3 and part in text_lower:
                        return location
        
        return "Unknown"

    def clean_numeric_value(self, value):
        """Enhanced numeric cleaning with better validation"""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['nan', 'n/a', '-', 'null']:
            return None
        
        # Remove all non-numeric characters except decimal points and negative signs
        cleaned = re.sub(r'[^\d.-]', '', value_str)
        
        if not cleaned or cleaned in ['-', '.', '-.']:
            return None
        
        try:
            result = float(cleaned)
            # Validate reasonable ranges
            if abs(result) > 100000:  # Very large number check
                print(f"WARNING: Unusually large number: {result} from '{value_str}'")
            if result < 0:
                print(f"WARNING: Negative number: {result} from '{value_str}'")
            return result
        except (ValueError, TypeError):
            print(f"WARNING: Could not convert '{value_str}' to number")
            return None

    def parse_harvest_date(self, date_str):
        """Enhanced date parsing with more formats"""
        if not date_str or pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Enhanced date patterns
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})',
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',
            r'(\d{4})/(\d{1,2})/(\d{1,2})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})',
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if pattern.startswith(r'(\d{4})'):
                        year, month, day = match.groups()
                    else:
                        day, month, year = match.groups()
                    
                    year = int(year)
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                    
                    month, day = int(month), int(day)
                    
                    if not (1 <= month <= 12 and 1 <= day <= 31):
                        continue
                    
                    parsed_date = datetime(year, month, day)
                    return parsed_date.strftime('%Y-%m-%d')
                    
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Date parsing failed for {date_str}: {e}")
                    continue
        
        print(f"WARNING: Could not parse date: '{date_str}'")
        return None

    def parse_data_line(self, line):
        """Enhanced data line parsing with better number extraction"""
        line = line.strip()
        if not line:
            return None
            
        print(f"DEBUG: Parsing line: '{line}'")
        
        # Skip unwanted lines
        skip_keywords = [
            'net stock', 'net beshara', 'net production', 'total', 'average', 
            'cleaning', 'summary', 'header', 'apiary number', 'gross weight',
            'drum weight', 'net weight', 'production', 'harvest date'
        ]
        
        if any(word in line.lower() for word in skip_keywords):
            print(f"DEBUG: Skipping line: '{line}'")
            return None
        
        # Enhanced number extraction - look for decimal numbers
        numbers = re.findall(r'\d+\.?\d*', line)
        print(f"DEBUG: Found numbers: {numbers}")
        
        if len(numbers) < 3:  # Need at least 3 numbers
            print(f"DEBUG: Not enough numbers in line: {len(numbers)}")
            return None
        
        # Extract apiary information
        apiary_number = ""
        location = "Unknown"
        
        # Look for apiary patterns
        apiary_patterns = [
            r'^(\d+)\s*-\s*([A-Za-z\s]+)',  # "47 - Taweeh"
            r'^(\d+)\s+([A-Za-z\s]+)',      # "47 Taweeh"
            r'^([A-Za-z\s]+\d+)',           # "Apiary47"
        ]
        
        for pattern in apiary_patterns:
            match = re.match(pattern, line.strip())
            if match:
                if len(match.groups()) >= 2:
                    apiary_number = f"{match.group(1)} - {match.group(2).strip()}"
                    location = self.extract_location_from_text(match.group(2))
                else:
                    apiary_number = match.group(1).strip()
                    location = self.extract_location_from_text(apiary_number)
                break
        
        if not apiary_number:
            # Try to extract location names directly
            for loc in self.uae_locations:
                if loc.lower() in line.lower():
                    location = loc
                    apiary_number = f"Unknown - {loc}"
                    break
        
        # Extract harvest date
        harvest_date = None
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})',
        ]
        
        for date_pattern in date_patterns:
            date_match = re.search(date_pattern, line)
            if date_match:
                harvest_date = self.parse_harvest_date(date_match.group(1))
                break
        
        try:
            # Map numbers to fields with validation
            result = {
                'apiary_number': apiary_number if apiary_number else f"Unknown-{numbers[0] if numbers else 'X'}",
                'location': location,
                'gross_weight_kg': self.clean_numeric_value(numbers[0]) if len(numbers) > 0 else None,
                'drum_weight_kg': self.clean_numeric_value(numbers[1]) if len(numbers) > 1 else None,
                'net_weight_kg': self.clean_numeric_value(numbers[2]) if len(numbers) > 2 else None,
                'beshara_kg': self.clean_numeric_value(numbers[3]) if len(numbers) > 3 else None,
                'production_kg': self.clean_numeric_value(numbers[4]) if len(numbers) > 4 else None,
                'num_production_hives': self.clean_numeric_value(numbers[5]) if len(numbers) > 5 else None,
                'production_per_hive_kg': self.clean_numeric_value(numbers[6]) if len(numbers) > 6 else None,
                'num_hive_supers': self.clean_numeric_value(numbers[7]) if len(numbers) > 7 else None,
                'harvest_date': harvest_date
            }
            
            # Validate that we have at least some meaningful data
            numeric_fields = ['gross_weight_kg', 'net_weight_kg', 'production_kg']
            if all(result.get(field) is None for field in numeric_fields):
                print(f"DEBUG: No meaningful numeric data found in line")
                return None
            
            print(f"DEBUG: Successfully parsed: {result}")
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to parse line '{line}': {e}")
            return None

    def extract_production_table(self, pdf_path):
        """Enhanced table extraction with multiple approaches"""
        production_rows = []
        
        try:
            # Method 1: Text-based extraction
            print(f"DEBUG: Starting text extraction for {pdf_path}")
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            
            print(f"DEBUG: Processing {len(full_text)} characters of text")
            
            # Process lines
            lines = full_text.split('\n')
            data_section = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for table start indicators
                table_indicators = ['apiary', 'production', 'weight', 'harvest', 'hive']
                if any(indicator in line.lower() for indicator in table_indicators):
                    if len([ind for ind in table_indicators if ind in line.lower()]) >= 2:
                        data_section = True
                        print(f"DEBUG: Found table header at line {i}: '{line}'")
                        continue
                
                if not data_section:
                    continue
                
                # Try to parse data line
                parsed_row = self.parse_data_line(line)
                if parsed_row:
                    production_rows.append(parsed_row)
            
            print(f"DEBUG: Text extraction found {len(production_rows)} rows")
            
            # Method 2: Camelot extraction (fallback)
            if len(production_rows) < 1:
                print(f"DEBUG: Trying Camelot extraction...")
                try:
                    # Try lattice first
                    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                    if len(tables) == 0:
                        # Try stream
                        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                    
                    print(f"DEBUG: Camelot found {len(tables)} tables")
                    
                    for table_idx, table in enumerate(tables):
                        df_table = table.df
                        print(f"DEBUG: Table {table_idx} shape: {df_table.shape}")
                        
                        if df_table.shape[1] >= 3:  # At least 3 columns
                            for idx, row in df_table.iterrows():
                                row_text = ' '.join([str(val).strip() for val in row.values if str(val).strip()])
                                if len(row_text) > 10:  # Meaningful row
                                    parsed_row = self.parse_data_line(row_text)
                                    if parsed_row:
                                        production_rows.append(parsed_row)
                                        
                except Exception as e:
                    print(f"DEBUG: Camelot extraction failed: {e}")
            
            # Method 3: Structured text search (last resort)
            if len(production_rows) < 1:
                print(f"DEBUG: Trying structured text search...")
                # Look for lines with multiple numbers
                for line in lines:
                    line = line.strip()
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if len(numbers) >= 5:  # Line with many numbers might be data
                        parsed_row = self.parse_data_line(line)
                        if parsed_row:
                            production_rows.append(parsed_row)
                            
        except Exception as e:
            print(f"ERROR: Table extraction failed for {pdf_path}: {e}")
            
        print(f"DEBUG: Final extraction result: {len(production_rows)} rows")
        return production_rows

    def calculate_additional_metrics(self, row):
        """Calculate additional metrics with null checks"""
        try:
            # Efficiency ratio
            if (row.get('production_per_hive_kg') and 
                row.get('net_weight_kg') and 
                row['net_weight_kg'] > 0):
                row['efficiency_ratio'] = (row['production_per_hive_kg'] / row['net_weight_kg']) * 100
            else:
                row['efficiency_ratio'] = None
                
            # Waste percentage
            if (row.get('beshara_kg') and 
                row.get('gross_weight_kg') and 
                row['gross_weight_kg'] > 0):
                row['waste_percentage'] = (row['beshara_kg'] / row['gross_weight_kg']) * 100
            else:
                row['waste_percentage'] = None
                
        except Exception as e:
            print(f"ERROR: Metric calculation failed: {e}")
            row['efficiency_ratio'] = None
            row['waste_percentage'] = None
            
        return row

    def process_report(self, pdf_path):
        """Enhanced report processing with better error handling"""
        results = []
        
        try:
            print(f"DEBUG: ===== Processing {pdf_path} =====")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                raise Exception("Failed to extract text from PDF")
            
            # Extract header info
            header_info = self.extract_header_info(text)
            print(f"DEBUG: Header extracted: {header_info}")
            
            # Extract production data
            production_data = self.extract_production_table(pdf_path)
            print(f"DEBUG: Production data extracted: {len(production_data)} rows")
            
            # If no data found, create minimal record
            if not production_data:
                print(f"WARNING: No production data found, creating minimal record")
                production_data = [{
                    'apiary_number': header_info.get('batch_number', 'Unknown'),
                    'location': 'Unknown',
                    'gross_weight_kg': None,
                    'drum_weight_kg': None,
                    'net_weight_kg': None,
                    'beshara_kg': None,
                    'production_kg': None,
                    'num_production_hives': None,
                    'production_per_hive_kg': None,
                    'num_hive_supers': None,
                    'harvest_date': None
                }]
            
            # Create final records
            for production in production_data:
                row = {
                    'batch_number': header_info.get('batch_number') or f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'report_year': header_info.get('year') or datetime.now().year,
                    'company_name': header_info.get('company_name') or 'MANAHIL LLC',
                    'apiary_number': production.get('apiary_number') or 'Unknown',
                    'location': production.get('location') or 'Unknown',
                    'gross_weight_kg': production.get('gross_weight_kg'),
                    'drum_weight_kg': production.get('drum_weight_kg'),
                    'net_weight_kg': production.get('net_weight_kg'),
                    'beshara_kg': production.get('beshara_kg'),
                    'production_kg': production.get('production_kg'),
                    'num_production_hives': production.get('num_production_hives'),
                    'production_per_hive_kg': production.get('production_per_hive_kg'),
                    'num_hive_supers': production.get('num_hive_supers'),
                    'harvest_date': production.get('harvest_date'),
                    'extraction_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                # Calculate metrics
                row = self.calculate_additional_metrics(row)
                results.append(row)
                print(f"DEBUG: Created final row: {row}")
            
            print(f"SUCCESS: Extracted {len(results)} records from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            print(f"ERROR: Processing failed for {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error record
            results = [{
                'batch_number': f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'report_year': datetime.now().year,
                'company_name': 'Unknown',
                'apiary_number': f"Error_{os.path.basename(pdf_path)}",
                'location': 'Unknown',
                'gross_weight_kg': None,
                'drum_weight_kg': None,
                'net_weight_kg': None,
                'beshara_kg': None,
                'production_kg': None,
                'num_production_hives': None,
                'production_per_hive_kg': None,
                'num_hive_supers': None,
                'harvest_date': None,
                'efficiency_ratio': None,
                'waste_percentage': None,
                'extraction_date': datetime.now().strftime('%Y-%m-%d')
            }]
        
        return results

    def process_all_reports(self, directory):
        """Process all PDF reports with progress tracking"""
        all_results = []
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        
        total_files = len(pdf_files)
        print(f"Found {total_files} PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n{'='*50}")
            print(f"Processing [{i+1}/{total_files}]: {os.path.basename(pdf_file)}")
            print(f"{'='*50}")
            
            try:
                report_data = self.process_report(pdf_file)
                all_results.extend(report_data)
                print(f"✓ Success: {len(report_data)} records added")
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
            
        return all_results

    def save_to_csv(self, data, output_path):
        """Save data to CSV with proper handling"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
                writer.writeheader()
                for row in data:
                    clean_row = {field: row.get(field, '') for field in self.csv_fields}
                    writer.writerow(clean_row)
            
            print(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return None

    def create_dataframe(self, data):
        """Create DataFrame with proper type conversion"""
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        numeric_cols = [
            'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg', 'beshara_kg',
            'production_kg', 'num_production_hives', 'production_per_hive_kg',
            'num_hive_supers', 'efficiency_ratio', 'waste_percentage'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['extraction_date', 'harvest_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df

def main():
    """Main function for testing"""
    reports_dir = "honey_production_reports"
    output_csv = "honey_production_data.csv"
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created directory: {reports_dir}")
        print("Please place your PDF files in this directory and run the script again.")
        return None
    
    extractor = HoneyProductionExtractor()
    all_data = extractor.process_all_reports(reports_dir)
    
    if not all_data:
        print("No data extracted. Please check your PDF files.")
        return None
    
    extractor.save_to_csv(all_data, output_csv)
    df = extractor.create_dataframe(all_data)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total records: {len(df)}")
    print(f"Unique batches: {df['batch_number'].nunique()}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Records with production data: {df['production_kg'].notna().sum()}")
    
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nSample Data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()