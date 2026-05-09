import requests
import PyPDF2
import pandas as pd
import re
import io
from typing import List, Dict
import os

class GroundwaterDataConverter:
    def __init__(self):
        self.pdf_urls = [
            'https://cgwb.gov.in/sites/default/files/inline-files/january_wl_1994-2024-compressed.pdf',
            'https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_1994-2003.pdf',
            'https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_2004-2013.pdf',
            'https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_2014-2024.pdf',
            'https://cgwb.gov.in/sites/default/files/inline-files/august_wl_1994-2023_compressed.pdf',
            'https://cgwb.gov.in/sites/default/files/inline-files/post-monsoon_wl_1994-2023_compressed.pdf'
        ]
        self.all_data = []

    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        print(f"Downloading: {url.split('/')[-1]}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def determine_period(self, filename: str) -> str:
        """Determine monitoring period from filename"""
        filename_lower = filename.lower()
        if 'january' in filename_lower:
            return 'January'
        elif 'pre-monsoon' in filename_lower:
            return 'Pre-Monsoon (April-May)'
        elif 'august' in filename_lower:
            return 'August'
        elif 'post-monsoon' in filename_lower:
            return 'Post-Monsoon (November)'
        return 'Unknown'

    def extract_year_range(self, filename: str) -> tuple:
        """Extract year range from filename"""
        match = re.search(r'(\d{4})-(\d{4})', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def parse_groundwater_data(self, text: str, filename: str) -> List[Dict]:
        """Parse groundwater data from extracted text"""
        data = []
        period = self.determine_period(filename)
        start_year, end_year = self.extract_year_range(filename)
        
        lines = text.split('\n')
        
        # Pattern to match common groundwater data formats
        # Looking for: State, District, Location, Depth values
        depth_pattern = r'(\d+\.?\d*)\s*(?:m|meter|metre|M)'
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for depth measurements
            depth_matches = re.findall(depth_pattern, line, re.IGNORECASE)
            
            if depth_matches:
                # Try to get context from previous and current lines
                context_lines = []
                for j in range(max(0, i-2), i+1):
                    if j < len(lines):
                        context_lines.append(lines[j].strip())
                
                location = ' '.join(context_lines[:2]) if len(context_lines) >= 2 else line
                
                for depth in depth_matches:
                    try:
                        depth_value = float(depth)
                        data.append({
                            'Period': period,
                            'Start_Year': start_year if start_year else 'N/A',
                            'End_Year': end_year if end_year else 'N/A',
                            'Location': location[:200],  # Limit length
                            'Water_Level_Depth_m': depth_value,
                            'Source_File': filename,
                            'Raw_Text': line[:300]  # Keep some context
                        })
                    except ValueError:
                        continue
        
        return data

    def process_local_files(self, file_paths: List[str]):
        """Process locally downloaded PDF files"""
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Processing local file: {file_path}")
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'rb') as f:
                    pdf_content = f.read()
                
                text = self.extract_text_from_pdf(pdf_content)
                parsed_data = self.parse_groundwater_data(text, filename)
                self.all_data.extend(parsed_data)
                print(f"Extracted {len(parsed_data)} records from {filename}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def process_from_urls(self):
        """Download and process PDFs from URLs"""
        for url in self.pdf_urls:
            filename = url.split('/')[-1]
            pdf_content = self.download_pdf(url)
            
            if pdf_content:
                text = self.extract_text_from_pdf(pdf_content)
                parsed_data = self.parse_groundwater_data(text, filename)
                self.all_data.extend(parsed_data)
                print(f"Extracted {len(parsed_data)} records from {filename}")

    def save_to_csv(self, output_file: str = 'groundwater_data_compiled.csv'):
        """Save all extracted data to CSV"""
        if not self.all_data:
            print("No data to save!")
            return
        
        df = pd.DataFrame(self.all_data)
        
        # Sort by period and year
        period_order = ['January', 'Pre-Monsoon (April-May)', 'August', 'Post-Monsoon (November)']
        df['Period_Sort'] = df['Period'].apply(lambda x: period_order.index(x) if x in period_order else 999)
        df = df.sort_values(['Start_Year', 'Period_Sort', 'Water_Level_Depth_m'])
        df = df.drop('Period_Sort', axis=1)
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nData saved to {output_file}")
        print(f"Total records: {len(df)}")
        print(f"\nData summary:")
        print(df.groupby('Period').size())

    def display_preview(self, n_rows: int = 10):
        """Display preview of extracted data"""
        if not self.all_data:
            print("No data available!")
            return
        
        df = pd.DataFrame(self.all_data)
        print("\nPreview of extracted data:")
        print(df.head(n_rows))
        print(f"\nTotal records: {len(df)}")


def main():
    """Main function to run the converter"""
    converter = GroundwaterDataConverter()
    
    print("CGWB Groundwater Data Converter")
    print("=" * 50)
    print("\nChoose an option:")
    print("1. Download PDFs from CGWB website and process")
    print("2. Process locally downloaded PDF files")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\nDownloading and processing PDFs from CGWB website...")
        converter.process_from_urls()
    
    elif choice == '2':
        print("\nEnter the paths to your PDF files (one per line).")
        print("Press Enter twice when done:")
        file_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            file_paths.append(path)
        
        if file_paths:
            converter.process_local_files(file_paths)
        else:
            print("No files provided!")
            return
    
    else:
        print("Invalid choice!")
        return
    
    # Display preview
    converter.display_preview()
    
    # Save to CSV
    output_file = input("\nEnter output CSV filename (default: groundwater_data_compiled.csv): ").strip()
    if not output_file:
        output_file = 'groundwater_data_compiled.csv'
    
    converter.save_to_csv(output_file)
    print("\nDone!")


if __name__ == "__main__":
    # Required packages:
    # pip install requests PyPDF2 pandas
    
    main()