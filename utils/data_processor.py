"""
Data processing utilities for Engineer Performance Dashboard
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import chardet
import re
from typing import Optional, Tuple, Dict, Any
from config.settings import DATE_FORMATS, FILE_ENCODINGS, TaskStatus, PRIORITY_ORDER

class DataProcessor:
    """Handles all data loading and preprocessing operations"""
    
    def __init__(self):
        self.df = None
        self.encoding_used = None
        
    def detect_encoding(self, file_path) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding']
        except:
            return 'utf-8'
    
    def load_from_path(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from a direct file path (CSV or Excel)"""
        try:
            file_path_lower = file_path.lower()
            
            if file_path_lower.endswith(('.xlsx', '.xls')):
                # Excel file
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    self.encoding_used = 'excel'
                except:
                    df = pd.read_excel(file_path, engine='xlrd')
                    self.encoding_used = 'excel-xls'
                print(f"Successfully loaded Excel from path: {file_path}")
                return df
            else:
                # CSV file with encoding detection
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
                
                # Try detected encoding first
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='warn', low_memory=False)
                    self.encoding_used = encoding
                    
                    # Check if we got reasonable column count, if not try different separators
                    if len(df.columns) <= 1:
                        for sep in ['\t', ';', '|']:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='warn', low_memory=False)
                                if len(df.columns) > 1:
                                    print(f"Successfully loaded {sep}-separated CSV from path: {file_path}")
                                    return df
                            except:
                                continue
                        # If still 1 column, return it anyway (might be single column data)
                        print(f"Successfully loaded CSV from path: {file_path} with encoding {encoding}")
                        return df
                    else:
                        print(f"Successfully loaded CSV from path: {file_path} with encoding {encoding}")
                        return df
                except:
                    # Fallback encodings
                    for enc in ['utf-8-sig', 'utf-16', 'utf-16-le', 'latin-1']:
                        try:
                            df = pd.read_csv(file_path, encoding=enc, on_bad_lines='warn', low_memory=False)
                            # Try different separators if only 1 column
                            if len(df.columns) <= 1:
                                for sep in ['\t', ';', '|']:
                                    try:
                                        df = pd.read_csv(file_path, encoding=enc, sep=sep, on_bad_lines='warn', low_memory=False)
                                        if len(df.columns) > 1:
                                            print(f"Successfully loaded {sep}-separated CSV with fallback encoding: {enc}")
                                            return df
                                    except:
                                        continue
                                # If still 1 column, return it anyway (might be single column data)
                                print(f"Successfully loaded CSV with fallback encoding: {enc}")
                                return df
                            else:
                                print(f"Successfully loaded CSV with fallback encoding: {enc}")
                                return df
                        except:
                            continue
                    raise Exception(f"Failed to load CSV file with any encoding")
                    
        except Exception as e:
            raise Exception(f"Failed to load file from path {file_path}: {str(e)}")

    def load_excel_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load Excel file (.xlsx, .xls)"""
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            self.encoding_used = 'excel'
            print("Successfully loaded Excel file")
            return df
        except Exception as e1:
            try:
                df = pd.read_excel(file_path, engine='xlrd')
                self.encoding_used = 'excel-xls'
                print("Successfully loaded Excel file (xlrd)")
                return df
            except Exception as e2:
                raise Exception(f"Failed to load Excel file: {str(e2)}")

    def load_data_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data file (CSV or Excel) with automatic format detection"""
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith(('.xlsx', '.xls')):
            return self.load_excel_file(uploaded_file)
        else:
            return self.load_csv_with_fallback(uploaded_file)
    
    def load_csv_with_fallback(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load CSV with multiple encoding fallbacks including BOM handling"""
        
        # Read file bytes
        file_bytes = uploaded_file.getvalue()
        
        # Try UTF-8 with BOM first (common Excel export format)
        encodings_to_try = ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be'] + FILE_ENCODINGS
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(
                    pd.io.common.BytesIO(file_bytes),
                    encoding=encoding,
                    on_bad_lines='warn',
                    low_memory=False
                )
                self.encoding_used = encoding
                print(f"Successfully loaded with encoding: {encoding}")
                
                # Check if we got reasonable column count
                if len(df.columns) > 1:
                    return df
                # If only 1 column, might be tab-separated or semicolon separated
                if len(df.columns) == 1:
                    # Try with different separators
                    for sep in ['\t', ';', '|']:
                        try:
                            df = pd.read_csv(
                                pd.io.common.BytesIO(file_bytes),
                                encoding=encoding,
                                sep=sep,
                                on_bad_lines='warn',
                                low_memory=False
                            )
                            if len(df.columns) > 1:
                                print(f"Successfully loaded with separator '{sep}'")
                                return df
                        except:
                            continue
                    # If still 1 column, return it anyway (might be single column data)
                    return df
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {str(e)}")
                continue
        
        # If all encodings fail, try with latin-1 as last resort
        try:
            df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes),
                encoding='latin-1',
                on_bad_lines='skip'
            )
            self.encoding_used = 'latin-1'
            return df
        except Exception as e:
            raise Exception(f"Failed to load file with any encoding: {str(e)}")
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('[^0-9a-zA-Z]+', '_', regex=True)
        df.columns = df.columns.str.replace('_+', '_', regex=True)
        df.columns = df.columns.str.strip('_')
        return df
    
    def parse_datetime(self, date_str) -> Optional[pd.Timestamp]:
        """Parse datetime with multiple formats"""
        if pd.isna(date_str) or date_str is None:
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        for fmt in DATE_FORMATS:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Try pandas flexible parser as last resort
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        
        # Clean column names
        df = self.clean_column_names(df)
        
        # Parse datetime columns
        datetime_columns = ['Start', 'End', 'Due', 'Created']
        for col in datetime_columns:
            # Try to find matching column
            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
            if matching_cols:
                col_name = matching_cols[0]
                # Convert to datetime properly
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        
        # Calculate duration
        start_col = self._find_column(df, ['Start'])
        end_col = self._find_column(df, ['End'])
        
        if start_col and end_col:
            # Ensure both columns are datetime
            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
            df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
            
            # Calculate duration in hours using timedeltas
            duration_td = df[end_col] - df[start_col]
            df['Duration_Hours'] = duration_td.dt.total_seconds() / 3600
            df['Duration_Minutes'] = df['Duration_Hours'] * 60
            
            # Filter only negative durations and extreme outliers (>30 days), allow 0 duration
            original_count = len(df)
            df = df[
                (df['Duration_Hours'].notna()) &
                (df['Duration_Hours'] >= -0.1) &  # Allow small negative due to timezone issues
                (df['Duration_Hours'] <= 720)    # Max 30 days (was 7 days)
            ]
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} rows with invalid durations")
        else:
            # If no duration columns, create default duration column
            df['Duration_Hours'] = 0
            df['Duration_Minutes'] = 0
            print("Warning: No Start/End columns found, setting default duration to 0")
        
        # Clean text columns
        text_columns = ['Owner', 'Status', 'Type', 'Priority', 'Account']
        for col in text_columns:
            matching_col = self._find_column(df, [col, col.lower(), col.upper()])
            if matching_col:
                df[matching_col] = df[matching_col].fillna('Not Specified')
                df[matching_col] = df[matching_col].astype(str).str.strip()
                df[matching_col] = df[matching_col].replace(['nan', 'None', ''], 'Not Specified')
                
                # Clean owner names (remove email domains)
                if col == 'Owner':
                    df[matching_col] = df[matching_col].apply(
                        lambda x: x.split('@')[0] if '@' in x else x
                    )
        
        # Extract date components - only if start_col is datetime
        if start_col:
            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df[start_col]):
                df['Date'] = df[start_col].dt.date
                df['DayOfWeek'] = df[start_col].dt.day_name()
                df['Hour'] = df[start_col].dt.hour
                df['Month'] = df[start_col].dt.month
                df['Year'] = df[start_col].dt.year
        
        # Categorize status
        status_col = self._find_column(df, ['Status'])
        if status_col:
            df['Status_Category'] = df[status_col].apply(
                lambda x: self._categorize_status(x)
            )
        
        # Clean priority
        priority_col = self._find_column(df, ['Priority'])
        if priority_col:
            df['Priority_Level'] = df[priority_col].apply(
                lambda x: self._get_priority_level(x)
            )
            df['Priority_Order'] = df[priority_col].map(PRIORITY_ORDER).fillna(5)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def _find_column(self, df: pd.DataFrame, possible_names: list) -> Optional[str]:
        """Find column by possible names - case insensitive and partial match"""
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for name in possible_names:
            # Exact match first
            if name.lower() in df_cols_lower:
                return df_cols_lower[name.lower()]
            
            # Partial match
            for col_lower, col_original in df_cols_lower.items():
                if name.lower() in col_lower or col_lower in name.lower():
                    return col_original
        
        return None
    
    def _categorize_status(self, status: str) -> str:
        """Categorize task status"""
        for category in TaskStatus:
            for pattern in category.value:
                if pattern.lower() in str(status).lower():
                    return category.name
        return 'OTHER'
    
    def _get_priority_level(self, priority: str) -> int:
        """Get numeric priority level"""
        priority_str = str(priority).strip()
        return PRIORITY_ORDER.get(priority_str, 5)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality summary"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'date_range': None,
            'encoding': self.encoding_used
        }
        
        date_col = self._find_column(df, ['Date', 'Start', 'Created'])
        if date_col:
            summary['date_range'] = {
                'min': df[date_col].min(),
                'max': df[date_col].max()
            }
        
        return summary

# Global instance
data_processor = DataProcessor()