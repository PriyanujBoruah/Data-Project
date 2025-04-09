import os
import uuid
import pandas as pd
import numpy as np
from flask import (Flask, request, render_template, redirect, url_for,
                   send_from_directory, flash, session, make_response) # Added make_response here
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import chardet # For encoding detection
from collections import Counter
import time
# --- Added for type checking in profile ---
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype, is_object_dtype, is_bool_dtype
import io
from thefuzz import fuzz
import re
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation
import plotly.express as px
import plotly.io as pio


# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
CLEANED_FOLDER = 'cleaned_data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_FILE_SIZE = 32 * 1024 * 1024  # 16 MB limit

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLEANED_FOLDER'] = CLEANED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
# IMPORTANT: Use a strong, random secret key in production!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_replace_me')

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_encoding(file_path):
    """Detects file encoding using chardet (primarily for CSV)."""
    # --- (Existing detect_encoding logic - unchanged) ---
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(50000)
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
            return encoding
    except Exception:
        return 'utf-8'

# <<< MODIFIED: Handle CSV and XLSX loading >>>
def load_data(file_path, file_extension, encoding=None):
    """Loads data from CSV or XLSX file into a pandas DataFrame."""
    df = None
    used_encoding = None # Track encoding only if CSV

    try:
        if file_extension == 'csv':
            detected_encoding = None
            if not encoding:
                detected_encoding = detect_encoding(file_path)
                encoding_to_try = detected_encoding
                flash(f"Auto-detected encoding for CSV: {encoding_to_try}", "info")
            else:
                encoding_to_try = encoding
            used_encoding = encoding_to_try # Store the encoding used

            df = pd.read_csv(file_path, encoding=encoding_to_try)

        elif file_extension == 'xlsx':
            # Encoding is generally not needed for xlsx via openpyxl
            # Read the first sheet by default
            # Consider adding sheet_name=None to read all sheets into a dict,
            # but that requires more complex UI handling later.
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            if not sheet_names:
                flash("Excel file contains no sheets.", "warning")
                return None, None
            # Read only the first sheet for simplicity in this version
            first_sheet_name = sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=first_sheet_name, engine='openpyxl')
            if len(sheet_names) > 1:
                flash(f"Read data from the first sheet ('{first_sheet_name}'). File contains multiple sheets: {', '.join(sheet_names)}.", "info")
            else:
                flash(f"Read data from sheet '{first_sheet_name}'.", "info")
            used_encoding = 'xlsx' # Indicate format, not encoding

        else:
            flash(f"Unsupported file extension: {file_extension}", "danger")
            return None, None

        if df is not None:
            # Attempt basic type inference improvement
            df = df.infer_objects()
        return df, used_encoding

    except UnicodeDecodeError as e: # Specific to CSV
        flash(f"Error decoding CSV with '{encoding_to_try}'. Common alternatives: 'utf-8', 'latin-1', 'cp1252'. Please specify encoding.", "danger")
        return None, encoding_to_try
    except FileNotFoundError:
         flash(f"Error: Uploaded file not found at path '{file_path}'.", "danger")
         return None, used_encoding
    except Exception as e:
        flash(f"Error loading {file_extension.upper()} file: {str(e)}", "danger")
        return None, used_encoding

# --- (save_data function remains the same - always saves intermediate as CSV) ---
def save_data(df, original_filename):
    """Saves the DataFrame to a new CSV file in the cleaned_data folder."""
    if not os.path.exists(app.config['CLEANED_FOLDER']):
        os.makedirs(app.config['CLEANED_FOLDER'])
    secure_base = secure_filename(os.path.splitext(original_filename)[0])
    if not secure_base: secure_base = "data"
    # Always save intermediate steps as CSV for consistency within the app's workflow
    new_filename = f"{uuid.uuid4().hex}_{secure_base}.csv"
    output_path = os.path.join(app.config['CLEANED_FOLDER'], new_filename)
    try:
        # Ensure consistent saving format (UTF-8 CSV) for intermediate steps
        df.to_csv(output_path, index=False, encoding='utf-8')
        return output_path
    except Exception as e:
        flash(f"Error saving intermediate cleaned file: {str(e)}", "danger")
        return None

# Inside app.py

def auto_explore_data(df):
    """
    Analyzes the DataFrame and returns a list of findings/suggestions
    with more specific actions recommended.
    """
    findings = []
    if df is None or df.empty:
        findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'No data loaded or DataFrame is empty.', 'details': [], 'suggestion': 'Upload a valid CSV or Excel file.'})
        return findings

    num_rows, num_cols = df.shape
    if num_rows == 0 or num_cols == 0:
         findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'DataFrame has zero rows or columns.', 'details': [], 'suggestion': 'Check the uploaded file content.'})
         return findings

    # --- Helper function to format column lists for suggestions ---
    def format_cols_for_suggestion(col_list, max_cols=3):
        if not col_list:
            return ""
        cols_to_show = [f"'{c}'" for c in col_list[:max_cols]]
        suffix = "..." if len(col_list) > max_cols else ""
        return f" on columns like [{', '.join(cols_to_show)}{suffix}]"

    # Finding 1: Missing Data Analysis
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if not missing_cols.empty:
        total_missing = missing_cols.sum()
        pct_total_missing = (total_missing / (num_rows * num_cols)) * 100 if num_rows * num_cols > 0 else 0
        severity = 'High' if pct_total_missing > 10 else ('Medium' if pct_total_missing > 1 else 'Low')
        # --- MODIFIED SUGGESTION ---
        affected_cols_str = format_cols_for_suggestion(missing_cols.index.tolist())
        suggestion = f"Use '1. Clean Missing Data'{affected_cols_str} to handle NaNs (e.g., Action: 'Fill with Median/Mode' or 'Drop Rows')."
        findings.append({
            'issue_type': 'Missing Data',
            'severity': severity,
            'message': f"Found {total_missing} missing values ({pct_total_missing:.2f}% of total cells) across {len(missing_cols)} column(s).",
            'details': [f"'{col}': {count} missing ({ (count/num_rows)*100:.1f}%)" for col, count in missing_cols.items()],
            'suggestion': suggestion
        })

    # Finding 2: Data Type Overview & Potential Issues
    dtypes = df.dtypes
    object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    # --- Suggestion remains general as specific type issues are hard to auto-detect reliably ---
    findings.append({
        'issue_type': 'Data Types',
        'severity': 'Info',
        'message': f"Dataset has {num_cols} columns: {len(numeric_cols)} numeric, {len(object_cols)} text/object, {len(datetime_cols)} datetime.",
        'details': [f"'{col}': {str(dtype)}" for col, dtype in dtypes.items()],
        'suggestion': "Review data types using the 'Data Profile' feature or use '8. Convert Data Type' if needed."
    })

    # Finding 3: Potential Outliers (Using IQR for simplicity)
    outlier_details_list = [] # Store details for reuse
    outlier_columns_affected = [] # Store just the column names
    for col in numeric_cols:
        if df[col].isnull().all() or df[col].nunique() < 2: continue
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0: continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if pd.isna(lower_bound) or pd.isna(upper_bound): continue

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                outlier_count = outliers.count()
                pct_outliers = (outlier_count / num_rows) * 100 if num_rows > 0 else 0
                if pct_outliers > 0.1: # Only report if more than a tiny fraction
                     outlier_details_list.append(f"'{col}': {outlier_count} potential outliers ({pct_outliers:.2f}%)")
                     outlier_columns_affected.append(col)
        except Exception as e:
             print(f"Outlier check failed for column '{col}': {e}")
             outlier_details_list.append(f"'{col}': Could not reliably check for outliers.")

    if outlier_details_list:
         # --- MODIFIED SUGGESTION ---
         affected_cols_str = format_cols_for_suggestion(outlier_columns_affected)
         suggestion = f"Use '2. Clean Outlier Data'{affected_cols_str} using Method: 'IQR' or 'Modified Z-Score' (e.g., Action: 'Cap Value' or 'Remove Row')."
         findings.append({
            'issue_type': 'Potential Outliers',
            'severity': 'Medium', # Keep severity based on presence, not specific count here
            'message': f"Potential outliers detected in {len(outlier_columns_affected)} numeric column(s) using the IQR method.",
            'details': outlier_details_list,
            'suggestion': suggestion
        })

    # Finding 4: Duplicate Rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        pct_duplicates = (duplicate_count / num_rows) * 100 if num_rows > 0 else 0
        severity = 'Medium' if pct_duplicates > 5 else 'Low'
        # --- MODIFIED SUGGESTION ---
        suggestion = f"Use '5. Deduplicate Records' (Action: 'Keep First', Based on: All Columns) to handle {duplicate_count} duplicate rows."
        findings.append({
            'issue_type': 'Duplicate Records',
            'severity': severity,
            'message': f"Found {duplicate_count} duplicate rows ({pct_duplicates:.2f}% of total).",
            'details': [],
            'suggestion': suggestion
        })

    # Finding 5: Low Variance / Constant Columns
    constant_cols = []
    low_variance_cols = []
    if num_rows > 0:
        for col in df.columns:
            nunique = df[col].nunique(dropna=False) # Consider NaNs as a potential unique value
            if nunique <= 1: # <= 1 handles empty columns or all NaN columns too
                 constant_cols.append(col)
            # Check for low variance only if not constant and more than one row exists
            elif nunique > 1 and num_rows > 1 and (nunique / num_rows) < 0.01:
                 # Exclude purely ID-like columns (high unique count close to row count)
                 if not (nunique > num_rows * 0.95):
                     low_variance_cols.append(col)

    low_var_msgs = []
    low_var_cols_affected = []
    if constant_cols:
        low_var_msgs.append(f"Constant columns: {', '.join([f'{c}' for c in constant_cols])}")
        low_var_cols_affected.extend(constant_cols)
    if low_variance_cols:
         low_var_msgs.append(f"Low variance columns (<1% unique): {', '.join([f'{c}' for c in low_variance_cols])}")
         low_var_cols_affected.extend(low_variance_cols)

    if low_var_msgs:
        # --- MODIFIED SUGGESTION (Updated Op number) ---
        affected_cols_str = format_cols_for_suggestion(list(set(low_var_cols_affected))) # Use set to avoid duplicates
        # Make sure operation number matches HTML (It's 14 in the latest HTML)
        suggestion = f"Use '14. Remove Variable (Column)'{affected_cols_str} if these columns are not informative."
        findings.append({
            'issue_type': 'Low Variance',
            'severity': 'Low',
            'message': "Found columns with very few unique values.",
            'details': low_var_msgs,
            'suggestion': suggestion
        })

    # Finding 6: Potential Date Columns (in Object types)
    potential_date_cols = []
    potential_date_cols_names = []
    for col in object_cols:
        # Basic check: does it contain digits frequently?
        if df[col].isnull().all(): continue
        non_na_series = df[col].dropna().astype(str) # Convert to string first
        if non_na_series.empty: continue

        # More robust check: sample and try parsing
        sample_size = min(100, len(non_na_series))
        try:
            sample = non_na_series.sample(sample_size, random_state=1)
            # Heuristic: Check if a good portion contains digits/separators common in dates
            looks_like_date_heuristic = sample.str.contains(r'[\d/\-:]').mean() > 0.6

            if looks_like_date_heuristic:
                # Try parsing the sample
                parsed_dates = pd.to_datetime(sample, errors='coerce')
                parseable_fraction = parsed_dates.notna().mean()

                if parseable_fraction > 0.7: # If > 70% of sample parses
                    potential_date_cols.append(f"'{col}' (Sample parse rate: {parseable_fraction:.1%})")
                    potential_date_cols_names.append(col)
        except Exception as e:
             print(f"Date check failed for column '{col}': {e}")

    if potential_date_cols:
         # --- MODIFIED SUGGESTION ---
         affected_cols_str = format_cols_for_suggestion(potential_date_cols_names)
         # Suggest 'Convert Data Type' as it's more direct now
         suggestion = f"Use '8. Convert Data Type'{affected_cols_str} selecting Target Type: 'Date/Time'."
         findings.append({
            'issue_type': 'Potential Dates',
            'severity': 'Info',
            'message': f"Found {len(potential_date_cols)} text column(s) that might contain dates.",
            'details': potential_date_cols,
            'suggestion': suggestion
        })

    # Finding 7: Text Issues (Case & Whitespace) - Check sample
    text_issue_cols_details = []
    text_issue_cols_names = []
    sample_size = min(500, num_rows)
    if sample_size > 0:
        df_sample = df.sample(sample_size, random_state=1)
        for col in object_cols:
            if df_sample[col].isnull().all(): continue
            try:
                col_str = df_sample[col].dropna().astype(str)
                if col_str.empty: continue

                # Check for leading/trailing whitespace
                has_whitespace = (col_str != col_str.str.strip()).any()

                # Check for mixed case (more than 1 unique value after lowercasing if original unique > 1)
                unique_vals = col_str.unique()
                has_mixed_case = False
                if len(unique_vals) > 1:
                     lower_unique_count = col_str.str.lower().nunique()
                     if lower_unique_count < len(unique_vals):
                         has_mixed_case = True

                if has_whitespace or has_mixed_case:
                    issues = []
                    if has_whitespace: issues.append("whitespace")
                    if has_mixed_case: issues.append("casing")
                    text_issue_cols_details.append(f"'{col}': Contains {'/'.join(issues)} (based on sample)")
                    text_issue_cols_names.append(col)
            except Exception as e:
                print(f"Text check failed for column '{col}': {e}")

    if text_issue_cols_details:
         # --- MODIFIED SUGGESTION ---
         affected_cols_str = format_cols_for_suggestion(text_issue_cols_names)
         suggestion = f"Use '7. Case & Whitespace'{affected_cols_str} (e.g., Action: 'Strip Leading/Trailing Whitespace' and 'Convert to Lowercase')."
         findings.append({
            'issue_type': 'Text Formatting',
            'severity': 'Low',
            'message': f"Potential text formatting issues found in {len(text_issue_cols_names)} column(s).",
            'details': text_issue_cols_details,
            'suggestion': suggestion
        })

    # Finding 8: High Cardinality Text Columns
    high_cardinality_cols = []
    if num_rows > 0:
        for col in object_cols:
            try:
                unique_count = df[col].nunique()
                # Define high cardinality: e.g., > 50 unique AND > 10% of rows are unique, but NOT almost entirely unique (like IDs)
                is_high_card = unique_count > 50 and (unique_count / num_rows) > 0.10
                is_likely_id = unique_count > num_rows * 0.9 # Heuristic for ID columns

                if is_high_card and not is_likely_id:
                    high_cardinality_cols.append(f"'{col}': {unique_count} unique values")
            except Exception as e:
                print(f"Cardinality check failed for column '{col}': {e}")

    if high_cardinality_cols:
         # --- Suggestion remains general - maybe suggest fuzzy matching? ---
         suggestion = "Review if these need cleaning (e.g., '10. Fuzzy Match Text Values' if variations exist) or are identifiers/free text."
         findings.append({
            'issue_type': 'High Cardinality Text',
            'severity': 'Info',
            'message': f"Found {len(high_cardinality_cols)} text column(s) with many unique values (excluding likely IDs).",
            'details': high_cardinality_cols,
            'suggestion': suggestion
        })

    return findings
# --- END auto_explore_data ---


# --- NEW Helper Function for Data Profiling ---
def generate_profile(df):
    """Generates detailed statistics for each column."""
    profile = {}
    if df is None or df.empty:
        return {"error": "DataFrame is empty or not loaded."}

    total_rows = len(df)

    for col in df.columns:
        column_data = df[col]
        stats = {}

        # Common Stats
        stats['dtype'] = str(column_data.dtype)
        stats['count'] = int(column_data.count()) # Non-missing count
        stats['missing_count'] = int(column_data.isnull().sum())
        stats['missing_percent'] = round((stats['missing_count'] / total_rows) * 100, 2) if total_rows > 0 else 0
        stats['unique_count'] = int(column_data.nunique())
        stats['unique_percent'] = round((stats['unique_count'] / total_rows) * 100, 2) if total_rows > 0 else 0

        # Type-Specific Stats
        if is_numeric_dtype(column_data):
            stats['type'] = 'Numeric'
            desc = column_data.describe()
            stats['mean'] = round(desc.get('mean', np.nan), 4)
            stats['std'] = round(desc.get('std', np.nan), 4)
            stats['min'] = round(desc.get('min', np.nan), 4)
            stats['25%'] = round(desc.get('25%', np.nan), 4)
            stats['50%'] = round(desc.get('50%', np.nan), 4) # Median
            stats['75%'] = round(desc.get('75%', np.nan), 4)
            stats['max'] = round(desc.get('max', np.nan), 4)
            try: # Skew and Kurt can fail on edge cases
                stats['skewness'] = round(column_data.skew(), 4)
                stats['kurtosis'] = round(column_data.kurt(), 4)
            except Exception:
                stats['skewness'] = np.nan
                stats['kurtosis'] = np.nan

        elif is_datetime64_any_dtype(column_data):
            stats['type'] = 'Datetime'
            # Ensure min/max are calculated only on non-NaT values
            non_na_dates = column_data.dropna()
            stats['min_date'] = str(non_na_dates.min()) if not non_na_dates.empty else 'N/A'
            stats['max_date'] = str(non_na_dates.max()) if not non_na_dates.empty else 'N/A'

        elif is_bool_dtype(column_data):
            stats['type'] = 'Boolean'
            # Count values including potential NAs if using nullable boolean
            value_counts = column_data.value_counts(dropna=False)
            stats['true_count'] = int(value_counts.get(True, 0))
            stats['false_count'] = int(value_counts.get(False, 0))

        # Check for pandas StringDtype explicitly, or fallback to object
        elif is_string_dtype(column_data) or is_object_dtype(column_data):
            stats['type'] = 'Text/Object'
            # Get top 5 most frequent values (handle potential non-hashable types)
            try:
                value_counts = column_data.value_counts().head(5)
                stats['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
            except TypeError:
                stats['top_values'] = {"Error": "Contains non-hashable types"}

            # Basic stats on string length (ignoring NaNs)
            try:
                str_series = column_data.dropna().astype(str)
                if not str_series.empty:
                    str_lengths = str_series.str.len()
                    stats['min_length'] = int(str_lengths.min())
                    stats['mean_length'] = round(str_lengths.mean(), 2)
                    stats['max_length'] = int(str_lengths.max())
                else:
                    stats['min_length'] = stats['mean_length'] = stats['max_length'] = 0
            except Exception: # Catch potential errors during string conversion/length calc
                stats['min_length'] = stats['mean_length'] = stats['max_length'] = 'Error'
        else:
            stats['type'] = 'Other'

        profile[col] = stats
    return profile
# --- END generate_profile ---


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the initial upload page."""
    # Clear session data when returning to index
    session.pop('current_file_path', None)
    session.pop('original_filename', None)
    session.pop('encoding', None) # Keep encoding to store format info (csv encoding or 'xlsx')
    session.pop('exploration_results', None)
    session.pop('profile_results', None)
    return render_template('index.html')

# <<< MODIFIED: Pass file extension to load_data >>>
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, saves it, loads data, and redirects to cleaning page."""
    if 'file' not in request.files:
        flash('No file part selected.', 'warning')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('index'))

    # <<< Check allowed extensions using the helper >>>
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # <<< Get file extension >>>
        file_extension = filename.rsplit('.', 1)[1].lower()

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        upload_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)

        try:
            file.save(upload_path)
        except Exception as e:
            flash(f"Error saving uploaded file: {str(e)}", "danger")
            return redirect(url_for('index'))

        # Get user-specified encoding (relevant only for CSV)
        user_encoding = request.form.get('encoding') or None

        # <<< Pass file extension to load_data >>>
        df, source_info = load_data(upload_path, file_extension, encoding=user_encoding)

        if df is None:
            if os.path.exists(upload_path):
                try: os.remove(upload_path)
                except OSError as e: print(f"Error removing failed upload file {upload_path}: {e}")
            return redirect(url_for('index'))

        # Save the initial state as intermediate CSV
        cleaned_path = save_data(df, filename)
        if cleaned_path is None:
            if os.path.exists(upload_path):
                try: os.remove(upload_path)
                except OSError as e: print(f"Error removing upload file {upload_path} after save fail: {e}")
            return redirect(url_for('index'))

        # Store info in session
        session['current_file_path'] = cleaned_path # Path to intermediate CSV
        session['original_filename'] = filename # Keep original name for downloads
        session['source_format'] = file_extension # Store original format
        session['source_encoding_info'] = source_info # Store CSV encoding or 'xlsx' tag

        session.pop('exploration_results', None)
        session.pop('profile_results', None)

        # Cleanup the initial upload file
        if os.path.exists(upload_path):
            try: os.remove(upload_path)
            except OSError as e: print(f"Error removing temp upload file {upload_path}: {e}")

        flash(f"File '{filename}' uploaded successfully.", "success")
        return redirect(url_for('clean_data'))
    else:
        # <<< Updated flash message for allowed types >>>
        flash(f"Invalid file type. Please upload a CSV or XLSX file.", 'danger')
        return redirect(url_for('index'))

@app.route('/explore', methods=['POST'])
def explore_data():
    """Performs automated data exploration."""
    file_path = session.get('current_file_path')
    if not file_path or not os.path.exists(file_path):
        flash("No data loaded to explore. Please upload a file first.", "warning")
        return redirect(url_for('index'))

    try:
        start_time = time.time()
        current_encoding = session.get('encoding', 'utf-8')
        df = pd.read_csv(file_path, encoding=current_encoding)
        df = df.infer_objects()

        flash("Running Auto Explore analysis...", "info")
        try:
            exploration_results = auto_explore_data(df)
            session['exploration_results'] = exploration_results
            session.pop('profile_results', None) # <<< Clear profile results when running explore
            duration = time.time() - start_time
            flash(f"Auto Explore completed in {duration:.2f} seconds. Found {len(exploration_results)} potential areas for review.", "success")
        except Exception as explore_err:
            flash(f"An error occurred during the Auto Explore analysis process: {str(explore_err)}", "danger")
            session.pop('exploration_results', None)

    except Exception as e:
        flash(f"An error occurred loading data for Auto Explore: {str(e)}", "danger")
        session.pop('exploration_results', None)

    return redirect(url_for('clean_data'))


# --- NEW Route for Generating Profile ---
@app.route('/profile', methods=['POST'])
def profile_data():
    """Generates and stores data profile."""
    file_path = session.get('current_file_path')
    if not file_path or not os.path.exists(file_path):
        flash("No data loaded to profile. Please upload a file first.", "warning")
        return redirect(url_for('index'))

    try:
        start_time = time.time()
        current_encoding = session.get('encoding', 'utf-8')
        df = pd.read_csv(file_path, encoding=current_encoding)
        df = df.infer_objects()

        flash("Generating Data Profile...", "info")
        try:
            profile_results = generate_profile(df)
            session['profile_results'] = profile_results # Store profile results
            session.pop('exploration_results', None) # <<< Clear exploration results when running profile
            duration = time.time() - start_time
            flash(f"Data Profile generated in {duration:.2f} seconds.", "success")
        except Exception as profile_err:
             flash(f"An error occurred during the profiling process: {str(profile_err)}", "danger")
             session.pop('profile_results', None)

    except Exception as e:
        flash(f"An error occurred loading data for Profiling: {str(e)}", "danger")
        session.pop('profile_results', None)

    return redirect(url_for('clean_data'))
# --- END profile_data route ---


# <<< NEW Route for Fuzzy Matching Analysis Step >>>
@app.route('/fuzzy_analyze', methods=['POST'])
def fuzzy_analyze():
    """Analyzes a column to find potentially similar groups using fuzzy matching."""
    file_path = session.get('current_file_path')
    if not file_path or not os.path.exists(file_path):
        flash("No data loaded to analyze. Please upload a file first.", "warning")
        return redirect(url_for('index'))

    col = request.form.get('fuzzy_column')
    try:
        threshold = int(request.form.get('fuzzy_threshold', 85)) # Default 85% similarity
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100.")
    except (ValueError, TypeError):
        flash("Invalid similarity threshold. Please enter an integer between 0 and 100.", "danger")
        return redirect(url_for('clean_data'))

    if not col:
         flash("Please select a column to analyze for fuzzy matching.", "warning")
         return redirect(url_for('clean_data'))

    try:
        start_time = time.time()
        current_encoding = session.get('encoding', 'utf-8') # Should always be utf-8 for intermediate
        df = pd.read_csv(file_path, encoding=current_encoding)
        df = df.infer_objects()

        if col not in df.columns:
            flash(f"Column '{col}' not found in the current dataset.", "warning")
            return redirect(url_for('clean_data'))

        # Ensure column is string type and get unique non-null values
        unique_vals = df[col].dropna().astype(str).unique()

        if len(unique_vals) < 2:
            flash(f"Not enough unique text values in column '{col}' to perform fuzzy matching.", "info")
            session.pop('fuzzy_results', None) # Clear any previous results
            return redirect(url_for('clean_data'))

        flash(f"Analyzing column '{col}' for values with similarity >= {threshold}%...", "info")

        groups = []
        processed_indices = set() # Use indices for faster lookup

        # Sort for potentially minor efficiency gain & consistent grouping order
        unique_vals_list = sorted(list(unique_vals))
        n_unique = len(unique_vals_list)

        # O(n^2) comparison - can be slow for very high cardinality columns
        for i in range(n_unique):
            if i in processed_indices:
                continue

            current_group = [unique_vals_list[i]]
            processed_indices.add(i)

            for j in range(i + 1, n_unique):
                if j in processed_indices:
                    continue

                # Calculate similarity ratio
                ratio = fuzz.ratio(unique_vals_list[i], unique_vals_list[j])

                if ratio >= threshold:
                    current_group.append(unique_vals_list[j])
                    processed_indices.add(j)

            # Only store groups with more than one member (i.e., matches found)
            if len(current_group) > 1:
                # Sort group members alphabetically for consistent display
                groups.append(sorted(current_group))

        duration = time.time() - start_time
        if groups:
            session['fuzzy_results'] = {'column': col, 'threshold': threshold, 'groups': groups}
            flash(f"Fuzzy analysis complete in {duration:.2f}s. Found {len(groups)} groups of similar values. Review below and apply changes.", "success")
        else:
            session.pop('fuzzy_results', None) # Clear if no groups found
            flash(f"Fuzzy analysis complete in {duration:.2f}s. No groups found meeting the {threshold}% similarity threshold.", "info")

    except FileNotFoundError:
        flash("Error: Intermediate data file missing during fuzzy analysis.", "danger")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An error occurred during fuzzy analysis: {str(e)}", "danger")
        session.pop('fuzzy_results', None) # Clear results on error

    return redirect(url_for('clean_data'))
# <<< END fuzzy_analyze route >>>



@app.route('/clean')
def clean_data():
    """Displays the data preview and cleaning options."""
    # --- (Reads intermediate CSV, logic largely unchanged) ---
    file_path = session.get('current_file_path')
    if not file_path or not os.path.exists(file_path):
        # --- (Existing error handling) ---
        flash("No data file found in session or file missing. Please upload again.", "warning")
        session.pop('exploration_results', None)
        session.pop('profile_results', None)
        session.pop('fuzzy_results', None) # <<< Clear fuzzy results
        return redirect(url_for('index'))
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df.infer_objects()
        columns = df.columns.tolist()

        # --- (Preview generation logic unchanged) ---
        preview_rows = 100
        actual_rows_shown = min(preview_rows, len(df))
        df_preview = df.head(preview_rows).to_html(classes='table table-striped table-sm', border=0, index=False)
        preview_text = f"Data Preview ({actual_rows_shown} of {len(df)} Rows):"

        # --- Get results from session ---
        exploration_results = session.get('exploration_results')
        profile_results = session.get('profile_results')
        fuzzy_results = session.get('fuzzy_results') # <<< Get fuzzy results

        return render_template('clean.html',
                               columns=columns,
                               df_preview=df_preview,
                               preview_text=preview_text,
                               exploration_results=exploration_results,
                               profile_results=profile_results,
                               fuzzy_results=fuzzy_results) # <<< Pass fuzzy results

    # --- (Existing error handling, ensure fuzzy_results cleared) ---
    except FileNotFoundError:
        flash(f"Error: The intermediate data file seems to be missing. Please upload again.", "danger")
        session.pop('current_file_path', None); session.pop('original_filename', None); session.pop('source_format', None); session.pop('source_encoding_info', None); session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None) # <<< Clear fuzzy
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error reading or displaying data file: {str(e)}. You might need to re-upload.", "danger")
        session.pop('current_file_path', None); session.pop('original_filename', None); session.pop('source_format', None); session.pop('source_encoding_info', None); session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None) # <<< Clear fuzzy
        return redirect(url_for('index'))


@app.route('/apply/<operation>', methods=['POST'])
def apply_cleaning(operation):
    """Applies the selected cleaning operation."""
    file_path = session.get('current_file_path')
    original_filename = session.get('original_filename')

    # Clear previous exploration, profile, AND fuzzy analysis results
    session.pop('exploration_results', None)
    session.pop('profile_results', None)
    # Don't clear fuzzy results if the operation IS fuzzy_apply, clear it AFTER applying
    if operation != 'fuzzy_apply':
        session.pop('fuzzy_results', None)

    if not file_path or not original_filename or not os.path.exists(file_path):
        flash("Session expired or file missing. Please upload again.", "warning")
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df.infer_objects()
        original_shape = df.shape
        cols_affected = []

        # --- Apply Operations ---

        # --- (Existing operations 1-7: missing, outlier, smooth, normalize, deduplicate, date, case) ---
        # --- (Logic remains the same for these) ---
        if operation == 'missing':
            cols = request.form.getlist('columns') # Use getlist for multiple select
            method = request.form.get('missing_method')
            fill_val = request.form.get('fill_value')
            target_cols = cols if cols else df.columns.tolist() # Explicitly list all columns if none selected

            if method == 'drop_row':
                subset_param = target_cols if cols else None # Drop row if NaN in ANY column only if specific cols aren't selected
                df.dropna(subset=subset_param, inplace=True)
                # This affects rows, not specific columns in place
            elif method == 'drop_col':
                # Drop columns if they contain *any* NaN values
                cols_to_drop = [col for col in target_cols if df[col].isnull().any()]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    cols_affected = cols_to_drop # These columns were removed
                else:
                    flash(f"No columns found with missing values among selected/all columns.", "info")

            else: # Fill methods
                applied_fill = False
                for col in target_cols:
                    if df[col].isnull().any(): # Only process columns with actual NaNs
                        original_dtype = df[col].dtype
                        try:
                            if method == 'fill_mean':
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    mean_val = df[col].mean()
                                    df[col].fillna(mean_val, inplace=True)
                                    cols_affected.append(col)
                                    applied_fill = True
                                else:
                                    flash(f"Column '{col}' is not numeric. Cannot fill with mean.", "warning")
                            elif method == 'fill_median':
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    median_val = df[col].median()
                                    df[col].fillna(median_val, inplace=True)
                                    cols_affected.append(col)
                                    applied_fill = True
                                else:
                                    flash(f"Column '{col}' is not numeric. Cannot fill with median.", "warning")
                            elif method == 'fill_mode':
                                # Mode can return multiple values, take the first
                                mode_val = df[col].mode()
                                if not mode_val.empty:
                                    df[col].fillna(mode_val[0], inplace=True)
                                    cols_affected.append(col)
                                    applied_fill = True
                                else:
                                    flash(f"Could not determine mode for column '{col}'.", "warning")
                            elif method == 'fill_value':
                                if fill_val is None: # Check if value was provided
                                    flash("Please provide a fill value.", "warning")
                                    continue
                                # Try to convert fill_val to the column's dtype if possible
                                try:
                                    # Create a temporary series to get the correct type
                                    typed_fill_val = pd.Series([fill_val]).astype(original_dtype).iloc[0]
                                    df[col].fillna(typed_fill_val, inplace=True)
                                except (ValueError, TypeError, OverflowError):
                                    # Fallback to raw string value if conversion fails
                                    df[col].fillna(fill_val, inplace=True)
                                cols_affected.append(col)
                                applied_fill = True
                        except Exception as e:
                            flash(f"Error filling column '{col}': {e}", "danger")
                if not applied_fill and method not in ['drop_row', 'drop_col']:
                    flash(f"No missing values found in selected/all columns to fill.", "info")

        elif operation == 'outlier':
            col = request.form.get('column')
            method = request.form.get('outlier_method')
            threshold_str = request.form.get('threshold', '') # Get threshold as string
            action = request.form.get('action') # remove or cap

            if col not in df.columns:
                flash(f"Column '{col}' not found.", "warning")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                flash(f"Column '{col}' is not numeric. Cannot perform outlier detection.", "warning")
            else:
                numeric_col_data = df[col].dropna() # Work on non-missing values
                if numeric_col_data.empty or numeric_col_data.nunique() < 2:
                    flash(f"Not enough valid numeric data in '{col}' for outlier detection.", "info")
                else:
                    lower_bound, upper_bound = None, None
                    outlier_indices = pd.Index([]) # Store indices of outliers

                    # --- Parameter Parsing and Validation ---
                    try:
                        if method in ['iqr', 'zscore', 'modified_zscore']:
                             threshold = float(threshold_str)
                             if threshold <= 0:
                                 raise ValueError("Threshold must be positive.")
                        elif method == 'percentile':
                             threshold = float(threshold_str) # Represents percentile 'p'
                             if not (0 < threshold < 50):
                                 raise ValueError("Percentile must be between 0 and 50 (exclusive).")
                        elif method == 'isolation_forest':
                             # Contamination parameter
                             if threshold_str.lower() == 'auto':
                                 threshold = 'auto'
                             else:
                                 threshold = float(threshold_str)
                                 if not (0.0 < threshold <= 0.5):
                                     raise ValueError("Contamination must be 'auto' or between 0.0 and 0.5.")
                        else:
                             flash(f"Invalid outlier method: {method}.", "danger")
                             return redirect(url_for('clean_data'))
                    except (ValueError, TypeError) as e:
                        flash(f"Invalid threshold/parameter value '{threshold_str}' for method '{method}': {e}", "danger")
                        return redirect(url_for('clean_data'))
                    # --- End Parameter Parsing ---


                    # --- Outlier Detection Logic ---
                    try: # Wrap detection in try-except
                        if method == 'iqr':
                            Q1 = numeric_col_data.quantile(0.25)
                            Q3 = numeric_col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            if IQR > 0:
                                lower_bound = Q1 - threshold * IQR
                                upper_bound = Q3 + threshold * IQR
                                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                                outlier_indices = df.index[outlier_mask & df[col].notna()] # Get indices of non-NaN outliers
                            else:
                                flash(f"IQR is zero for column '{col}'. Cannot detect outliers with IQR.", "info")

                        elif method == 'zscore':
                            mean_val = numeric_col_data.mean()
                            std_val = numeric_col_data.std()
                            if std_val > 0:
                                z_scores = np.abs((df[col] - mean_val) / std_val)
                                lower_bound = mean_val - threshold * std_val
                                upper_bound = mean_val + threshold * std_val
                                outlier_mask = z_scores > threshold
                                outlier_indices = df.index[outlier_mask & df[col].notna()]
                            else:
                                flash(f"Standard deviation is zero for column '{col}'. Cannot detect outliers with Z-score.", "info")

                        # --- NEW: Percentile Method ---
                        elif method == 'percentile':
                            lower_p = threshold / 100.0
                            upper_p = 1.0 - (threshold / 100.0)
                            lower_bound = numeric_col_data.quantile(lower_p)
                            upper_bound = numeric_col_data.quantile(upper_p)
                            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                            outlier_indices = df.index[outlier_mask & df[col].notna()]
                            flash(f"Using bounds: < {lower_bound:.4f} ({lower_p*100:.1f}th) or > {upper_bound:.4f} ({upper_p*100:.1f}th percentile).", "info")

                        # --- NEW: Modified Z-score Method ---
                        elif method == 'modified_zscore':
                            median_val = numeric_col_data.median()
                            # Use scipy's median_abs_deviation for robustness
                            mad = median_abs_deviation(numeric_col_data, nan_policy='omit') # 'omit' handles potential NaNs passed, though we dropped them

                            if mad > 0:
                                # Calculate modified z-scores only for non-NaN values
                                mod_z = 0.6745 * (df[col] - median_val) / mad
                                # Define bounds based on the threshold
                                lower_bound = median_val - threshold * mad / 0.6745
                                upper_bound = median_val + threshold * mad / 0.6745
                                outlier_mask = np.abs(mod_z) > threshold
                                outlier_indices = df.index[outlier_mask & df[col].notna()]
                            else:
                                flash(f"Median Absolute Deviation (MAD) is zero for column '{col}'. Cannot detect outliers with Modified Z-score.", "info")

                        # --- NEW: Isolation Forest Method ---
                        elif method == 'isolation_forest':
                            # Reshape data for Isolation Forest (expects 2D array)
                            data_for_iforest = numeric_col_data.values.reshape(-1, 1)

                            # Handle contamination='auto' or float
                            contam_value = 'auto' if threshold == 'auto' else float(threshold)

                            iforest = IsolationForest(contamination=contam_value, random_state=42, n_estimators=100) # Added random_state and n_estimators
                            iforest.fit(data_for_iforest)

                            # Predict on the original non-NaN data to get indices
                            # Need to reshape the data being predicted as well
                            predictions = iforest.predict(df.loc[numeric_col_data.index, [col]].values) # Predict on non-NaN values only

                            # Get indices where prediction is -1 (outlier)
                            outlier_indices = numeric_col_data.index[predictions == -1]

                            # Capping is less standard for Isolation Forest, bounds aren't directly derived
                            lower_bound, upper_bound = None, None
                            if action == 'cap':
                                flash("Warning: Capping action is generally not recommended or well-defined for Isolation Forest. Consider using 'Remove Row'.", "warning")
                                # Prevent capping attempt by unsetting bounds
                                # Or, could attempt capping based on min/max of inliers, but that's arbitrary

                    except Exception as detect_err:
                         flash(f"Error during outlier detection with method '{method}': {detect_err}", "danger")
                         return redirect(url_for('clean_data'))
                    # --- End Outlier Detection ---


                    # --- Apply Action ---
                    num_outliers = len(outlier_indices)

                    if num_outliers == 0:
                        flash(f"No outliers detected in column '{col}' using method '{method}' with parameter '{threshold_str}'.", "info")
                    elif action == 'remove':
                        df.drop(index=outlier_indices, inplace=True)
                        cols_affected = [] # Row operation
                        flash(f"Removed {num_outliers} outliers from column '{col}' using {method}.", "success")
                    elif action == 'cap':
                        if lower_bound is not None and upper_bound is not None:
                            # Perform capping only on the identified outliers
                            # Ensure we only cap actual numbers, not NaNs that might be caught by broad index mask
                            cap_mask = df.index.isin(outlier_indices) & df[col].notna()
                            df.loc[cap_mask, col] = df.loc[cap_mask, col].clip(lower=lower_bound, upper=upper_bound)
                            cols_affected = [col] # Value operation
                            flash(f"Capped {num_outliers} outliers in column '{col}' using {method} (Bounds: {lower_bound:.4f} / {upper_bound:.4f}).", "success")
                        else:
                            flash(f"Cannot perform 'Cap' action for method '{method}' as bounds were not determined (or method is unsuitable for capping).", "warning")

                    else:
                        flash("Invalid action selected.", "danger")

        elif operation == 'smooth':
            col = request.form.get('column')
            method = request.form.get('smooth_method')
            try:
                window = int(request.form.get('window', 3))
            except (ValueError, TypeError):
                flash("Invalid window size. Please enter an integer.", "danger")
                return redirect(url_for('clean_data'))

            if col not in df.columns:
                flash(f"Column '{col}' not found.", "warning")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                flash(f"Column '{col}' is not numeric. Cannot smooth data.", "warning")
            elif window < 2:
                flash("Window size must be at least 2.", "warning")
            else:
                if method == 'moving_average':
                    # min_periods=1 ensures edges are calculated, center=True is often preferred
                    df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean()
                    cols_affected = [col]
                # Add other smoothing methods here (e.g., exponential)
                else:
                    flash("Invalid smoothing method.", "danger")

        elif operation == 'normalize':
            cols = request.form.getlist('columns')
            method = request.form.get('normalize_method')
            numeric_cols_in_df = df.select_dtypes(include=np.number).columns.tolist()

            # If specific columns selected, use them; otherwise use all numeric
            target_cols = [c for c in cols if c in numeric_cols_in_df] if cols else numeric_cols_in_df

            if not target_cols:
                flash("No numeric columns selected or found for normalization.", "warning")
            else:
                applied_norm = False
                for col in target_cols:
                    # Reshape data for scaler: scaler expects 2D array
                    data_to_scale = df[[col]].dropna() # Drop NaN before scaling
                    if data_to_scale.empty or data_to_scale.nunique()[0] < 2 : # Skip empty or constant columns
                        flash(f"Skipping normalization for column '{col}': not enough unique numeric data.", "info")
                        continue

                    try:
                        if method == 'min_max':
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(data_to_scale)
                        elif method == 'z_score':
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(data_to_scale)
                        else:
                            flash("Invalid normalization method.", "danger")
                            continue # Skip this column

                        # Put scaled data back into the DataFrame, preserving original index
                        df.loc[data_to_scale.index, col] = scaled_data.flatten()
                        cols_affected.append(col)
                        applied_norm = True
                    except Exception as e:
                        flash(f"Error normalizing column '{col}': {e}", "danger")

                if not applied_norm and target_cols:
                    flash("Normalization not applied to any selected columns (check data type or variance).", "info")

        elif operation == 'deduplicate':
            cols = request.form.getlist('columns')
            keep = request.form.get('keep')
            if keep == 'none': keep_param = False # drop_duplicates uses False for this
            elif keep in ['first', 'last']: keep_param = keep
            else:
                flash("Invalid 'keep' parameter for deduplication.", "danger")
                return redirect(url_for('clean_data'))

            subset_param = cols if cols else None
            df.drop_duplicates(subset=subset_param, keep=keep_param, inplace=True)
            # Affects rows

        elif operation == 'date':
            cols = request.form.getlist('columns')
            output_format = request.form.get('date_format') or None # None uses default pandas format
            errors_coerce = request.form.get('errors_coerce') == 'true'

            if not cols:
                flash("Please select at least one column for date standardization.", "warning")
            else:
                applied_date_conv = False
                for col in cols:
                    if col not in df.columns:
                        flash(f"Column '{col}' not found.", "warning")
                        continue

                    try:
                        # Attempt conversion - infer_datetime_format can speed up common formats
                        df[col] = pd.to_datetime(df[col], errors='coerce' if errors_coerce else 'raise', infer_datetime_format=True)
                        cols_affected.append(col)
                        applied_date_conv = True

                        # Apply formatting if specified and conversion resulted in datetime type
                        if output_format and pd.api.types.is_datetime64_any_dtype(df[col]):
                            # Format valid dates, keep NaT as NaT
                            # Handle potential errors during formatting itself
                            try:
                                # NaNs (NaT) will become NaN string if formatted, handle explicitly
                                is_nat = df[col].isna()
                                df.loc[~is_nat, col] = df.loc[~is_nat, col].dt.strftime(output_format)
                                # Convert NaT back to None or pd.NA for consistency after strftime makes them objects
                                df.loc[is_nat, col] = pd.NA
                            except ValueError as format_error:
                                flash(f"Invalid date format string '{output_format}': {format_error}", "danger")
                                # Revert this column's type if formatting failed? Or leave as datetime? Let's leave as datetime.
                                continue # Skip formatting for this column
                    except ValueError as e:
                        # This should only happen if errors='raise' and parsing fails
                        flash(f"Error parsing date in column '{col}': {e}. Try enabling 'Convert unparseable dates to NaT'.", "danger")
                    except Exception as e:
                        flash(f"An unexpected error occurred during date conversion for column '{col}': {e}", "danger")

                if not applied_date_conv and cols:
                    flash("Date conversion not applied to any selected columns (check errors or data).", "info")

        elif operation == 'case':
            cols = request.form.getlist('columns')
            case_type = request.form.get('case_type')

            if not cols:
                flash("Please select at least one column for case standardization.", "warning")
            else:
                applied_case = False
                for col in cols:
                    if col not in df.columns:
                        flash(f"Column '{col}' not found.", "warning")
                        continue
                    # Ensure the column is treated as string for these operations
                    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                        # Convert potential mixed types to string, handling NaN correctly
                        # .astype(str) converts NaN to 'nan', use .astype("string") (Pandas string type)
                        df[col] = df[col].astype("string") # Uses pd.NA for missing

                        if case_type == 'lower':
                            df[col] = df[col].str.lower()
                        elif case_type == 'upper':
                            df[col] = df[col].str.upper()
                        elif case_type == 'title':
                            df[col] = df[col].str.title()
                        elif case_type == 'strip':
                            df[col] = df[col].str.strip()
                        else:
                            flash("Invalid case type.", "danger")
                            continue # Skip invalid case types
                        cols_affected.append(col)
                        applied_case = True
                    else:
                        flash(f"Column '{col}' is not a text type. Cannot apply case standardization.", "warning")
                if not applied_case and cols:
                    flash("Case standardization not applied (check column types).", "info")

        # --- NEW Operation 8: Convert Data Type ---
        elif operation == 'convert_type':
            col = request.form.get('column')
            target_type = request.form.get('target_type')

            if not col or col not in df.columns:
                flash(f"Column '{col}' not found or not selected.", "warning")
            elif not target_type:
                flash("Please select a target data type.", "warning")
            else:
                original_nan_count = df[col].isnull().sum()
                # original_series = df[col].copy() # Keep copy only if needed for complex revert logic
                converted_series = None
                conversion_error = None
                coerced_count = 0

                try:
                    if target_type == 'string':
                        converted_series = df[col].astype('string') # Use pandas nullable string
                    elif target_type == 'Int64':
                        # Use pandas nullable Int64 type
                        numeric_temp = pd.to_numeric(df[col], errors='coerce')
                        # Check if all coerced values are integer-like before converting
                        if numeric_temp.dropna().apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                            converted_series = numeric_temp.astype('Int64')
                        else:
                            conversion_error = f"Cannot convert to Int64: Column '{col}' contains non-integer values after coercion."
                    elif target_type == 'float64':
                        # Coerce errors during numeric conversion
                        converted_series = pd.to_numeric(df[col], errors='coerce')
                        # Ensure it's explicitly float64 even if source was int
                        if converted_series.notna().any(): # Check if not all NaNs after coercion
                            converted_series = converted_series.astype('float64')
                    elif target_type == 'boolean':
                        # Use pandas nullable BooleanDtype
                        map_dict = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False, 1: True, 0: False, 1.0: True, 0.0: False}
                        # Handle potential non-string values before lowercasing
                        str_series = df[col].astype(str).str.lower().str.strip()
                        mapped_series = str_series.map(map_dict)
                        # Convert unmapped to NA, then to boolean
                        converted_series = mapped_series.astype('boolean')
                    elif target_type == 'datetime64[ns]':
                        converted_series = pd.to_datetime(df[col], errors='coerce')
                    else:
                        conversion_error = f"Unsupported target type: {target_type}"

                    if conversion_error:
                        flash(conversion_error, "danger")
                    elif converted_series is not None:
                        df[col] = converted_series
                        cols_affected.append(col)
                        final_nan_count = df[col].isnull().sum()
                        # Calculate coerced count more accurately
                        coerced_count = final_nan_count - original_nan_count
                        # Adjust if original NaNs were somehow filled during conversion (unlikely here but safe)
                        if coerced_count < 0: coerced_count = 0

                        if coerced_count > 0:
                            flash(f"Converted column '{col}' to {target_type}. {coerced_count} value(s) could not be converted and became missing.", "warning")
                        else:
                            flash(f"Successfully converted column '{col}' to {target_type}.", "success")
                    # else: # Fallback if converted_series is None without error
                    #      flash(f"Conversion to {target_type} failed unexpectedly for column '{col}'.", "danger")

                except Exception as e:
                    flash(f"Error converting column '{col}' to {target_type}: {str(e)}", "danger")

        # <<< NEW Operation 9: Regex Find/Replace >>>
        elif operation == 'regex_replace':
            col = request.form.get('regex_column')
            pattern = request.form.get('regex_pattern')
            replacement = request.form.get('regex_replacement', '') # Default to empty string replacement

            if not col or col not in df.columns:
                flash(f"Column '{col}' not found or not selected.", "warning")
            elif not pattern:
                flash("Regex pattern cannot be empty.", "warning")
            else:
                # Ensure column is string type
                if not is_string_dtype(df[col]) and not is_object_dtype(df[col]):
                    flash(f"Regex replace requires a text column. '{col}' is {df[col].dtype}.", "warning")
                else:
                    try:
                        # Convert to nullable string type to handle potential mix + NaNs correctly
                        df[col] = df[col].astype('string')
                        # Apply regex replace
                        df[col] = df[col].str.replace(pattern, replacement, regex=True)
                        cols_affected.append(col)
                        flash(f"Applied Regex replace on column '{col}'.", "success")
                    except re.error as regex_err:
                        flash(f"Invalid Regular Expression: {regex_err}", "danger")
                    except Exception as e:
                        flash(f"Error during Regex replace on '{col}': {e}", "danger")

        # <<< NEW Operation 10: Apply Fuzzy Matching Changes >>>
        elif operation == 'fuzzy_apply':
            fuzzy_results_from_session = session.get('fuzzy_results')
            if not fuzzy_results_from_session:
                flash("No fuzzy matching results found in session to apply.", "warning")
                return redirect(url_for('clean_data'))

            col = fuzzy_results_from_session['column']
            groups = fuzzy_results_from_session['groups']

            if col not in df.columns:
                flash(f"Column '{col}' from fuzzy analysis not found in current data.", "warning")
                session.pop('fuzzy_results', None) # Clear bad results
                return redirect(url_for('clean_data'))

            # Build the replacement map from the submitted form data
            replacement_map = {}
            applied_count = 0
            try:
                for i, group in enumerate(groups):
                    # Get the chosen canonical value for this group from the form
                    # The input name should be like 'canonical_value_0', 'canonical_value_1', etc.
                    canonical_value = request.form.get(f'canonical_value_{i}')

                    # Only create replacements if a canonical value was actually provided/chosen
                    if canonical_value is not None and canonical_value != "__KEEP_ORIGINAL__": # Check for special value if added
                        for original_value in group:
                            # Don't map the canonical value to itself if it was part of the group
                            if original_value != canonical_value:
                                replacement_map[original_value] = canonical_value
                
                if replacement_map:
                    # Apply all replacements using the map
                    df[col] = df[col].replace(replacement_map)
                    cols_affected.append(col)
                    applied_count = len(replacement_map)
                    flash(f"Applied {applied_count} fuzzy match replacements in column '{col}'.", "success")
                else:
                    flash("No changes selected or applied from fuzzy matching results.", "info")

            except Exception as e:
                flash(f"Error applying fuzzy match changes: {e}", "danger")

            # Clear the fuzzy results from session AFTER attempting to apply
            session.pop('fuzzy_results', None)


        # --- (Existing operations 11-14: constraint, sort, rename, remove) ---
        elif operation == 'constraint':
            col = request.form.get('column')
            min_val_str = request.form.get('min_val')
            max_val_str = request.form.get('max_val')

            if col not in df.columns:
                 flash(f"Column '{col}' not found.", "warning")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                flash(f"Column '{col}' is not numeric. Cannot apply numeric range constraint.", "warning")
            elif not min_val_str and not max_val_str:
                 flash("Please provide at least a minimum or maximum value for the constraint.", "warning")
            else:
                condition = pd.Series([True] * len(df), index=df.index) # Start with all true
                try:
                    if min_val_str:
                        min_val = float(min_val_str)
                        # Handle NaNs correctly in comparison
                        condition &= (df[col] >= min_val) & df[col].notna()
                    if max_val_str:
                        max_val = float(max_val_str)
                        condition &= (df[col] <= max_val) & df[col].notna()

                    df = df[condition] # Keep only rows matching the condition (NaNs excluded unless only one bound specified)
                    # Affects rows
                except ValueError:
                     flash("Invalid number entered for min/max value.", "danger")

        elif operation == 'sort':
            cols = request.form.getlist('columns')
            ascending_str = request.form.get('ascending', 'True')
            ascending = ascending_str == 'True' # Convert string 'True'/'False' to boolean

            valid_cols = [c for c in cols if c in df.columns]
            invalid_cols = [c for c in cols if c not in df.columns]

            if invalid_cols:
                flash(f"Columns not found for sorting: {', '.join(invalid_cols)}", "warning")
            if not valid_cols:
                flash("Please select valid columns to sort by.", "warning")
            else:
                df.sort_values(by=valid_cols, ascending=ascending, inplace=True, ignore_index=True, na_position='last') # ignore_index resets index
                # Affects row order

        elif operation == 'rename':
            old_name = request.form.get('old_name')
            new_name = request.form.get('new_name', '').strip() # Get new name and strip whitespace

            if not old_name or old_name not in df.columns:
                flash(f"Original column '{old_name}' not found or not selected.", "warning")
            elif not new_name:
                flash("New column name cannot be empty.", "warning")
            elif new_name in df.columns and new_name != old_name:
                flash(f"Column name '{new_name}' already exists. Choose a different name.", "warning")
            else:
                df.rename(columns={old_name: new_name}, inplace=True)
                flash(f"Renamed column '{old_name}' to '{new_name}'.", "success")
                # Structure change

        elif operation == 'remove':
            cols_to_remove = request.form.getlist('columns')
            cols_exist = [c for c in cols_to_remove if c in df.columns]
            cols_not_exist = [c for c in cols_to_remove if c not in df.columns]

            if cols_not_exist:
                flash(f"Columns not found and could not be removed: {', '.join(cols_not_exist)}", "warning")

            if cols_exist:
                df.drop(columns=cols_exist, inplace=True)
                flash(f"Removed columns: {', '.join(cols_exist)}", "success")
            elif not cols_not_exist: # Only show if no columns selected AND no invalid columns given
                flash("No columns selected to remove.", "info")
            # Structure change


        else:
            flash(f"Unknown operation: {operation}", "danger")
            return redirect(url_for('clean_data'))

        # --- Save the modified DataFrame ---
        new_file_path = save_data(df, original_filename)
        if new_file_path:
            if file_path != new_file_path and os.path.exists(file_path):
                 try: os.remove(file_path)
                 except OSError as e: print(f"Error removing intermediate file {file_path}: {e}")

            session['current_file_path'] = new_file_path
            final_shape = df.shape
            rows_changed = original_shape[0] - final_shape[0]
            cols_changed = original_shape[1] - final_shape[1]
            change_msg = "" # Calculate change_msg as before...

            if rows_changed > 0: change_msg += f" {rows_changed} rows removed."
            elif rows_changed < 0: change_msg += f" {-rows_changed} rows added."
            if cols_changed > 0: change_msg += f" {cols_changed} columns removed."
            elif cols_changed < 0: change_msg += f" {-cols_changed} columns added."

            if not change_msg and operation not in ['rename', 'sort'] and cols_affected:
                 unique_affected = sorted(list(set(cols_affected)))
                 change_msg = f" Values potentially modified in column(s): {', '.join(unique_affected)}."
            elif not change_msg and operation not in ['rename', 'sort']:
                 value_ops = ['missing', 'outlier', 'smooth', 'normalize', 'date', 'case', 'convert_type'] # Added convert_type
                 structure_ops = ['deduplicate', 'constraint', 'remove']
                 if operation in value_ops or operation in structure_ops:
                      change_msg = " Operation applied, but shape/values appear unchanged (check inputs/data)."
                 else:
                       change_msg = " Operation applied."

            flash(f"Operation '{operation}' applied successfully.{change_msg}", "success")
        else:
            flash("Failed to save changes. No modifications were persisted.", "danger")

        return redirect(url_for('clean_data'))

    # --- Error Handling ---
    except FileNotFoundError:
         flash(f"Error applying operation '{operation}': Data file missing. Please upload again.", "danger")
         session.pop('current_file_path', None)
         session.pop('original_filename', None)
         session.pop('exploration_results', None)
         session.pop('profile_results', None) # <<< Clear profile
         return redirect(url_for('index'))
    except KeyError as e:
        flash(f"Error applying operation '{operation}': Missing column/parameter '{e}'. Check inputs.", "danger")
        return redirect(url_for('clean_data'))
    except ValueError as e:
         flash(f"Error applying operation '{operation}': Invalid value/data type mismatch - {e}. Check inputs/types.", "danger")
         return redirect(url_for('clean_data'))
    except MemoryError:
         flash(f"Error applying operation '{operation}': Insufficient memory.", "danger")
         return redirect(url_for('clean_data'))
    except Exception as e:
        print(f"Unexpected error during operation '{operation}': {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        flash(f"Unexpected error during operation '{operation}': {type(e).__name__}. Check data/parameters.", "danger")
        # Try redirecting back, state might be ok, but changes weren't saved
        return redirect(url_for('clean_data'))



@app.route('/visualize')
def visualize_data():
    """Generates and displays basic data visualizations."""
    file_path = session.get('current_file_path')
    original_filename = session.get('original_filename')

    if not file_path or not original_filename or not os.path.exists(file_path):
        flash("No data loaded to visualize. Please upload a file first.", "warning")
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df.infer_objects()

        if df.empty:
            flash("Cannot visualize empty dataset.", "warning")
            return redirect(url_for('clean_data'))

        plots = {'histograms': [], 'bar_charts': [], 'scatter_plots': []}
        max_categories_bar = 15 # Limit categories for bar charts
        max_scatter_plots = 10 # Limit number of scatter plots for performance

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Consider object and specific string type for categorical
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        # --- Generate Histograms for Numeric Columns ---
        for col in numeric_cols:
            try:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}', template='plotly_white')
                fig.update_layout(bargap=0.1, title_x=0.5, height=350, margin=dict(l=40, r=20, t=40, b=30))
                plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                plots['histograms'].append({'title': f'Distribution of {col}', 'div': plot_div})
            except Exception as e:
                print(f"Could not generate histogram for {col}: {e}")
                flash(f"Could not generate histogram for '{col}'.", "warning")


        # --- Generate Bar Charts for Categorical Columns ---
        for col in categorical_cols:
            # Exclude high cardinality columns for basic bar charts
            unique_count = df[col].nunique()
            if 0 < unique_count <= max_categories_bar:
                 try:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count'] # Rename columns for px.bar
                    fig = px.bar(counts, x=col, y='count', title=f'Counts for {col}', template='plotly_white')
                    fig.update_layout(title_x=0.5, height=350, margin=dict(l=40, r=20, t=40, b=30))
                    plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                    plots['bar_charts'].append({'title': f'Counts for {col}', 'div': plot_div})
                 except Exception as e:
                    print(f"Could not generate bar chart for {col}: {e}")
                    flash(f"Could not generate bar chart for '{col}'.", "warning")
            elif unique_count > max_categories_bar:
                 flash(f"Skipping bar chart for '{col}': Too many unique values ({unique_count} > {max_categories_bar}).", "info")


        # --- Generate Scatter Plots for pairs of Numeric Columns (Limited) ---
        scatter_count = 0
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if scatter_count >= max_scatter_plots:
                        break
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    try:
                        fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}', template='plotly_white', opacity=0.6)
                        fig.update_layout(title_x=0.5, height=400, margin=dict(l=40, r=20, t=40, b=30))
                        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                        plots['scatter_plots'].append({'title': f'{col1} vs {col2}', 'div': plot_div})
                        scatter_count += 1
                    except Exception as e:
                        print(f"Could not generate scatter plot for {col1} vs {col2}: {e}")
                        flash(f"Could not generate scatter plot for '{col1}' vs '{col2}'.", "warning")
                if scatter_count >= max_scatter_plots:
                    flash(f"Stopped generating scatter plots after {max_scatter_plots} due to limit.", "info")
                    break


        return render_template('visualize.html',
                               filename=original_filename,
                               plots=plots)

    except FileNotFoundError:
         flash("Error: Could not find the data file for visualization.", "danger")
         return redirect(url_for('clean_data'))
    except Exception as e:
        flash(f"An error occurred generating visualizations: {str(e)}", "danger")
        # Redirect back to clean page if visualization fails catastrophically
        return redirect(url_for('clean_data'))



# <<< MODIFIED: Handle different download formats >>>
@app.route('/download')
def download_file():
    """Provides the current cleaned file for download in chosen format."""
    file_path = session.get('current_file_path') # Path to the intermediate CSV
    original_filename = session.get('original_filename', 'data.csv') # Fallback name
    # <<< Get requested format from query parameter >>>
    requested_format = request.args.get('format', 'csv').lower() # Default to csv

    if not file_path or not os.path.exists(file_path):
        flash("No file available for download or session expired.", "warning")
        return redirect(url_for('clean_data')) # Redirect back to clean page

    # Secure the base filename from original upload
    base, _ = os.path.splitext(original_filename)
    secure_base = secure_filename(base)
    if not secure_base: secure_base = "data"

    try:
        # Read the latest intermediate CSV file
        df = pd.read_csv(file_path, encoding='utf-8')

        if requested_format == 'xlsx':
            output_filename = f"{secure_base}_cleaned.xlsx"
            # Use BytesIO for in-memory processing
            excel_buffer = io.BytesIO()
            # Write to buffer using openpyxl engine
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                 df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            # Important: Seek to the beginning of the stream
            excel_buffer.seek(0)

            response = make_response(excel_buffer.getvalue())
            response.headers['Content-Disposition'] = f'attachment; filename="{output_filename}"'
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            return response

        elif requested_format == 'csv':
            output_filename = f"{secure_base}_cleaned.csv"
            # Send the existing intermediate CSV file directly
            return send_from_directory(
                directory=app.config['CLEANED_FOLDER'],
                path=os.path.basename(file_path),
                as_attachment=True,
                download_name=output_filename # Use the derived name
            )
        else:
            flash(f"Unsupported download format requested: {requested_format}", "warning")
            return redirect(url_for('clean_data'))

    except FileNotFoundError:
        flash("Error: Could not find the intermediate data file for download.", "danger")
        return redirect(url_for('clean_data'))
    except Exception as e:
        flash(f"Error preparing file for download: {str(e)}", "danger")
        return redirect(url_for('clean_data'))
    

# --- (Cleanup function definition - unchanged) ---
def cleanup_old_files(folder, max_age_seconds=3600 * 24):
    """Removes files older than max_age_seconds from the specified folder."""
    """Removes files older than max_age_seconds from the specified folder."""
    now = time.time()
    cleaned_count = 0
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                 if os.path.isfile(file_path):
                     if os.stat(file_path).st_mtime < now - max_age_seconds:
                         os.remove(file_path)
                         cleaned_count += 1
                         # print(f"Cleaned up old file: {filename}") # Optional log
            except Exception as e:
                 print(f"Error processing file {filename} during cleanup: {e}") # Log error for specific file
        if cleaned_count > 0:
             print(f"Cleaned up {cleaned_count} old files from {folder}.")
    except Exception as e:
        print(f"Error during cleanup of folder {folder}: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # Create folders if they don't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(CLEANED_FOLDER):
        os.makedirs(CLEANED_FOLDER)
    if not os.path.exists('static/css'):
         os.makedirs('static/css', exist_ok=True)

    # Optional: Cleanup on start
    # cleanup_old_files(app.config['UPLOAD_FOLDER'])
    # cleanup_old_files(app.config['CLEANED_FOLDER'])

    port = int(os.environ.get('PORT', 5000))
    # Set debug=False for production!
    app.run(host='0.0.0.0', port=port, debug=True)
