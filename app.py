import streamlit as st
import os
import uuid
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import chardet # For encoding detection
from collections import Counter
import time
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype, is_string_dtype, is_datetime64_any_dtype, is_object_dtype, is_bool_dtype
import io
from thefuzz import fuzz
import re


# --- Configuration ---
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_FILE_SIZE_MB = 32 # MB limit for Streamlit upload widget
# <<< Define the key for session state tracking >>>
OPERATION_APPLIED_KEY = "operation_applied_flag"


# --- Helper Functions ---

# @st.cache_data # Caching can be tricky with file uploads, use carefully
def detect_encoding(file_bytes):
    """Detects file encoding using chardet from bytes."""
    try:
        # Read a portion of the bytes
        raw_data = file_bytes[:50000]
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        # Treat ascii as utf-8
        if encoding and encoding.lower() == 'ascii':
            encoding = 'utf-8'
        return encoding, result['confidence']
    except Exception:
        return 'utf-8', 0.0 # Default fallback

# @st.cache_data # Caching can speed this up if the same file bytes are processed
def load_data(uploaded_file, user_encoding=None):
    """Loads data from uploaded CSV or XLSX file into a pandas DataFrame."""
    df = None
    used_encoding = None
    file_extension = os.path.splitext(uploaded_file.name)[1].lower().strip('.')

    if file_extension == 'csv':
        file_bytes = uploaded_file.getvalue()
        detected_encoding = None
        confidence = 0.0

        if not user_encoding:
            detected_encoding, confidence = detect_encoding(file_bytes)
            encoding_to_try = detected_encoding
            st.info(f"Auto-detected encoding: {encoding_to_try} (Confidence: {confidence:.2f}). You can specify encoding below if needed.")
        else:
            encoding_to_try = user_encoding
            st.info(f"Using specified encoding: {encoding_to_try}")

        used_encoding = encoding_to_try

        try:
            # Use io.BytesIO to treat bytes as a file
            bytes_io = io.BytesIO(file_bytes)
            # Try parsing dates during loading for better initial types
            try:
                df = pd.read_csv(bytes_io, encoding=encoding_to_try, parse_dates=True, infer_datetime_format=True)
            except ValueError: # Catch specific error if inferring fails badly
                 bytes_io.seek(0)
                 st.warning("Could not automatically parse dates during initial load. You may need to use the 'Standardize Date Column' operation.", icon="⚠️")
                 df = pd.read_csv(bytes_io, encoding=encoding_to_try)
            except Exception: # Fallback without date parsing if it fails for other reasons
                bytes_io.seek(0) # Reset pointer
                df = pd.read_csv(bytes_io, encoding=encoding_to_try)


            st.toast(f"Successfully loaded CSV with encoding '{used_encoding}'.", icon="✅")

        except UnicodeDecodeError as e:
            st.error(f"Error decoding CSV with '{encoding_to_try}'. Common alternatives: 'utf-8', 'latin-1', 'cp1252'. Please specify encoding manually. Error: {e}")
            return None, encoding_to_try # Return encoding that failed
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None, used_encoding

    elif file_extension == 'xlsx':
        try:
            # Read the first sheet by default
            excel_file = pd.ExcelFile(uploaded_file) # Can read directly from UploadedFile
            sheet_names = excel_file.sheet_names
            if not sheet_names:
                st.warning("Excel file contains no sheets.")
                return None, None
            first_sheet_name = sheet_names[0]
            # Try parsing dates during loading for better initial types
            try:
                 df = pd.read_excel(uploaded_file, sheet_name=first_sheet_name, engine='openpyxl', parse_dates=True, infer_datetime_format=True)
            except ValueError: # Catch specific error if inferring fails badly
                 st.warning("Could not automatically parse dates during initial load. You may need to use the 'Standardize Date Column' operation.", icon="⚠️")
                 df = pd.read_excel(uploaded_file, sheet_name=first_sheet_name, engine='openpyxl')
            except Exception: # Fallback without date parsing
                 df = pd.read_excel(uploaded_file, sheet_name=first_sheet_name, engine='openpyxl')


            if len(sheet_names) > 1:
                st.info(f"Read data from the first sheet ('{first_sheet_name}'). File contains multiple sheets: {', '.join(sheet_names)}.")
            else:
                st.info(f"Read data from sheet '{first_sheet_name}'.")
            used_encoding = 'xlsx' # Indicate format
            st.toast("Successfully loaded XLSX file.", icon="✅")

        except Exception as e:
            st.error(f"Error loading XLSX file: {str(e)}")
            return None, None
    else:
        st.error(f"Unsupported file extension: {file_extension}. Please upload CSV or XLSX.")
        return None, None

    if df is not None:
        # Attempt basic type inference improvement
        df = df.infer_objects()
    return df, used_encoding

# --- Data Exploration & Profiling Functions (Mostly unchanged logic, removed flash messages) ---

# @st.cache_data # Good candidate for caching if df is hashable
def auto_explore_data(df):
    """Analyzes the DataFrame and returns a list of findings/suggestions."""
    findings = []
    if df is None or df.empty:
        findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'No data loaded or DataFrame is empty.', 'details': [], 'suggestion': 'Upload a valid data file.'})
        return findings

    num_rows, num_cols = df.shape
    if num_rows == 0 or num_cols == 0:
         findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'DataFrame has zero rows or columns.', 'details': [], 'suggestion': 'Check the uploaded file content.'})
         return findings

    # Finding 1: Missing Data Analysis
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if not missing_cols.empty:
        total_missing = missing_cols.sum()
        pct_total_missing = (total_missing / (num_rows * num_cols)) * 100 if num_rows * num_cols > 0 else 0
        findings.append({
            'issue_type': 'Missing Data',
            'severity': 'High' if pct_total_missing > 10 else ('Medium' if pct_total_missing > 1 else 'Low'),
            'message': f"Found {total_missing} missing values ({pct_total_missing:.2f}% of total cells) across {len(missing_cols)} column(s).",
            'details': [f"'{col}': {count} missing ({ (count/num_rows)*100:.1f}%)" for col, count in missing_cols.items()],
            'suggestion': "Use '1. Clean Missing Data' to handle NaNs (fill or drop)."
        })

    # Finding 2: Data Type Overview & Potential Issues
    dtypes = df.dtypes
    object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist() # Check if already datetime

    findings.append({
        'issue_type': 'Data Types',
        'severity': 'Info',
        'message': f"Dataset has {num_cols} columns: {len(numeric_cols)} numeric, {len(object_cols)} text/object, {len(datetime_cols)} datetime.",
        'details': [f"'{col}': {str(dtype)}" for col, dtype in dtypes.items()], # Ensure dtype is string
        'suggestion': "Review data types using the 'Data Profile' feature or use '8. Convert Data Type'."
    })

    # Finding 3: Potential Outliers (Using IQR for simplicity)
    outlier_suggestions = []
    for col in numeric_cols:
        if df[col].isnull().all(): continue # Skip fully NaN columns
        if df[col].nunique() < 2: continue # Skip constant columns
        try:
            # Convert to float before calculating quantiles for nullable Int types
            numeric_data = df[col].astype(float)
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or pd.isna(IQR): continue # Skip if IQR is zero or NaN
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if pd.isna(lower_bound) or pd.isna(upper_bound): continue

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                outlier_count = outliers.count()
                pct_outliers = (outlier_count / num_rows) * 100 if num_rows > 0 else 0
                if pct_outliers > 0.1: # Only suggest if more than 0.1%
                     outlier_suggestions.append(f"'{col}': {outlier_count} potential outliers ({pct_outliers:.2f}%)")
        except Exception as e:
             print(f"Explore Outlier check failed for column '{col}': {e}") # Keep print for debugging
             # Optionally add a finding about the failure
             # outlier_suggestions.append(f"'{col}': Could not reliably check for outliers.")

    if outlier_suggestions:
         findings.append({
            'issue_type': 'Potential Outliers',
            'severity': 'Medium',
            'message': f"Potential outliers detected in {len(outlier_suggestions)} numeric column(s) using the IQR method.",
            'details': outlier_suggestions,
            'suggestion': "Use '2. Handle Outliers' to investigate and handle (remove or cap)."
        })

    # Finding 4: Duplicate Rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        pct_duplicates = (duplicate_count / num_rows) * 100 if num_rows > 0 else 0
        findings.append({
            'issue_type': 'Duplicate Records',
            'severity': 'Medium' if pct_duplicates > 5 else 'Low',
            'message': f"Found {duplicate_count} duplicate rows ({pct_duplicates:.2f}% of total).",
            'details': [],
            'suggestion': "Use '5. Deduplicate Records' to remove identical rows."
        })

    # Finding 5: Low Variance / Constant Columns
    constant_cols = []
    low_variance_cols = []
    if num_rows > 0:
        for col in df.columns:
            nunique = df[col].nunique(dropna=False)
            if nunique == 1:
                 constant_cols.append(col)
            # Check low variance only if not constant and has reasonable number of rows
            elif nunique > 1 and num_rows > 10 and (nunique / num_rows) < 0.01:
                 low_variance_cols.append(col)

    low_var_msgs = []
    if constant_cols:
        low_var_msgs.append(f"Constant columns: {', '.join([f'`{c}`' for c in constant_cols])}") # Use markdown for code
    if low_variance_cols:
         low_var_msgs.append(f"Low variance columns (<1% unique): {', '.join([f'`{c}`' for c in low_variance_cols])}")

    if low_var_msgs:
        findings.append({
            'issue_type': 'Low Variance',
            'severity': 'Low',
            'message': "Found columns with very few unique values.",
            'details': low_var_msgs,
            'suggestion': "Consider removing using '14. Remove Column(s)' if not informative."
        })

    # Finding 6: Potential Date Columns (in Object types)
    potential_date_cols = []
    for col in object_cols:
        if is_datetime64_any_dtype(df[col]): continue # Skip if already datetime
        if df[col].isnull().all(): continue
        non_na_series = df[col].dropna()
        sample_size = min(100, len(non_na_series))
        if sample_size == 0: continue
        try:
            # Increased sample robustness check
            sample = non_na_series.sample(sample_size, random_state=1)
            # Ensure sample elements are treated as strings for checks
            sample_str = sample.astype(str)
            num_digits = sample_str.str.count(r'\d').mean()
            num_chars = sample_str.str.len().mean()

            # Basic heuristic: Needs enough digits, shouldn't be purely numeric already
            # Check if the first element looks numeric (more robust than checking the whole series dtype)
            first_val_numeric = False
            if not non_na_series.empty:
                try:
                    pd.to_numeric(non_na_series.iloc[0])
                    first_val_numeric = True
                except (ValueError, TypeError):
                    first_val_numeric = False

            if num_chars > 3 and num_digits / num_chars > 0.5 and not first_val_numeric:
                with np.errstate(all='ignore'): # Suppress warnings during parse attempt
                    # Try parsing the sample
                    parsed_dates = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True) # Use infer
                parseable_ratio = parsed_dates.notna().sum() / sample_size
                if parseable_ratio > 0.7: # 70% parse success on sample
                    potential_date_cols.append(f"'{col}'")
        except Exception as e:
             print(f"Explore Date check failed for column '{col}': {e}")

    if potential_date_cols:
         findings.append({
            'issue_type': 'Potential Dates',
            'severity': 'Info',
            'message': f"Found {len(potential_date_cols)} text column(s) that might contain dates.",
            'details': potential_date_cols,
            'suggestion': "Use '6. Standardize Date Column' or '8. Convert Data Type'."
        })

    # Finding 7: Text Issues (Case & Whitespace) - Check sample
    text_issue_cols = []
    sample_size = min(500, num_rows)
    if sample_size > 0:
        df_sample = df.sample(sample_size, random_state=1)
        for col in object_cols:
            if df_sample[col].isnull().all(): continue
            try:
                # Use nullable string type for robust checks
                col_str = df_sample[col].astype('string').dropna()
                if col_str.empty: continue

                has_whitespace = (col_str != col_str.str.strip()).any()
                unique_vals = col_str.unique()
                has_mixed_case = False
                if len(unique_vals) > 1:
                    lower_unique_count = col_str.str.lower().nunique()
                    if lower_unique_count < len(unique_vals):
                         # More nuanced check: ignore if only difference is title case vs lower/upper
                         title_unique_count = col_str.str.title().nunique()
                         upper_unique_count = col_str.str.upper().nunique()
                         if not (lower_unique_count == title_unique_count == upper_unique_count):
                              has_mixed_case = True

                if has_whitespace or has_mixed_case:
                    issues = []
                    if has_whitespace: issues.append("leading/trailing whitespace")
                    if has_mixed_case: issues.append("mixed casing")
                    text_issue_cols.append(f"'{col}': Contains {' and '.join(issues)} (based on sample)")
            except Exception as e:
                print(f"Explore Text check failed for column '{col}': {e}")

    if text_issue_cols:
         findings.append({
            'issue_type': 'Text Formatting',
            'severity': 'Low',
            'message': f"Potential text formatting issues found in {len(text_issue_cols)} column(s).",
            'details': text_issue_cols,
            'suggestion': "Use '7. Standardize Case & Whitespace' to clean text."
        })

    # Finding 8: High Cardinality Text Columns
    high_cardinality_cols = []
    # More reasonable threshold: e.g., > 50 unique values AND > 20% of rows are unique
    high_card_thresh_abs = 50
    high_card_thresh_rel = 0.20
    if num_rows > 0:
        for col in object_cols:
            try:
                unique_count = df[col].nunique()
                unique_ratio = unique_count / num_rows
                if unique_count > high_card_thresh_abs and unique_ratio > high_card_thresh_rel:
                     # Further check: if almost all values are unique, it might be an ID
                     if unique_ratio < 0.95: # Don't flag likely IDs
                        high_cardinality_cols.append(f"'{col}': {unique_count} unique values ({unique_ratio*100:.1f}%)")
            except Exception as e:
                # Ignore errors for columns with non-hashable types (like lists/dicts)
                if "unhashable type" not in str(e):
                    print(f"Explore Cardinality check failed for column '{col}': {e}")


    if high_cardinality_cols:
         findings.append({
            'issue_type': 'High Cardinality Text',
            'severity': 'Info',
            'message': f"Found {len(high_cardinality_cols)} text column(s) with many distinct values.",
            'details': high_cardinality_cols,
            'suggestion': "Review if these are categorical features needing cleaning/grouping (e.g., via '10. Fuzzy Match Text Grouping') or identifiers."
        })

    return findings

# @st.cache_data # Good candidate for caching
def generate_profile(df):
    """Generates detailed statistics for each column."""
    profile = {}
    if df is None or df.empty:
        return {"error": "DataFrame is empty or not loaded."}

    total_rows = len(df)
    profile['__overview__'] = { # Add overview stats
        'rows': total_rows,
        'columns': len(df.columns),
        'total_cells': total_rows * len(df.columns),
        'total_missing': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum())
    }

    for col in df.columns:
        column_data = df[col]
        stats = {}

        # Common Stats
        stats['dtype'] = str(column_data.dtype)
        stats['count'] = int(column_data.count()) # Non-missing count
        stats['missing_count'] = int(column_data.isnull().sum())
        stats['missing_percent'] = round((stats['missing_count'] / total_rows) * 100, 2) if total_rows > 0 else 0
        try: # Unique count can fail on unhashable types
             stats['unique_count'] = int(column_data.nunique())
             stats['unique_percent'] = round((stats['unique_count'] / total_rows) * 100, 2) if total_rows > 0 else 0
        except Exception:
             stats['unique_count'] = "Error (unhashable?)"
             stats['unique_percent'] = "N/A"


        # Type-Specific Stats
        try: # Wrap type-specific analysis in try-except
            if is_numeric_dtype(column_data):
                stats['type'] = 'Numeric'
                # Use .astype(float) before describe for Int64/nullable types to ensure stats calc correctly
                desc = column_data.astype(float).describe()
                stats['mean'] = round(desc.get('mean', np.nan), 4)
                stats['std'] = round(desc.get('std', np.nan), 4)
                stats['min'] = round(desc.get('min', np.nan), 4)
                stats['25%'] = round(desc.get('25%', np.nan), 4)
                stats['50%'] = round(desc.get('50%', np.nan), 4) # Median
                stats['75%'] = round(desc.get('75%', np.nan), 4)
                stats['max'] = round(desc.get('max', np.nan), 4)
                try: # Skew and Kurt can fail on edge cases
                    # Ensure skipna=True
                    stats['skewness'] = round(column_data.astype(float).skew(skipna=True), 4) if stats['count'] > 2 else np.nan
                    stats['kurtosis'] = round(column_data.astype(float).kurt(skipna=True), 4) if stats['count'] > 3 else np.nan
                except Exception:
                    stats['skewness'] = np.nan
                    stats['kurtosis'] = np.nan

            elif is_datetime64_any_dtype(column_data):
                stats['type'] = 'Datetime'
                non_na_dates = column_data.dropna()
                stats['min_date'] = str(non_na_dates.min()) if not non_na_dates.empty else 'N/A'
                stats['max_date'] = str(non_na_dates.max()) if not non_na_dates.empty else 'N/A'

            elif is_bool_dtype(column_data):
                stats['type'] = 'Boolean'
                value_counts = column_data.value_counts(dropna=False)
                stats['true_count'] = int(value_counts.get(True, 0))
                stats['false_count'] = int(value_counts.get(False, 0))
                # Account for potential NAs in nullable boolean
                stats['na_count'] = int(value_counts.get(pd.NA, value_counts.get(None, 0))) # Check for pd.NA or None

            # Check for pandas StringDtype explicitly, or fallback to object
            elif is_string_dtype(column_data) or is_object_dtype(column_data):
                stats['type'] = 'Text/Object'
                # Get top 5 most frequent values (handle potential non-hashable types)
                try:
                    # Ensure we handle actual NaN/None correctly in value counts
                    value_counts = column_data.value_counts(dropna=False).head(5)
                    stats['top_values'] = {str(k) if pd.notna(k) else 'NaN/None': int(v) for k, v in value_counts.items()}
                except TypeError:
                    stats['top_values'] = {"Error": "Contains non-hashable types"}
                except Exception as e_vc:
                    stats['top_values'] = {"Error": f"Could not get value counts: {e_vc}"}


                # Basic stats on string length (ignoring NaNs)
                try:
                    # Use astype("string") for potentially mixed types -> pd.NA
                    str_series = column_data.astype("string").dropna()
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
                # Catch-all for other types like 'category', etc.
                stats['type'] = f"Other ({stats['dtype']})"


        except Exception as e:
            stats['error'] = f"Error during analysis: {str(e)}"

        profile[col] = stats
    return profile

# --- Function to convert DF to downloadable format ---
# @st.cache_data # Cache the conversion result based on the DataFrame
def convert_df_to_csv(df):
   """Converts DataFrame to CSV bytes."""
   # Use utf-8-sig to ensure BOM for Excel compatibility with UTF-8 chars
   return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

# @st.cache_data
def convert_df_to_excel(df):
    """Converts DataFrame to Excel XLSX bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    # No need to writer.save(), context manager handles it
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Data Cleaner Pro")

st.title("🧹 Data Cleaner Pro")
st.markdown("Upload your CSV or XLSX file, explore, clean, and download the results.")

# --- Session State Initialization ---
# Store the main DataFrame, upload info, and analysis results here
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None
if 'source_format_info' not in st.session_state: # Stores encoding or 'xlsx'
    st.session_state.source_format_info = None
if 'exploration_results' not in st.session_state:
    st.session_state.exploration_results = None
if 'profile_results' not in st.session_state:
    st.session_state.profile_results = None
if 'fuzzy_results' not in st.session_state:
    st.session_state.fuzzy_results = None # Stores {'column': col, 'threshold': threshold, 'groups': groups}
if 'operation_log' not in st.session_state:
     st.session_state.operation_log = [] # Track applied operations
if OPERATION_APPLIED_KEY not in st.session_state: # Use the defined constant
    st.session_state[OPERATION_APPLIED_KEY] = False


# --- Upload Section ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=list(ALLOWED_EXTENSIONS), key="file_uploader")

    # Encoding selection (only relevant for CSV)
    # Need to access the uploaded file object *before* calling load_data if we want to show this conditionally
    show_encoding_input = False
    if uploaded_file and os.path.splitext(uploaded_file.name)[1].lower().strip('.') == 'csv':
         show_encoding_input = True

    user_encoding = None
    if show_encoding_input:
        # Use selectbox for common encodings + custom input
        common_encodings = ["Auto-detect", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
        selected_encoding = st.selectbox("Specify CSV Encoding:", common_encodings, index=0, key="encoding_select")
        if selected_encoding != "Auto-detect":
            user_encoding = selected_encoding
        custom_encoding = st.text_input("Or enter custom encoding:", key="custom_encoding")
        if custom_encoding:
            user_encoding = custom_encoding


    if uploaded_file is not None:
        # Use a button to explicitly trigger loading, preventing reload on every interaction
        if st.button("Load Data", key="load_data_button", type="primary"):
            # Clear previous state when loading new file
            st.session_state.df = None
            st.session_state.original_filename = None
            st.session_state.source_format_info = None
            st.session_state.exploration_results = None
            st.session_state.profile_results = None
            st.session_state.fuzzy_results = None
            st.session_state.operation_log = []
            st.session_state[OPERATION_APPLIED_KEY] = False # Reset flag


            with st.spinner(f"Loading {uploaded_file.name}..."):
                start_time = time.time()
                df, source_info = load_data(uploaded_file, user_encoding if user_encoding else None)
                duration = time.time() - start_time
                if df is not None:
                    st.session_state.df = df
                    st.session_state.original_filename = uploaded_file.name
                    st.session_state.source_format_info = source_info
                    st.success(f"Loaded successfully in {duration:.2f}s!")
                    # Automatically run exploration after successful load? Optional.
                    with st.spinner("Running initial exploration..."):
                       st.session_state.exploration_results = auto_explore_data(st.session_state.df)
                       st.session_state.profile_results = None # Clear profile
                    st.rerun() # Rerun to update main page display
                else:
                    # Error messages handled within load_data
                    pass # Keep the uploader state

    st.divider()
    st.header("Operation Log")
    if st.session_state.operation_log:
        # Display log in reverse chronological order (most recent first)
        st.dataframe(pd.DataFrame(st.session_state.operation_log, columns=["Log"]), height=200, use_container_width=True)
    else:
        st.info("No operations applied yet.")

# --- Main Display Area (conditional on data loaded) ---
if st.session_state.df is not None:
    df = st.session_state.df # Get the current dataframe from state

    # --- Display Header ---
    header_cols = st.columns([3, 1, 1])
    with header_cols[0]:
        st.header(f"Current Data: `{st.session_state.original_filename}`")
    with header_cols[1]:
        st.metric("Rows", f"{len(df):,}") # Add comma formatting
    with header_cols[2]:
        st.metric("Columns", len(df.columns))

    tab_titles = ["📊 Data Preview & Analysis", "🧼 Cleaning Operations", "💾 Download"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    # --- Tab 1: Data Preview & Analysis ---
    with tab1:
        st.subheader("Data Preview")
        preview_rows = st.slider("Number of rows to preview:", 5, min(100, len(df)), 10, key="preview_slider")
        # Use st.dataframe for better interactivity, consider st.data_editor for editing (more advanced)
        st.dataframe(df.head(preview_rows))

        st.divider()
        st.subheader("Data Analysis")
        analysis_cols = st.columns(2)
        with analysis_cols[0]:
            if st.button("🔍 Run Auto Explore", key="explore_button", help="Analyze data for common issues like missing values, outliers, duplicates etc."):
                with st.spinner("Analyzing data for potential issues..."):
                    start_time = time.time()
                    st.session_state.exploration_results = auto_explore_data(df)
                    st.session_state.profile_results = None # Clear profile results
                    st.session_state.fuzzy_results = None # Clear fuzzy results
                    duration = time.time() - start_time
                    st.toast(f"Auto Explore completed in {duration:.2f}s!", icon="🔍")
                    st.rerun() # Update display

        with analysis_cols[1]:
            if st.button("📈 Generate Data Profile", key="profile_button", help="Calculate detailed statistics for each column."):
                with st.spinner("Generating detailed column profiles..."):
                    start_time = time.time()
                    st.session_state.profile_results = generate_profile(df)
                    st.session_state.exploration_results = None # Clear explore results
                    st.session_state.fuzzy_results = None # Clear fuzzy results
                    duration = time.time() - start_time
                    st.toast(f"Profile generated in {duration:.2f}s!", icon="📈")
                    st.rerun() # Update display

        # --- Display Analysis Results ---
        # Use columns to display Explore and Profile side-by-side if both exist
        results_col1, results_col2 = st.columns(2)

        with results_col1:
            if st.session_state.exploration_results:
                st.subheader("💡 Auto Explore Findings")
                findings = st.session_state.exploration_results
                if not findings or (len(findings) == 1 and findings[0]['issue_type'] == 'Data Error'):
                     st.info("No significant issues automatically detected, or data is empty.")
                else:
                     # Sort findings by severity (High > Medium > Low > Info)
                     severity_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Info': 3, 'Data Error': -1}
                     findings.sort(key=lambda x: severity_order.get(x.get('severity'), 99))

                     for finding in findings:
                         severity = finding.get('severity', 'Info')
                         icon = "🔥" if severity == 'High' else ("⚠️" if severity == 'Medium' else ("ℹ️" if severity == 'Low' else "✅")) # Updated icons
                         with st.expander(f"{icon} **{finding['issue_type']}** ({severity}): {finding['message']}", expanded=severity in ['High', 'Medium']):
                             if finding.get('details'):
                                 st.markdown("**Details:**")
                                 for detail in finding['details']:
                                     st.markdown(f"- {detail}") # Already markdown-safe in generation
                             if finding.get('suggestion'):
                                 st.markdown(f"**Suggestion:** _{finding['suggestion']}_") # Italicize suggestion

        with results_col2:
            if st.session_state.profile_results:
                st.subheader("📊 Data Profile Overview")
                profile = st.session_state.profile_results
                if "error" in profile:
                    st.error(profile["error"])
                else:
                    overview = profile.get('__overview__', {})
                    if overview:
                         total_cells = overview.get('total_cells', 0)
                         total_missing = overview.get('total_missing', 0)
                         total_rows = overview.get('rows', 0)
                         duplicate_rows = overview.get('duplicate_rows', 0)

                         missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
                         duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0

                         st.markdown(f"**Dataset Shape:** {overview.get('rows', 0):,} rows, {overview.get('columns', 0):,} columns ({total_cells:,} cells).")
                         st.markdown(f"**Missing Cells:** {total_missing:,} ({missing_pct:.1f}%).")
                         st.markdown(f"**Duplicate Rows:** {duplicate_rows:,} ({duplicate_pct:.1f}%).")
                         st.divider()

                    # Display profile per column - Use a selectbox for columns instead of expanders everywhere
                    st.subheader("Column Details")
                    profile_search = st.selectbox("Select column to view profile:",
                                                  options=["--- Select Column ---"] + sorted([c for c in profile.keys() if c != '__overview__']),
                                                  key="profile_select_col")

                    if profile_search != "--- Select Column ---":
                        col_name = profile_search
                        stats = profile.get(col_name, {})
                        if not stats: st.warning("No profile data found for this column."); st.stop() # Should not happen

                        if stats.get('error'):
                             st.error(f"Analysis Error for {col_name}: {stats['error']}")
                        else:
                             type_icon = "🔢" if stats.get('type') == 'Numeric' else \
                                         "🅰️" if stats.get('type') == 'Text/Object' else \
                                         "📅" if stats.get('type') == 'Datetime' else \
                                         "☑️" if stats.get('type') == 'Boolean' else \
                                         "❔" # Other
                             st.markdown(f"#### {type_icon} {col_name} ({stats.get('dtype', 'N/A')})")

                             prof_c1, prof_c2 = st.columns(2)
                             with prof_c1:
                                st.markdown(f"**Type:** {stats.get('type', 'N/A')}")
                                st.markdown(f"**Missing:** {stats.get('missing_count', 0)} ({stats.get('missing_percent', 0):.1f}%)")
                                # Prepare unique count and percentage strings separately for clarity
                                unique_count_val = stats.get('unique_count', 'N/A')
                                unique_percent_str = ""
                                # Only calculate and add percentage if unique_count is a valid integer
                                if isinstance(unique_count_val, int):
                                    unique_percent_val = stats.get("unique_percent", 0)
                                    unique_percent_str = f" ({unique_percent_val:.1f}%)"
                                # Now display using the prepared strings
                                st.markdown(f"**Unique:** {unique_count_val}{unique_percent_str}")

                             with prof_c2:
                                 if stats.get('type') == 'Numeric':
                                    st.markdown(f"**Mean:** {stats.get('mean', 'N/A')} | **Median:** {stats.get('50%', 'N/A')}")
                                    st.markdown(f"**Std Dev:** {stats.get('std', 'N/A')}")
                                    st.markdown(f"**Min:** {stats.get('min', 'N/A')} | **Max:** {stats.get('max', 'N/A')}")
                                    st.markdown(f"**Skew:** {stats.get('skewness', 'N/A')} | **Kurt:** {stats.get('kurtosis', 'N/A')}")
                                 elif stats.get('type') == 'Datetime':
                                    st.markdown(f"**Min:** {stats.get('min_date', 'N/A')}")
                                    st.markdown(f"**Max:** {stats.get('max_date', 'N/A')}")
                                 elif stats.get('type') == 'Boolean':
                                    st.markdown(f"**True:** {stats.get('true_count', 0)} | **False:** {stats.get('false_count', 0)} | **NA:** {stats.get('na_count', 0)}")
                                 elif stats.get('type') == 'Text/Object':
                                    st.markdown("**Length (Min/Mean/Max):**")
                                    st.markdown(f"{stats.get('min_length', 'N/A')} / {stats.get('mean_length', 'N/A')} / {stats.get('max_length', 'N/A')}")
                                    if 'top_values' in stats:
                                         st.markdown("**Top 5 Values:**")
                                         top_vals = stats['top_values']
                                         if isinstance(top_vals, dict) and "Error" not in top_vals:
                                            sorted_top_vals = sorted(top_vals.items(), key=lambda item: item[1], reverse=True)
                                            for val, count in sorted_top_vals:
                                                st.markdown(f"- `{val}`: {count:,}") # Use code formatting, add comma separator
                                         elif isinstance(top_vals, dict):
                                             st.warning(top_vals["Error"])
                                         else:
                                             st.write(top_vals) # Fallback


    # --- Tab 2: Cleaning Operations ---
    with tab2:
        st.subheader("Apply Cleaning Operations")
        st.markdown("Select an operation, choose parameters, and click 'Apply'. Changes modify the data in memory.")

        # Get current columns for selection widgets
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Include category for text-like operations? Generally safe.
        text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        # Use string 'datetime' or 'datetime64' which pandas understands for select_dtypes
        # This includes both timezone-naive and timezone-aware datetimes.
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        # Alternatively, 'datetime64' often works as a shorthand for both as well:
        # date_cols = df.select_dtypes(include='datetime64').columns.tolist()
        # Potential date columns (text columns identified by exploration)
        potential_date_cols_from_explore = []
        if st.session_state.exploration_results:
            for finding in st.session_state.exploration_results:
                if finding['issue_type'] == 'Potential Dates':
                    # Extract column names (remove quotes)
                    potential_date_cols_from_explore.extend([re.sub(r"['`]", "", d) for d in finding.get('details', [])])


        operation_selection = st.selectbox("Choose Cleaning Operation:", [
            "--- Select Operation ---",
            "1. Clean Missing Data (NaN)",
            "2. Handle Outliers",
            "3. Smooth Numeric Data (Rolling Avg)",
            "4. Normalize/Scale Numeric Data",
            "5. Deduplicate Records",
            "6. Standardize Date Column",
            "7. Standardize Case & Whitespace",
            "8. Convert Data Type",
            "9. Regex Find/Replace",
            "10. Fuzzy Match Text Grouping", # Combined Analysis + Apply
            "11. Apply Numeric Constraint",
            "12. Sort Data",
            "13. Rename Column",
            "14. Remove Column(s)",
        ], key="operation_choice", index=0) # Reset to placeholder


        # Function to apply changes and log
        def apply_changes(operation_name, affected_cols=[], message="", df_modified=None, success_msg="Operation applied successfully."):
            original_shape = st.session_state.df.shape
            if df_modified is not None:
                 # Reset index after potential row drops/sorts for consistency
                 st.session_state.df = df_modified.reset_index(drop=True)

            final_shape = st.session_state.df.shape
            rows_changed = original_shape[0] - final_shape[0]
            cols_changed = original_shape[1] - final_shape[1]

            change_msg = ""
            if rows_changed > 0: change_msg += f" {rows_changed:,} rows removed."
            elif rows_changed < 0: change_msg += f" {-rows_changed:,} rows added." # Unlikely here
            if cols_changed > 0: change_msg += f" {cols_changed} columns removed."
            elif cols_changed < 0: change_msg += f" {-cols_changed} columns added." # Unlikely here

            # Log entry
            log_entry_base = f"{time.strftime('%H:%M:%S')} - {operation_name}"
            log_message = message if message else f"{success_msg}{change_msg}"
            affected_str = f" (Cols: {', '.join(affected_cols)})" if affected_cols else (" (Row operation)" if rows_changed != 0 else "")
            st.session_state.operation_log.insert(0, f"{log_entry_base}{affected_str}: {log_message}") # Insert at top

            # Clear analysis results as they might be outdated
            st.session_state.exploration_results = None
            st.session_state.profile_results = None
            # Don't clear fuzzy results if the operation is fuzzy related, handle within fuzzy logic
            if 'fuzzy' not in operation_name.lower():
                st.session_state.fuzzy_results = None

            st.toast(f"{success_msg}{change_msg}", icon="👍")
            st.session_state[OPERATION_APPLIED_KEY] = True # Flag that we need to rerun

        # --- Operation Forms ---
        form_placeholder = st.empty() # Placeholder for operation forms

        # Use the placeholder to dynamically create the form based on selection
        with form_placeholder.container():
            try: # Wrap operations in try-except for robustness

                # --- 1. Missing Data ---
                if operation_selection == "1. Clean Missing Data (NaN)":
                    with st.form("missing_form"):
                        st.markdown("**Handle Missing Values (NaN)**")
                        missing_cols_select = st.multiselect("Select Columns (or leave blank for all):", all_columns, key="missing_cols")
                        target_cols = missing_cols_select if missing_cols_select else all_columns
                        missing_method = st.radio("Method:", ('Drop Rows with NaNs', 'Drop Columns with any NaNs', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with Value'), key="missing_method")
                        fill_value_input = None
                        if missing_method == 'Fill with Value':
                            fill_value_input = st.text_input("Value to fill with:", key="missing_fill_val", value="0") # Default to 0

                        submitted = st.form_submit_button("Apply Missing Data Handling", type="primary")
                        if submitted:
                            df_copy = df.copy()
                            original_shape = df_copy.shape
                            cols_affected = []
                            operation_msg = ""

                            if not target_cols:
                                st.warning("No columns selected or available to process.")
                                st.stop()

                            if missing_method == 'Drop Rows with NaNs':
                                subset_param = target_cols if missing_cols_select else None # Drop row if NaN in ANY column only if specific cols aren't selected
                                df_copy.dropna(subset=subset_param, inplace=True)
                                operation_msg = f"Dropped rows with NaNs"
                                if subset_param: operation_msg += f" in columns: {', '.join(subset_param)}"
                            elif missing_method == 'Drop Columns with any NaNs':
                                cols_to_drop = [col for col in target_cols if df_copy[col].isnull().any()]
                                if cols_to_drop:
                                    df_copy.drop(columns=cols_to_drop, inplace=True)
                                    cols_affected = cols_to_drop
                                    operation_msg = f"Dropped columns with NaNs: {', '.join(cols_to_drop)}"
                                else:
                                    st.info("No columns found with missing values among selected/all columns to drop.")
                                    st.stop() # Prevent applying no changes
                            else: # Fill methods
                                applied_fill = False
                                fill_details = []
                                for col in target_cols:
                                    if df_copy[col].isnull().any(): # Only process columns with actual NaNs
                                        original_dtype = df_copy[col].dtype
                                        fill_val = None
                                        fill_desc = ""
                                        try:
                                            if missing_method == 'Fill with Mean':
                                                if is_numeric_dtype(df_copy[col]):
                                                    # Use .astype(float) for mean calculation with nullable integers
                                                    mean_val = df_copy[col].astype(float).mean()
                                                    fill_val = mean_val
                                                    fill_desc = f"mean ({mean_val:.2f})"
                                                else:
                                                    st.warning(f"Column '{col}' is not numeric. Cannot fill with mean.")
                                                    continue
                                            elif missing_method == 'Fill with Median':
                                                if is_numeric_dtype(df_copy[col]):
                                                     # Use .astype(float) for median calculation with nullable integers
                                                    median_val = df_copy[col].astype(float).median()
                                                    fill_val = median_val
                                                    fill_desc = f"median ({median_val:.2f})"
                                                else:
                                                    st.warning(f"Column '{col}' is not numeric. Cannot fill with median.")
                                                    continue
                                            elif missing_method == 'Fill with Mode':
                                                mode_val = df_copy[col].mode()
                                                if not mode_val.empty:
                                                    fill_val = mode_val[0]
                                                    fill_desc = f"mode ({fill_val})"
                                                else:
                                                    st.warning(f"Could not determine mode for column '{col}'. Skipping.")
                                                    continue
                                            elif missing_method == 'Fill with Value':
                                                if fill_value_input is None: # Should not happen with default
                                                    st.error("Please provide a fill value.")
                                                    st.stop()
                                                fill_val = fill_value_input
                                                fill_desc = f"value '{fill_val}'"

                                            # Try to convert fill_val to the column's dtype if possible
                                            typed_fill_val = fill_val
                                            try:
                                                if pd.notna(fill_val):
                                                     # Special handling for boolean targets
                                                     if is_bool_dtype(original_dtype):
                                                          bool_map = {'true': True, 'false': False, '1': True, '0': False}
                                                          typed_fill_val = bool_map.get(str(fill_val).lower(), pd.NA) # Use NA if not recognized
                                                     elif is_datetime64_any_dtype(original_dtype):
                                                         typed_fill_val = pd.to_datetime(fill_val, errors='coerce') # Coerce to NaT if invalid
                                                     else:
                                                         # Attempt direct conversion otherwise
                                                         typed_fill_val = pd.Series([fill_val]).astype(original_dtype)[0]
                                            except (ValueError, TypeError, OverflowError) as e:
                                                st.warning(f"Could not convert fill value '{fill_val}' to original type {original_dtype} for column '{col}'. Filling as object/string. Error: {e}")
                                                typed_fill_val = str(fill_val) # Fallback

                                            if pd.isna(typed_fill_val) and missing_method == 'Fill with Value' and not is_datetime64_any_dtype(original_dtype): # Avoid warning for intentional NaT fill
                                                 st.warning(f"Fill value '{fill_val}' resulted in NaN for column '{col}'. Ensure it's compatible with type {original_dtype}.")


                                            df_copy[col].fillna(typed_fill_val, inplace=True)
                                            cols_affected.append(col)
                                            fill_details.append(f"'{col}' with {fill_desc}")
                                            applied_fill = True

                                        except Exception as e:
                                            st.error(f"Error filling column '{col}': {e}")
                                if not applied_fill and target_cols: # Check if any fills happened across all target columns
                                    st.info("No missing values found in selected/all columns to fill.")
                                    st.stop()
                                elif applied_fill:
                                    operation_msg = f"Filled NaNs in {len(fill_details)} column(s): {'; '.join(fill_details)}"

                            # Apply changes if any were made
                            if not df_copy.equals(df):
                                 apply_changes("Missing Data Handling", affected_cols=list(set(cols_affected)), message=operation_msg, df_modified=df_copy)
                            # else case handled by st.info/st.stop above


                # --- 2. Handle Outliers ---
                elif operation_selection == "2. Handle Outliers":
                    if not numeric_cols:
                        st.warning("No numeric columns available for outlier detection.")
                    else:
                        with st.form("outlier_form"):
                            st.markdown("**Handle Outliers in Numeric Columns**")
                            outlier_col = st.selectbox("Select Numeric Column:", numeric_cols, key="outlier_col")
                            outlier_method = st.radio("Method:", ('IQR', 'Z-Score'), key="outlier_method")
                            threshold = 1.5
                            if outlier_method == 'IQR':
                                threshold = st.number_input("IQR Multiplier Threshold:", min_value=0.1, value=1.5, step=0.1, key="outlier_thresh_iqr", help="Standard is 1.5. Higher values are less sensitive.")
                            else: # Z-Score
                                threshold = st.number_input("Z-Score Threshold:", min_value=0.1, value=3.0, step=0.1, key="outlier_thresh_z", help="Standard is 3. Higher values are less sensitive.")
                            action = st.radio("Action on Outliers:", ('Remove Rows', 'Cap Values (Clip)'), key="outlier_action")

                            submitted = st.form_submit_button("Apply Outlier Handling", type="primary")
                            if submitted:
                                df_copy = df.copy()
                                col = outlier_col
                                num_outliers = 0
                                operation_msg = ""

                                if col not in df_copy.columns:
                                    st.error(f"Column '{col}' not found.")
                                    st.stop()
                                if not is_numeric_dtype(df_copy[col]):
                                    st.error(f"Column '{col}' is not numeric.")
                                    st.stop()

                                # Work with float version for calculations, handle potential NaNs
                                numeric_col_data = df_copy[col].astype(float).dropna()
                                if numeric_col_data.empty or numeric_col_data.nunique() < 2:
                                    st.info(f"Not enough valid numeric data in '{col}' for outlier detection.")
                                    st.stop()

                                lower_bound, upper_bound = None, None
                                outlier_indices = pd.Index([]) # Store indices of outliers

                                if outlier_method == 'IQR':
                                    Q1 = numeric_col_data.quantile(0.25)
                                    Q3 = numeric_col_data.quantile(0.75)
                                    IQR = Q3 - Q1
                                    if IQR > 0:
                                        lower_bound = Q1 - threshold * IQR
                                        upper_bound = Q3 + threshold * IQR
                                        # Find indices in the original dataframe (using float version for comparison)
                                        outlier_indices = df_copy.index[(df_copy[col].astype(float) < lower_bound) | (df_copy[col].astype(float) > upper_bound)]
                                    else:
                                        st.warning(f"IQR is zero for column '{col}', cannot detect outliers with this method.")
                                        st.stop()

                                elif outlier_method == 'Z-Score':
                                    mean_val = numeric_col_data.mean()
                                    std_val = numeric_col_data.std()
                                    if std_val > 0:
                                        # Calculate z-scores on the float version of the column
                                        z_scores = np.abs((df_copy[col].astype(float) - mean_val) / std_val)
                                        # Find indices where z_score > threshold (works even if original had NaNs)
                                        outlier_indices = df_copy.index[z_scores > threshold]
                                        # Define bounds for capping
                                        lower_bound = mean_val - threshold * std_val
                                        upper_bound = mean_val + threshold * std_val
                                    else:
                                        st.warning(f"Standard deviation is zero for column '{col}', cannot detect outliers or cap with Z-Score method.")
                                        st.stop()

                                num_outliers = len(outlier_indices)

                                if num_outliers == 0:
                                    st.info(f"No outliers detected in column '{col}' using method '{outlier_method}' with threshold {threshold}.")
                                    st.stop()

                                if action == 'Remove Rows':
                                    df_copy.drop(outlier_indices, inplace=True)
                                    operation_msg = f"Removed {num_outliers} outlier rows from '{col}' (Method: {outlier_method}, Threshold: {threshold})"
                                    apply_changes("Outlier Handling (Remove)", affected_cols=[], message=operation_msg, df_modified=df_copy)
                                elif action == 'Cap Values (Clip)':
                                    if lower_bound is not None and upper_bound is not None:
                                        original_dtype = df_copy[col].dtype
                                        # Clip the column - this might change dtype (e.g., Int64 to float)
                                        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                                        # Try to restore original dtype if possible and appropriate
                                        if is_integer_dtype(original_dtype) and not is_float_dtype(df_copy[col].dtype):
                                             try:
                                                 # If clipping resulted in floats that are still whole numbers, convert back
                                                 if (df_copy[col].dropna() % 1 == 0).all():
                                                     df_copy[col] = df_copy[col].astype(original_dtype)
                                             except Exception:
                                                 pass # Keep the potentially changed type if conversion fails
                                        operation_msg = f"Capped {num_outliers} outlier values in '{col}' (Method: {outlier_method}, Threshold: {threshold})"
                                        apply_changes("Outlier Handling (Cap)", affected_cols=[col], message=operation_msg, df_modified=df_copy)
                                    else:
                                         st.error(f"Could not calculate bounds to cap outliers for column '{col}'.")
                                         st.stop()

                # --- 3. Smooth Numeric Data ---
                elif operation_selection == "3. Smooth Numeric Data (Rolling Avg)":
                     if not numeric_cols:
                         st.warning("No numeric columns available for smoothing.")
                     else:
                        with st.form("smooth_form"):
                            st.markdown("**Smooth Numeric Data using Rolling Average**")
                            smooth_col = st.selectbox("Select Numeric Column:", numeric_cols, key="smooth_col")
                            window = st.number_input("Window Size:", min_value=2, value=3, step=1, key="smooth_window", help="Number of observations to include in the moving average window.")
                            center_smooth = st.checkbox("Center window", value=True, key="smooth_center", help="Align window label with the center of the window (requires odd window size for perfect alignment).")

                            submitted = st.form_submit_button("Apply Smoothing", type="primary")
                            if submitted:
                                 df_copy = df.copy()
                                 col = smooth_col
                                 if col not in df_copy.columns: st.error(f"Column '{col}' not found."); st.stop()
                                 if not is_numeric_dtype(df_copy[col]): st.error(f"Column '{col}' is not numeric."); st.stop()

                                 try:
                                     # Convert to float before rolling for consistent calculation
                                     df_copy[col] = df_copy[col].astype(float).rolling(window=window, min_periods=1, center=center_smooth).mean()
                                     operation_msg = f"Applied rolling average (window={window}, center={center_smooth}) to '{col}'"
                                     apply_changes("Smoothing", affected_cols=[col], message=operation_msg, df_modified=df_copy)
                                 except Exception as e:
                                     st.error(f"Error applying smoothing to '{col}': {e}")


                # --- 4. Normalize/Scale Numeric Data ---
                elif operation_selection == "4. Normalize/Scale Numeric Data":
                     if not numeric_cols:
                         st.warning("No numeric columns available for normalization.")
                     else:
                        with st.form("normalize_form"):
                            st.markdown("**Normalize or Scale Numeric Data**")
                            norm_cols_select = st.multiselect("Select Numeric Columns (or leave blank for all numeric):", numeric_cols, key="norm_cols")
                            target_cols = norm_cols_select if norm_cols_select else numeric_cols
                            norm_method = st.radio("Method:", ('Min-Max Scaling (to 0-1)', 'Z-Score Standardization (zero mean, unit variance)'), key="norm_method")

                            submitted = st.form_submit_button("Apply Normalization/Scaling", type="primary")
                            if submitted:
                                if not target_cols:
                                    st.warning("Please select at least one numeric column.")
                                    st.stop()

                                df_copy = df.copy()
                                applied_norm = False
                                cols_affected = []
                                norm_details = []

                                for col in target_cols:
                                    # Convert to float, dropna for scaler
                                    data_to_scale = df_copy[[col]].astype(float).dropna()
                                    if data_to_scale.empty or data_to_scale.nunique()[0] < 2 :
                                        st.info(f"Skipping normalization for column '{col}': not enough unique numeric data or all NaN.")
                                        continue

                                    try:
                                        if norm_method == 'Min-Max Scaling (to 0-1)':
                                            scaler = MinMaxScaler()
                                            method_name = "Min-Max"
                                        else: # Z-Score
                                            scaler = StandardScaler()
                                            method_name = "Z-Score"

                                        scaled_data = scaler.fit_transform(data_to_scale)
                                        # Put scaled data back into the DataFrame, preserving original index
                                        df_copy.loc[data_to_scale.index, col] = scaled_data.flatten()
                                        cols_affected.append(col)
                                        norm_details.append(f"'{col}' ({method_name})")
                                        applied_norm = True
                                    except Exception as e:
                                        st.error(f"Error normalizing column '{col}': {e}")

                                if not applied_norm:
                                    st.info("Normalization not applied to any selected columns (check data type or variance).")
                                    st.stop()
                                else:
                                    operation_msg = f"Applied {method_name} scaling to {len(norm_details)} column(s)."
                                    apply_changes("Normalization/Scaling", affected_cols=cols_affected, message=operation_msg, df_modified=df_copy)

                # --- 5. Deduplicate Records ---
                elif operation_selection == "5. Deduplicate Records":
                    with st.form("dedup_form"):
                        st.markdown("**Remove Duplicate Rows**")
                        dedup_cols_select = st.multiselect("Check for duplicates based on Columns (leave blank for all):", all_columns, key="dedup_cols")
                        keep_option = st.radio("Which duplicate to keep:", ('First', 'Last', 'Remove All Duplicates'), key="dedup_keep", index=0)

                        submitted = st.form_submit_button("Apply Deduplication", type="primary")
                        if submitted:
                            df_copy = df.copy()
                            original_count = len(df_copy)
                            subset_param = dedup_cols_select if dedup_cols_select else None

                            keep_param = False # Default for 'Remove All Duplicates'
                            if keep_option == 'First': keep_param = 'first'
                            elif keep_option == 'Last': keep_param = 'last'

                            df_copy.drop_duplicates(subset=subset_param, keep=keep_param, inplace=True)
                            final_count = len(df_copy)
                            rows_removed = original_count - final_count

                            if rows_removed > 0:
                                subset_msg = f" based on columns: {', '.join(subset_param)}" if subset_param else " based on all columns"
                                operation_msg = f"Removed {rows_removed:,} duplicate rows ({keep_option} kept){subset_msg}."
                                apply_changes("Deduplication", affected_cols=[], message=operation_msg, df_modified=df_copy)
                            else:
                                st.info("No duplicate rows found based on the selected criteria.")
                                st.stop()

                # --- 6. Standardize Date Column ---
                elif operation_selection == "6. Standardize Date Column":
                     # Combine known date cols and potential ones for selection
                     selectable_date_cols = sorted(list(set(all_columns) - set(numeric_cols))) # Start with non-numeric
                     # prioritize known date cols
                     default_selection = [c for c in date_cols if c in selectable_date_cols]
                     # Add potential date cols identified by explore, if any
                     if potential_date_cols_from_explore:
                          for p_col in potential_date_cols_from_explore:
                              if p_col in selectable_date_cols and p_col not in default_selection:
                                  default_selection.append(p_col)


                     if not selectable_date_cols:
                          st.warning("No suitable text or date columns available for date standardization.")
                     else:
                        with st.form("date_form"):
                            st.markdown("**Standardize Date Column(s)**")
                            date_cols_select = st.multiselect("Select Columns to convert/standardize:",
                                                               options=selectable_date_cols,
                                                               default=default_selection, # Pre-select likely candidates
                                                               key="date_cols")
                            output_format = st.text_input("Output Format String (optional, e.g., %Y-%m-%d):",
                                                          key="date_format",
                                                          help="Leave blank to keep as datetime objects. See Python strftime codes.")
                            errors_coerce = st.checkbox("Convert unparseable dates to NaT (missing)?", value=True, key="date_errors",
                                                       help="If unchecked, errors during parsing will stop the operation.")

                            submitted = st.form_submit_button("Apply Date Standardization", type="primary")
                            if submitted:
                                if not date_cols_select:
                                    st.warning("Please select at least one column.")
                                    st.stop()

                                df_copy = df.copy()
                                applied_date_conv = False
                                cols_affected = []
                                date_conv_details = []

                                for col in date_cols_select:
                                    if col not in df_copy.columns:
                                        st.warning(f"Column '{col}' not found. Skipping.")
                                        continue
                                    try:
                                        original_dtype = df_copy[col].dtype
                                        # Attempt conversion - infer_datetime_format can speed up common formats
                                        converted_series = pd.to_datetime(df_copy[col], errors='coerce' if errors_coerce else 'raise', infer_datetime_format=True)

                                        # Check if conversion actually happened (or if it was already datetime)
                                        if not is_datetime64_any_dtype(converted_series.dtype):
                                            if errors_coerce:
                                                 st.warning(f"Column '{col}' could not be reliably converted to datetime, even with coercion. Skipping.")
                                                 continue # Skip if coercion failed to produce datetime
                                            else: # errors='raise' would have errored before here
                                                 st.error(f"Conversion failed unexpectedly for '{col}'.")
                                                 continue


                                        df_copy[col] = converted_series
                                        format_applied = False

                                        # Apply formatting if specified
                                        if output_format and is_datetime64_any_dtype(df_copy[col]):
                                            try:
                                                is_nat = df_copy[col].isna()
                                                # Apply formatting only to non-NaT values
                                                df_copy.loc[~is_nat, col] = df_copy.loc[~is_nat, col].dt.strftime(output_format)
                                                 # After strftime, the column becomes object type. Convert NaT back to None/pd.NA
                                                df_copy.loc[is_nat, col] = pd.NA
                                                df_copy[col] = df_copy[col].astype("string") # Ensure consistent Nullable String type
                                                format_applied = True
                                            except ValueError as format_error:
                                                st.error(f"Invalid date format string '{output_format}' for column '{col}': {format_error}. Leaving as datetime object.")
                                                # Keep it as datetime object if formatting fails
                                                format_applied = False # Indicate formatting failed
                                            except Exception as fe:
                                                st.error(f"Error formatting column '{col}': {fe}. Leaving as datetime object.")
                                                format_applied = False


                                        cols_affected.append(col)
                                        detail = f"'{col}' converted to {'string (format: ' + output_format + ')' if format_applied else 'datetime'}"
                                        date_conv_details.append(detail)
                                        applied_date_conv = True

                                    except ValueError as e:
                                        # This should only happen if errors='raise' and parsing fails
                                        st.error(f"Error parsing date in column '{col}': {e}. Try enabling 'Convert unparseable dates to NaT'.")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred during date conversion for column '{col}': {e}")

                                if not applied_date_conv:
                                    st.info("Date conversion not applied to any selected columns (check errors or data).")
                                    st.stop()
                                else:
                                    operation_msg = f"Standardized {len(date_conv_details)} column(s): {'; '.join(date_conv_details)}"
                                    apply_changes("Date Standardization", affected_cols=cols_affected, message=operation_msg, df_modified=df_copy)

                # --- 7. Standardize Case & Whitespace ---
                elif operation_selection == "7. Standardize Case & Whitespace":
                    if not text_cols:
                        st.warning("No text columns available for case/whitespace standardization.")
                    else:
                        with st.form("case_form"):
                            st.markdown("**Standardize Text Case and Whitespace**")
                            case_cols_select = st.multiselect("Select Text Columns (or leave blank for all text):", text_cols, key="case_cols")
                            target_cols = case_cols_select if case_cols_select else text_cols
                            case_type = st.selectbox("Select Action:", ('Trim Whitespace', 'Convert to Lowercase', 'Convert to Uppercase', 'Convert to Title Case'), key="case_type")

                            submitted = st.form_submit_button("Apply Case/Whitespace Standardization", type="primary")
                            if submitted:
                                if not target_cols:
                                    st.warning("Please select at least one text column.")
                                    st.stop()

                                df_copy = df.copy()
                                applied_case = False
                                cols_affected = []

                                for col in target_cols:
                                    if col not in df_copy.columns:
                                        st.warning(f"Column '{col}' not found. Skipping.")
                                        continue
                                    # Check if column is text-like before applying string methods
                                    if is_string_dtype(df_copy[col]) or is_object_dtype(df_copy[col]):
                                         # Use pandas nullable string type for safe operations
                                        str_series = df_copy[col].astype("string")

                                        if case_type == 'Trim Whitespace':
                                            df_copy[col] = str_series.str.strip()
                                        elif case_type == 'Convert to Lowercase':
                                            df_copy[col] = str_series.str.lower()
                                        elif case_type == 'Convert to Uppercase':
                                            df_copy[col] = str_series.str.upper()
                                        elif case_type == 'Convert to Title Case':
                                            # Title case can be tricky (e.g., Mc'Donalds -> Mc'donalds)
                                            # Pandas default might be sufficient for most cases
                                            df_copy[col] = str_series.str.title()

                                        cols_affected.append(col)
                                        applied_case = True
                                    else:
                                         st.warning(f"Column '{col}' is not a text type ({df_copy[col].dtype}). Cannot apply {case_type}.")

                                if not applied_case:
                                    st.info("Operation not applied (check column types).")
                                    st.stop()
                                else:
                                    operation_msg = f"Applied '{case_type}' to {len(cols_affected)} column(s)."
                                    apply_changes("Case/Whitespace", affected_cols=cols_affected, message=operation_msg, df_modified=df_copy)

                # --- 8. Convert Data Type ---
                elif operation_selection == "8. Convert Data Type":
                     with st.form("convert_type_form"):
                         st.markdown("**Convert Column Data Type**")
                         conv_col = st.selectbox("Select Column to Convert:", all_columns, key="conv_col")
                         # Sensible target types based on column content could be suggested here
                         target_type = st.selectbox("Convert to Type:",
                                                     ['string (text)', 'Int64 (nullable integer)', 'float64 (decimal)', 'boolean (True/False)', 'datetime64[ns] (date/time)', 'category (optimized text)'],
                                                     key="conv_target_type")

                         submitted = st.form_submit_button("Apply Type Conversion", type="primary")
                         if submitted:
                             df_copy = df.copy()
                             col = conv_col
                             if col not in df_copy.columns: st.error(f"Column '{col}' not found."); st.stop()

                             original_nan_count = df_copy[col].isnull().sum()
                             converted_series = None
                             conversion_error = None
                             final_type = None

                             try:
                                 current_dtype = str(df_copy[col].dtype)
                                 target_type_str = target_type.split(" ")[0] # e.g., "string", "Int64"

                                 # Check if conversion is necessary
                                 if target_type_str == current_dtype or \
                                    (target_type_str == 'string' and is_string_dtype(df_copy[col])) or \
                                    (target_type_str == 'Int64' and is_integer_dtype(df_copy[col]) and df_copy[col].isnull().any()) or \
                                    (target_type_str == 'float64' and is_float_dtype(df_copy[col])) or \
                                    (target_type_str == 'boolean' and is_bool_dtype(df_copy[col])) or \
                                    (target_type_str == 'datetime64[ns]' and is_datetime64_any_dtype(df_copy[col])) or \
                                    (target_type_str == 'category' and current_dtype == 'category'):
                                     st.info(f"Column '{col}' is already compatible with type '{target_type}'. No conversion needed.")
                                     st.stop()


                                 if target_type_str == 'string':
                                     # Convert non-string types safely to nullable string
                                     converted_series = df_copy[col].astype('string')
                                     final_type = 'string'
                                 elif target_type_str == 'Int64':
                                     # Coerce to numeric first, then check if convertible to int, then use Int64
                                     numeric_temp = pd.to_numeric(df_copy[col], errors='coerce')
                                     # Check if all non-NA coerced values are integer-like
                                     is_int_like = numeric_temp.dropna().apply(lambda x: x == np.floor(x) if pd.notna(x) else True).all()
                                     if is_int_like:
                                         converted_series = numeric_temp.astype('Int64')
                                         final_type = 'Int64'
                                     else:
                                         conversion_error = f"Cannot convert to Int64: Column '{col}' contains non-integer values or values that couldn't be coerced to numeric."
                                 elif target_type_str == 'float64':
                                     # Coerce errors during numeric conversion
                                     converted_series = pd.to_numeric(df_copy[col], errors='coerce')
                                     # Ensure it's explicitly float64 even if source was int
                                     if converted_series.notna().any(): # Check if not all NaNs after coercion
                                         converted_series = converted_series.astype('float64')
                                         final_type = 'float64'
                                     elif df_copy[col].isnull().all(): # Handle case where original is all NaN
                                          converted_series = pd.Series([np.nan] * len(df_copy), index=df_copy.index, dtype='float64')
                                          final_type = 'float64'
                                     else:
                                          conversion_error = f"Column '{col}' could not be converted to numeric (float)."

                                 elif target_type_str == 'boolean':
                                    # Use pandas nullable BooleanDtype
                                    map_dict_lower = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False}
                                    map_dict_numeric = {1: True, 0: False, 1.0: True, 0.0: False}

                                    # Create a series to map, start with object type
                                    temp_series = pd.Series(index=df_copy.index, dtype='object')
                                    # Handle original boolean inputs correctly
                                    bool_mask = df_copy[col].apply(lambda x: isinstance(x, bool))
                                    temp_series[bool_mask] = df_copy[col][bool_mask]

                                    # Map numeric (that are not already boolean)
                                    numeric_mask = pd.to_numeric(df_copy[col], errors='coerce').notna() & (~bool_mask)
                                    temp_series[numeric_mask] = pd.to_numeric(df_copy[col][numeric_mask], errors='coerce').map(map_dict_numeric)

                                    # Map strings (lowercase) for the rest (not bool, not mapped numeric)
                                    string_mask = ~(bool_mask | numeric_mask)
                                    temp_series[string_mask] = df_copy[col][string_mask].astype(str).str.lower().str.strip().map(map_dict_lower)

                                    # Convert the resulting series (with True/False/NaN/None) to nullable boolean
                                    converted_series = temp_series.astype('boolean')
                                    final_type = 'boolean'


                                 elif target_type_str == 'datetime64[ns]':
                                     converted_series = pd.to_datetime(df_copy[col], errors='coerce', infer_datetime_format=True)
                                     # Verify conversion produced datetime
                                     if not is_datetime64_any_dtype(converted_series.dtype):
                                         conversion_error = f"Column '{col}' could not be reliably converted to datetime, even with coercion."
                                     else:
                                         final_type = 'datetime64[ns]'
                                 elif target_type_str == 'category':
                                      # Useful for text columns with relatively low cardinality
                                      converted_series = df_copy[col].astype('category')
                                      final_type = 'category'

                                 else: # Should not happen with selectbox
                                     conversion_error = f"Unsupported target type: {target_type}"


                                 if conversion_error:
                                     st.error(conversion_error)
                                     st.stop()
                                 elif converted_series is not None:
                                     df_copy[col] = converted_series
                                     final_nan_count = df_copy[col].isnull().sum()
                                     coerced_count = final_nan_count - original_nan_count
                                     # Adjust if original NaNs were somehow filled (unlikely but safe)
                                     coerced_count = max(0, coerced_count)

                                     op_message = f"Converted '{col}' from '{current_dtype}' to '{final_type}'."
                                     if coerced_count > 0:
                                         op_message += f" {coerced_count} value(s) became missing during conversion."
                                         st.warning(f"Conversion resulted in {coerced_count} new missing values for column '{col}'.")

                                     apply_changes("Convert Data Type", affected_cols=[col], message=op_message, df_modified=df_copy)
                                 # else case implies no conversion happened or was needed, don't apply changes

                             except Exception as e:
                                 st.error(f"Error converting column '{col}' to {target_type}: {e}")
                                 st.stop()

                # --- 9. Regex Find/Replace ---
                elif operation_selection == "9. Regex Find/Replace":
                    if not text_cols:
                        st.warning("No text columns available for Regex operations.")
                    else:
                        with st.form("regex_form"):
                            st.markdown("**Regex Find and Replace in Text Column**")
                            regex_col = st.selectbox("Select Text Column:", text_cols, key="regex_col")
                            pattern = st.text_input("Find Regex Pattern:", key="regex_pattern", value="", help="e.g., `\\d+` to find numbers, `[^a-zA-Z]` to find non-letters")
                            replacement = st.text_input("Replace With:", key="regex_replacement", value="", help="Leave empty to remove matches. Use `\\1`, `\\2` etc. for capture groups.")
                            case_sensitive = st.checkbox("Case Sensitive Match", value=True, key="regex_case")

                            submitted = st.form_submit_button("Apply Regex Replace", type="primary")
                            if submitted:
                                 df_copy = df.copy()
                                 col = regex_col
                                 if not col or col not in df_copy.columns: st.error(f"Column '{col}' not found."); st.stop()
                                 if not pattern: st.warning("Regex pattern cannot be empty."); st.stop()

                                 # Ensure column is string type for safe replacement
                                 if not is_string_dtype(df_copy[col]) and not is_object_dtype(df_copy[col]):
                                     st.warning(f"Regex replace requires a text column. '{col}' is {df_copy[col].dtype}. Converting to string first.")
                                     df_copy[col] = df_copy[col].astype('string')
                                 else:
                                     # Ensure it's the nullable string type
                                     df_copy[col] = df_copy[col].astype('string')

                                 try:
                                     # Apply regex replace
                                     df_copy[col] = df_copy[col].str.replace(pattern, replacement, regex=True, case=case_sensitive, flags=0 if case_sensitive else re.IGNORECASE)
                                     operation_msg = f"Applied Regex replace (pattern='{pattern}', repl='{replacement}', case={case_sensitive}) on '{col}'"
                                     apply_changes("Regex Replace", affected_cols=[col], message=operation_msg, df_modified=df_copy)
                                 except re.error as regex_err:
                                     st.error(f"Invalid Regular Expression: {regex_err}")
                                 except Exception as e:
                                     st.error(f"Error during Regex replace on '{col}': {e}")

                # --- 10. Fuzzy Match Text Grouping ---
                elif operation_selection == "10. Fuzzy Match Text Grouping":
                     if not text_cols:
                         st.warning("No text columns available for Fuzzy Matching.")
                     else:
                         st.markdown("**Fuzzy Match Text Grouping**")
                         st.info("This tool helps find and standardize similar text values (e.g., 'New York', 'new york', 'NY').")

                         # --- Analysis Part ---
                         with st.form("fuzzy_analyze_form"):
                             st.markdown("**Step 1: Analyze Column for Similar Values**")
                             fuzzy_col = st.selectbox("Select Text Column to Analyze:", text_cols, key="fuzzy_col_analyze")
                             threshold = st.slider("Similarity Threshold (%):", min_value=50, max_value=100, value=85, key="fuzzy_threshold",
                                                   help="Minimum similarity score (using fuzz.ratio) to group values.")

                             analyze_submitted = st.form_submit_button("Analyze for Similar Groups", type="primary")
                             if analyze_submitted:
                                 st.session_state.fuzzy_results = None # Clear previous results
                                 col = fuzzy_col
                                 if not col or col not in df.columns: st.error(f"Column '{col}' not found."); st.stop()

                                 with st.spinner(f"Analyzing '{col}' for values with similarity >= {threshold}%... This may take time for many unique values."):
                                     start_time = time.time()
                                     # Ensure column is string type and get unique non-null values
                                     try:
                                          unique_vals = df[col].dropna().astype(str).unique()
                                     except Exception as e:
                                          st.error(f"Could not get unique string values from column '{col}'. Is it convertible to string? Error: {e}")
                                          st.stop()


                                     if len(unique_vals) < 2:
                                         st.info(f"Not enough unique text values in column '{col}' to perform fuzzy matching.")
                                         st.stop()
                                     if len(unique_vals) > 5000: # Add warning for large number of uniques
                                         st.warning(f"Column '{col}' has {len(unique_vals):,} unique values. Fuzzy matching may be slow.", icon="⏳")


                                     groups = []
                                     processed_indices = set()
                                     unique_vals_list = sorted(list(unique_vals)) # Sorting helps consistency
                                     n_unique = len(unique_vals_list)

                                     # Optimized slightly: Compare each item only with subsequent items
                                     for i in range(n_unique):
                                         if i in processed_indices: continue
                                         current_group = [unique_vals_list[i]] # Start group with current item
                                         # Find matches for the *first* item in the potential group
                                         base_item = unique_vals_list[i]
                                         matches_found_for_base = False
                                         for j in range(i + 1, n_unique):
                                             if j in processed_indices: continue
                                             ratio = fuzz.ratio(base_item, unique_vals_list[j])
                                             if ratio >= threshold:
                                                 current_group.append(unique_vals_list[j])
                                                 processed_indices.add(j)
                                                 matches_found_for_base = True

                                         # Only add the group if the base item actually found matches
                                         if matches_found_for_base:
                                             processed_indices.add(i) # Mark base item as processed only if it led a group
                                             groups.append(sorted(current_group)) # Sort within group for display

                                     duration = time.time() - start_time
                                     if groups:
                                         st.session_state.fuzzy_results = {'column': col, 'threshold': threshold, 'groups': groups}
                                         st.success(f"Analysis complete in {duration:.2f}s. Found {len(groups):,} groups of similar values. Proceed to Step 2 to apply changes.")
                                         # Rerun to display Step 2
                                         st.rerun()
                                     else:
                                         st.info(f"Analysis complete in {duration:.2f}s. No groups found meeting the {threshold}% similarity threshold in '{col}'.")
                                     # No rerun needed here if no groups found


                         # --- Display Results & Apply Part ---
                         if st.session_state.fuzzy_results:
                             results = st.session_state.fuzzy_results
                             st.divider()
                             st.markdown(f"**Step 2: Review Groups Found in `{results['column']}` (Threshold: {results['threshold']}%) and Choose Replacements**")
                             st.markdown("For each group, select the value you want to keep (the 'canonical' value). All other values in that group will be replaced.")

                             with st.form("fuzzy_apply_form"):
                                 canonical_choices = {} # Store user's choice for each group
                                 num_groups_to_show = 25 # Limit display for performance
                                 total_groups = len(results['groups'])

                                 st.info(f"Showing {min(num_groups_to_show, total_groups):,} of {total_groups:,} groups found.")

                                 # Use columns for better layout of radio buttons
                                 num_cols = 2
                                 group_cols = st.columns(num_cols)
                                 col_idx = 0

                                 for i, group in enumerate(results['groups'][:num_groups_to_show]):
                                     current_col = group_cols[col_idx % num_cols]
                                     with current_col:
                                         # Default choice is often the first item (or maybe the most frequent if pre-calculated)
                                         st.markdown(f"**Group {i+1}:**")
                                         # Default to the first item in the sorted group
                                         choice = st.radio(f"Canonical:", options=group, index=0, key=f"fuzzy_group_{i}", label_visibility="collapsed")
                                         canonical_choices[i] = choice
                                         # Display the items being grouped compactly
                                         st.caption(f"`{' | '.join(group)}`")
                                         st.markdown("---") # Separator between groups in a column
                                     col_idx += 1


                                 if total_groups > num_groups_to_show:
                                     st.warning(f"Only showing the first {num_groups_to_show} groups. To process all, consider adjusting the code or threshold.")

                                 apply_submitted = st.form_submit_button("Apply Selected Replacements", type="primary")
                                 if apply_submitted:
                                     df_copy = df.copy()
                                     col = results['column']
                                     groups_to_process = results['groups'][:num_groups_to_show] # Only process displayed groups

                                     if col not in df_copy.columns:
                                         st.error(f"Column '{col}' from fuzzy analysis not found in current data.")
                                         st.session_state.fuzzy_results = None # Clear bad results
                                         st.stop()

                                     replacement_map = {}
                                     applied_count = 0
                                     try:
                                         for i, group in enumerate(groups_to_process):
                                             canonical_value = canonical_choices.get(i) # Get user choice
                                             if canonical_value is not None: # Should always exist from radio
                                                 for original_value in group:
                                                     # Map original values (that are NOT the chosen one) to the canonical one
                                                     if original_value != canonical_value:
                                                         replacement_map[original_value] = canonical_value

                                         if replacement_map:
                                             # Ensure column is string before mapping
                                             df_copy[col] = df_copy[col].astype('string')
                                             # Use replace for direct mapping (more robust than map for this case)
                                             df_copy[col] = df_copy[col].replace(replacement_map)
                                             applied_count = len(replacement_map)
                                             operation_msg = f"Applied {applied_count} fuzzy match replacements in '{col}'."
                                             # Clear fuzzy results AFTER applying successfully
                                             st.session_state.fuzzy_results = None
                                             apply_changes("Fuzzy Match Apply", affected_cols=[col], message=operation_msg, df_modified=df_copy)
                                         else:
                                             st.info("No changes selected or applied from fuzzy matching results.")
                                             # Clear fuzzy results even if no changes applied
                                             st.session_state.fuzzy_results = None
                                             st.rerun()


                                     except Exception as e:
                                         st.error(f"Error applying fuzzy match changes: {e}")
                                         # Don't clear results on error, user might want to retry
                                         st.stop()


                # --- 11. Apply Numeric Constraint ---
                elif operation_selection == "11. Apply Numeric Constraint":
                     if not numeric_cols:
                         st.warning("No numeric columns available to apply constraints.")
                     else:
                        with st.form("constraint_form"):
                            st.markdown("**Filter Rows by Numeric Range**")
                            constraint_col = st.selectbox("Select Numeric Column:", numeric_cols, key="constraint_col")
                            c1, c2 = st.columns(2)
                            min_val_str = c1.text_input("Minimum Value (optional):", key="constraint_min")
                            max_val_str = c2.text_input("Maximum Value (optional):", key="constraint_max")

                            submitted = st.form_submit_button("Apply Constraint", type="primary")
                            if submitted:
                                 df_copy = df.copy()
                                 col = constraint_col
                                 if col not in df_copy.columns: st.error(f"Column '{col}' not found."); st.stop()
                                 if not is_numeric_dtype(df_copy[col]): st.error(f"Column '{col}' is not numeric."); st.stop()
                                 if not min_val_str and not max_val_str: st.warning("Please provide at least a minimum or maximum value."); st.stop()

                                 original_count = len(df_copy)
                                 condition = pd.Series(True, index=df_copy.index) # Start with all true

                                 try:
                                     min_val, max_val = None, None
                                     op_details = []
                                     # Convert numeric column to float for comparison to handle Int64 etc.
                                     numeric_series = pd.to_numeric(df_copy[col], errors='coerce')

                                     if min_val_str:
                                         min_val = float(min_val_str)
                                         # Handle NaNs correctly: condition should be False if value is NaN
                                         condition &= (numeric_series >= min_val) # NaNs automatically become False here
                                         op_details.append(f"min >= {min_val}")
                                     if max_val_str:
                                         max_val = float(max_val_str)
                                         condition &= (numeric_series <= max_val) # NaNs automatically become False here
                                         op_details.append(f"max <= {max_val}")

                                     df_copy = df_copy[condition] # Keep only rows matching the condition
                                     final_count = len(df_copy)
                                     rows_removed = original_count - final_count

                                     if rows_removed >= 0: # Should always be true
                                         operation_msg = f"Filtered rows in '{col}' ({', '.join(op_details)}). {rows_removed:,} rows removed."
                                         apply_changes("Numeric Constraint", affected_cols=[], message=operation_msg, df_modified=df_copy)
                                     # No need for 'else' as filtering can result in 0 rows removed

                                 except ValueError:
                                      st.error("Invalid number entered for min/max value. Please enter numeric values only.")
                                 except Exception as e:
                                      st.error(f"Error applying constraint: {e}")


                # --- 12. Sort Data ---
                elif operation_selection == "12. Sort Data":
                     with st.form("sort_form"):
                         st.markdown("**Sort DataFrame Rows**")
                         sort_cols_select = st.multiselect("Select Columns to Sort By (order matters):", all_columns, key="sort_cols")
                         # Allow specifying ascending/descending per column? Simpler: one choice for all selected
                         ascending = st.radio("Sort Order:", ('Ascending', 'Descending'), key="sort_order") == 'Ascending'

                         submitted = st.form_submit_button("Apply Sort", type="primary")
                         if submitted:
                             if not sort_cols_select:
                                 st.warning("Please select at least one column to sort by.")
                                 st.stop()

                             df_copy = df.copy()
                             valid_cols = [c for c in sort_cols_select if c in df_copy.columns]
                             invalid_cols = [c for c in sort_cols_select if c not in df_copy.columns]

                             if invalid_cols:
                                 st.warning(f"Columns not found for sorting: {', '.join(invalid_cols)}. Ignoring them.")
                             if not valid_cols:
                                 st.error("No valid columns selected for sorting.")
                                 st.stop()

                             try:
                                 # ignore_index=True resets index after sorting
                                 # na_position='last' keeps NaNs at the end regardless of sort order
                                 df_copy.sort_values(by=valid_cols, ascending=ascending, inplace=True, ignore_index=True, na_position='last')
                                 order_str = "Ascending" if ascending else "Descending"
                                 operation_msg = f"Sorted data by: {', '.join(valid_cols)} ({order_str})."
                                 apply_changes("Sort Data", affected_cols=valid_cols, message=operation_msg, df_modified=df_copy)
                             except Exception as e:
                                 st.error(f"Error sorting data: {e}")


                # --- 13. Rename Column ---
                elif operation_selection == "13. Rename Column":
                     with st.form("rename_form"):
                         st.markdown("**Rename a Column**")
                         old_name = st.selectbox("Select Column to Rename:", all_columns, key="rename_old")
                         new_name = st.text_input("Enter New Column Name:", key="rename_new").strip()

                         submitted = st.form_submit_button("Apply Rename", type="primary")
                         if submitted:
                              df_copy = df.copy()
                              if not old_name or old_name not in df_copy.columns:
                                  st.error(f"Original column '{old_name}' not found or not selected.")
                                  st.stop()
                              if not new_name:
                                  st.error("New column name cannot be empty.")
                                  st.stop()
                              if new_name in df_copy.columns and new_name != old_name:
                                  st.error(f"Column name '{new_name}' already exists. Choose a different name.")
                                  st.stop()
                              if new_name == old_name:
                                   st.info("New name is the same as the old name. No change applied.")
                                   st.stop()

                              try:
                                 df_copy.rename(columns={old_name: new_name}, inplace=True)
                                 operation_msg = f"Renamed column '{old_name}' to '{new_name}'."
                                 # Update column lists used by other operations? Handled by rerun.
                                 apply_changes("Rename Column", affected_cols=[old_name, new_name], message=operation_msg, df_modified=df_copy)
                              except Exception as e:
                                   st.error(f"Error renaming column: {e}")


                # --- 14. Remove Column(s) ---
                elif operation_selection == "14. Remove Column(s)":
                     with st.form("remove_form"):
                         st.markdown("**Remove Columns**")
                         remove_cols_select = st.multiselect("Select Columns to Remove:", all_columns, key="remove_cols")

                         submitted = st.form_submit_button("Apply Remove Columns", type="primary")
                         if submitted:
                             if not remove_cols_select:
                                 st.warning("Please select at least one column to remove.")
                                 st.stop()

                             df_copy = df.copy()
                             cols_exist = [c for c in remove_cols_select if c in df_copy.columns]
                             cols_not_exist = [c for c in remove_cols_select if c not in df_copy.columns]

                             if cols_not_exist:
                                 st.warning(f"Columns not found and could not be removed: {', '.join(cols_not_exist)}")

                             if not cols_exist:
                                 st.error("None of the selected columns exist in the current data.")
                                 st.stop()

                             try:
                                 df_copy.drop(columns=cols_exist, inplace=True)
                                 operation_msg = f"Removed columns: {', '.join(cols_exist)}"
                                 apply_changes("Remove Columns", affected_cols=cols_exist, message=operation_msg, df_modified=df_copy)
                             except Exception as e:
                                 st.error(f"Error removing columns: {e}")

            except Exception as general_error:
                 # Catch unexpected errors during operation setup or execution
                 st.error(f"An unexpected error occurred in the '{operation_selection}' operation: {general_error}")


    # --- Tab 3: Download ---
    with tab3:
        st.subheader("Download Cleaned Data")
        st.markdown("Download the current state of your data.")

        final_df = st.session_state.df
        original_base, _ = os.path.splitext(st.session_state.original_filename)
        # Sanitize filename - allow alphanumeric, underscore, hyphen
        safe_base = re.sub(r'[^\w\-]+', '_', original_base).strip('_')
        if not safe_base: safe_base = "cleaned_data" # Fallback


        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.markdown("**Download as CSV**")
            try:
                csv_data = convert_df_to_csv(final_df)
                st.download_button(
                    label="Download CSV File",
                    data=csv_data,
                    file_name=f"{safe_base}_cleaned.csv",
                    mime='text/csv',
                    key='download_csv'
                )
            except Exception as e:
                st.error(f"Error preparing CSV for download: {e}")

        with dl_cols[1]:
            st.markdown("**Download as Excel (XLSX)**")
            try:
                excel_data = convert_df_to_excel(final_df)
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name=f"{safe_base}_cleaned.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel'
                )
            except Exception as e:
                st.error(f"Error preparing Excel for download: {e}")

        st.divider()
        st.markdown("**Operation Summary:**")
        st.metric("Total Operations Applied", len(st.session_state.operation_log))
        if st.session_state.operation_log:
             st.dataframe(pd.DataFrame(st.session_state.operation_log, columns=["Log Entry"]), height=300, use_container_width=True)


# --- Footer or Initial Message ---
else:
    st.info("⬆️ Upload a CSV or XLSX file using the sidebar to get started.")
    st.markdown("""
    **Features:**
    - Load CSV/XLSX files with auto-encoding detection.
    - Explore data for potential issues (missing values, outliers, duplicates, etc.).
    - Profile columns with detailed statistics.
    - Apply various cleaning operations:
        - Handle missing data (drop/fill).
        - Detect and handle outliers (IQR/Z-Score, remove/cap).
        - Normalize/Scale numeric data.
        - Remove duplicates.
        - Standardize dates and text (case, whitespace).
        - Convert data types.
        - Perform Regex find/replace.
        - Group and standardize similar text using Fuzzy Matching.
        - Filter, sort, rename, and remove columns.
    - Download the cleaned data as CSV or Excel.
    - Track applied operations in the log.
    """)


# --- Rerun Logic ---
# If an operation was applied, rerun the script to update the UI
# Use .get() for safe access in case the key somehow disappears
if st.session_state.get(OPERATION_APPLIED_KEY, False):
    st.session_state[OPERATION_APPLIED_KEY] = False # Reset the flag
    st.rerun()
