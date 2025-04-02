import os
import uuid
import pandas as pd
import numpy as np
from flask import (Flask, request, render_template, redirect, url_for,
                   send_from_directory, flash, session)
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import chardet # For encoding detection

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
CLEANED_FOLDER = 'cleaned_data'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB limit

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLEANED_FOLDER'] = CLEANED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.secret_key = os.urandom(24) # Replace with a strong secret key in production

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_encoding(file_path):
    """Detects file encoding using chardet."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(50000) # Read first 50k bytes for detection
            result = chardet.detect(raw_data)
            return result['encoding'] if result['encoding'] else 'utf-8'
    except Exception:
        return 'utf-8' # Default to utf-8 on error

def load_data(file_path, encoding=None):
    """Loads data from CSV file into a pandas DataFrame."""
    try:
        if not encoding:
            encoding = detect_encoding(file_path)
            flash(f"Auto-detected encoding: {encoding}", "info")

        df = pd.read_csv(file_path, encoding=encoding)
        # Attempt basic type inference improvement
        df = df.infer_objects()
        return df, encoding
    except UnicodeDecodeError:
        flash(f"Error decoding file with {encoding}. Try specifying a different encoding (e.g., 'latin-1', 'cp1252').", "danger")
        return None, encoding
    except Exception as e:
        flash(f"Error loading CSV file: {str(e)}", "danger")
        return None, encoding

def save_data(df, original_filename):
    """Saves the DataFrame to a new CSV file in the cleaned_data folder."""
    if not os.path.exists(app.config['CLEANED_FOLDER']):
        os.makedirs(app.config['CLEANED_FOLDER'])

    new_filename = f"{uuid.uuid4().hex}_{secure_filename(original_filename)}"
    output_path = os.path.join(app.config['CLEANED_FOLDER'], new_filename)
    try:
        df.to_csv(output_path, index=False, encoding='utf-8') # Save cleaned as UTF-8
        return output_path
    except Exception as e:
        flash(f"Error saving cleaned file: {str(e)}", "danger")
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the initial upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, saves it, loads data, and redirects to cleaning page."""
    if 'file' not in request.files:
        flash('No file part', 'warning')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Use UUID to avoid filename conflicts during upload
        upload_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        try:
            file.save(upload_path)
        except Exception as e:
             flash(f"Error saving uploaded file: {str(e)}", "danger")
             return redirect(url_for('index'))

        # Get user-specified encoding or try to detect
        user_encoding = request.form.get('encoding') or None

        df, detected_encoding = load_data(upload_path, encoding=user_encoding)

        if df is None:
            # Cleanup failed upload attempt
            if os.path.exists(upload_path):
                os.remove(upload_path)
            # Flash message already set in load_data
            return redirect(url_for('index'))

        # Save the initial state as the first "cleaned" file
        cleaned_path = save_data(df, filename)
        if cleaned_path is None:
             # Cleanup upload if saving initial state fails
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return redirect(url_for('index')) # Flash message set in save_data

        # Store info in session
        session['current_file_path'] = cleaned_path # Start with the first saved version
        session['original_filename'] = filename
        session['encoding'] = detected_encoding if user_encoding is None else user_encoding
        session['upload_path'] = upload_path # Keep original upload path for potential re-reads? Might not be needed.

        # Cleanup the initial upload file now that we have a copy in cleaned_data
        if os.path.exists(upload_path):
             try:
                os.remove(upload_path)
                del session['upload_path'] # Remove from session if deleted
             except OSError as e:
                 print(f"Error removing temp upload file {upload_path}: {e}") # Log error but continue

        return redirect(url_for('clean_data'))

    else:
        flash('Invalid file type. Please upload a CSV file.', 'danger')
        return redirect(url_for('index'))

@app.route('/clean')
def clean_data():
    """Displays the data preview and cleaning options."""
    file_path = session.get('current_file_path')
    if not file_path or not os.path.exists(file_path):
        flash("No data file found in session or file missing. Please upload again.", "warning")
        return redirect(url_for('index'))

    try:
        # Always read the latest version from the cleaned folder
        # Assuming UTF-8 was used for saving cleaned files
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df.infer_objects() # Re-infer types after reading
        columns = df.columns.tolist()
        df_preview = df.head(100).to_html(classes='table table-striped table-sm', border=0, index=False)
        return render_template('clean.html', columns=columns, df_preview=df_preview)
    except Exception as e:
        flash(f"Error reading current data file: {str(e)}. You might need to re-upload.", "danger")
        session.pop('current_file_path', None) # Clear broken path
        session.pop('original_filename', None)
        return redirect(url_for('index'))


@app.route('/apply/<operation>', methods=['POST'])
def apply_cleaning(operation):
    """Applies the selected cleaning operation."""
    file_path = session.get('current_file_path')
    original_filename = session.get('original_filename')

    if not file_path or not original_filename or not os.path.exists(file_path):
        flash("Session expired or file missing. Please upload again.", "warning")
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file_path, encoding='utf-8') # Assume cleaned files are utf-8
        df = df.infer_objects()
        original_shape = df.shape
        cols_affected = []

        # --- Apply Operations ---
        if operation == 'missing':
            cols = request.form.getlist('columns') # Use getlist for multiple select
            method = request.form.get('missing_method')
            fill_val = request.form.get('fill_value')
            target_cols = cols if cols else df.columns

            cols_affected = list(target_cols) # Assume all selected/default columns could be affected

            if method == 'drop_row':
                subset = target_cols if cols else None # Drop if NaN in ANY column if cols is empty, else check subset
                df.dropna(subset=subset, inplace=True)
            elif method == 'drop_col':
                 # Only drop if ALL values in the column are NaN? Or any? Let's do 'any'.
                 # Calculate which cols to drop *before* dropping
                 cols_to_drop = [col for col in target_cols if df[col].isnull().any()]
                 cols_affected = cols_to_drop # More accurate
                 df.dropna(axis=1, how='any', subset=None, inplace=True) # Pandas dropna axis=1 doesn't use subset well for 'any'. It checks across all rows.
                 # We need to drop specific columns if they contain *any* NaN
                 df.drop(columns=cols_to_drop, inplace=True)
            else:
                # Fill methods - only apply to specified columns
                for col in target_cols:
                    if df[col].isnull().any(): # Only process columns with actual NaNs
                        try:
                            if method == 'fill_mean':
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    mean_val = df[col].mean()
                                    df[col].fillna(mean_val, inplace=True)
                                else:
                                    flash(f"Column '{col}' is not numeric. Cannot fill with mean.", "warning")
                            elif method == 'fill_median':
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    median_val = df[col].median()
                                    df[col].fillna(median_val, inplace=True)
                                else:
                                     flash(f"Column '{col}' is not numeric. Cannot fill with median.", "warning")
                            elif method == 'fill_mode':
                                # Mode can return multiple values, take the first
                                mode_val = df[col].mode()
                                if not mode_val.empty:
                                    df[col].fillna(mode_val[0], inplace=True)
                            elif method == 'fill_value':
                                # Try to convert fill_val to the column's dtype if possible
                                try:
                                    typed_fill_val = pd.Series([fill_val]).astype(df[col].dtype).iloc[0]
                                    df[col].fillna(typed_fill_val, inplace=True)
                                except (ValueError, TypeError):
                                     df[col].fillna(fill_val, inplace=True) # Fallback to raw value
                        except Exception as e:
                             flash(f"Error filling column '{col}': {e}", "danger")
                    else:
                         # Remove from affected if no NaNs were present
                         if col in cols_affected: cols_affected.remove(col)


        elif operation == 'outlier':
            col = request.form.get('column')
            method = request.form.get('outlier_method')
            threshold = float(request.form.get('threshold', 1.5)) # Default IQR threshold
            action = request.form.get('action')
            cols_affected = [col]

            if not pd.api.types.is_numeric_dtype(df[col]):
                flash(f"Column '{col}' is not numeric. Cannot perform outlier detection.", "warning")
            else:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col].dropna())) # Drop NaN before zscore calc
                    # We need to map z_scores back to original index if NaNs existed
                    z_series = pd.Series(z_scores, index=df[col].dropna().index)
                    # Define outliers based on original data using the threshold
                    is_outlier = df[col].apply(lambda x: z_series.get(x, 0) > threshold if pd.notna(x) else False)
                    # For capping, we need bounds - usually mean +/- threshold * std dev
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                else:
                     flash("Invalid outlier method.", "danger")
                     return redirect(url_for('clean_data'))

                if method in ['iqr', 'zscore']:
                     if action == 'remove':
                         if method == 'iqr':
                              outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
                         else: # zscore
                              outlier_condition = is_outlier
                         df = df[~outlier_condition]
                     elif action == 'cap':
                         df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


        elif operation == 'smooth':
             col = request.form.get('column')
             method = request.form.get('smooth_method')
             window = int(request.form.get('window', 3))
             cols_affected = [col]

             if not pd.api.types.is_numeric_dtype(df[col]):
                  flash(f"Column '{col}' is not numeric. Cannot smooth data.", "warning")
             elif window < 2:
                  flash("Window size must be at least 2.", "warning")
             else:
                 if method == 'moving_average':
                     df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean()
                     # Note: rolling mean can introduce NaNs at the edges if min_periods isn't 1
                 # Add other smoothing methods here (e.g., exponential)
                 else:
                     flash("Invalid smoothing method.", "danger")


        elif operation == 'normalize':
             cols = request.form.getlist('columns')
             method = request.form.get('normalize_method')
             numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
             target_cols = [c for c in cols if c in numeric_cols] if cols else numeric_cols
             cols_affected = target_cols

             if not target_cols:
                 flash("No numeric columns selected or found for normalization.", "warning")
             else:
                 for col in target_cols:
                     # Reshape data for scaler: scaler expects 2D array
                     data_to_scale = df[[col]].dropna() # Drop NaN before scaling
                     if data_to_scale.empty:
                         continue # Skip empty columns or columns with only NaNs

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


        elif operation == 'deduplicate':
             cols = request.form.getlist('columns')
             keep = request.form.get('keep')
             if keep == 'none': keep_param = False # drop_duplicates uses False for this
             else: keep_param = keep

             subset = cols if cols else None
             df.drop_duplicates(subset=subset, keep=keep_param, inplace=True)
             # Can't easily track affected columns here, affects rows


        elif operation == 'date':
             cols = request.form.getlist('columns')
             output_format = request.form.get('date_format') or None # None uses default pandas format
             errors_coerce = request.form.get('errors_coerce') == 'true'
             cols_affected = cols

             for col in cols:
                 try:
                    # Attempt conversion
                    original_dtype = df[col].dtype
                    df[col] = pd.to_datetime(df[col], errors='coerce' if errors_coerce else 'raise', infer_datetime_format=True)

                    # Apply formatting if specified and conversion was successful
                    if output_format and pd.api.types.is_datetime64_any_dtype(df[col]):
                         # Format valid dates, keep NaT as NaT
                         df[col] = df[col].dt.strftime(output_format)
                         # Note: strftime converts the column back to object/string type
                 except ValueError as e:
                     # This should only happen if errors='raise' and parsing fails
                     flash(f"Error parsing date in column '{col}': {e}. Try enabling 'Convert unparseable dates to NaT'.", "danger")
                 except Exception as e:
                     flash(f"An unexpected error occurred during date conversion for column '{col}': {e}", "danger")


        elif operation == 'case':
            cols = request.form.getlist('columns')
            case_type = request.form.get('case_type')
            cols_affected = cols

            for col in cols:
                 # Ensure the column is treated as string
                 if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                     df[col] = df[col].astype(str) # Convert potential mixed types just in case
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
                 else:
                      flash(f"Column '{col}' is not a text type. Cannot apply case standardization.", "warning")


        elif operation == 'constraint': # Basic numeric range constraint
            col = request.form.get('column')
            min_val_str = request.form.get('min_val')
            max_val_str = request.form.get('max_val')
            cols_affected = [col] # Constraint defined by this column

            if not pd.api.types.is_numeric_dtype(df[col]):
                flash(f"Column '{col}' is not numeric. Cannot apply numeric range constraint.", "warning")
            else:
                condition = pd.Series([True] * len(df), index=df.index) # Start with all true
                try:
                    if min_val_str:
                        min_val = float(min_val_str)
                        condition &= (df[col] >= min_val)
                    if max_val_str:
                        max_val = float(max_val_str)
                        condition &= (df[col] <= max_val)

                    df = df[condition] # Keep only rows matching the condition
                except ValueError:
                     flash("Invalid number entered for min/max value.", "danger")


        elif operation == 'sort':
            cols = request.form.getlist('columns')
            ascending_str = request.form.get('ascending', 'True')
            ascending = ascending_str == 'True' # Convert string 'True'/'False' to boolean
            df.sort_values(by=cols, ascending=ascending, inplace=True, ignore_index=True) # ignore_index resets index


        elif operation == 'rename':
             old_name = request.form.get('old_name')
             new_name = request.form.get('new_name')
             if not new_name:
                 flash("New column name cannot be empty.", "warning")
             elif new_name in df.columns and new_name != old_name:
                  flash(f"Column name '{new_name}' already exists.", "warning")
             elif old_name not in df.columns:
                   flash(f"Column '{old_name}' not found.", "warning")
             else:
                 df.rename(columns={old_name: new_name}, inplace=True)
                 flash(f"Renamed column '{old_name}' to '{new_name}'.", "success")
                 # No specific affected cols, structure change


        elif operation == 'remove':
            cols_to_remove = request.form.getlist('columns')
            cols_exist = [c for c in cols_to_remove if c in df.columns]
            cols_not_exist = [c for c in cols_to_remove if c not in df.columns]

            if cols_not_exist:
                 flash(f"Columns not found and could not be removed: {', '.join(cols_not_exist)}", "warning")

            if cols_exist:
                df.drop(columns=cols_exist, inplace=True)
                flash(f"Removed columns: {', '.join(cols_exist)}", "success")
            elif not cols_not_exist:
                 flash("No columns selected to remove.", "info")
             # Structure change


        else:
            flash(f"Unknown operation: {operation}", "danger")
            return redirect(url_for('clean_data'))

        # --- Save the modified DataFrame ---
        new_file_path = save_data(df, original_filename)
        if new_file_path:
            # Clean up the *previous* intermediate file before updating session
            if file_path != new_file_path and os.path.exists(file_path):
                 try:
                    os.remove(file_path)
                 except OSError as e:
                     print(f"Error removing intermediate file {file_path}: {e}") # Log error

            session['current_file_path'] = new_file_path # Update session with the path to the NEW file
            final_shape = df.shape
            rows_changed = original_shape[0] - final_shape[0]
            cols_changed = original_shape[1] - final_shape[1]

            change_msg = ""
            if rows_changed > 0: change_msg += f" {rows_changed} rows removed."
            if rows_changed < 0: change_msg += f" {-rows_changed} rows added (unexpected)." # Should not happen often
            if cols_changed > 0: change_msg += f" {cols_changed} columns removed."
            if cols_changed < 0: change_msg += f" {-cols_changed} columns added (unexpected)." # Only with complex ops
            if not change_msg and operation not in ['rename', 'sort']: # Some ops don't change shape but modify values
                 if cols_affected:
                      change_msg = f" Values potentially modified in column(s): {', '.join(cols_affected)}."
                 elif operation not in ['deduplicate']: # Deduplicate might not report cols but changes rows
                      change_msg = " Operation applied, but shape unchanged."


            flash(f"Operation '{operation}' applied successfully.{change_msg}", "success")
        else:
            # Saving failed, don't change session, flash message already set by save_data
            pass

        return redirect(url_for('clean_data'))

    except KeyError as e:
        flash(f"Error applying operation '{operation}': Missing column or incorrect parameter '{e}'. Check inputs.", "danger")
        return redirect(url_for('clean_data'))
    except ValueError as e:
         flash(f"Error applying operation '{operation}': Invalid value entered - {e}. Check numeric inputs.", "danger")
         return redirect(url_for('clean_data'))
    except Exception as e:
        flash(f"An unexpected error occurred during operation '{operation}': {str(e)}", "danger")
        # Optionally clear session if state might be corrupted
        # session.pop('current_file_path', None)
        # session.pop('original_filename', None)
        # return redirect(url_for('index'))
        return redirect(url_for('clean_data')) # Try returning to clean page


@app.route('/download')
def download_file():
    """Provides the current cleaned file for download."""
    file_path = session.get('current_file_path')
    original_filename = session.get('original_filename', 'cleaned_data.csv')

    if not file_path or not os.path.exists(file_path):
        flash("No file available for download or session expired.", "warning")
        return redirect(url_for('index'))

    try:
        # Ensure filename is secure and add a suffix
        base, ext = os.path.splitext(original_filename)
        download_name = f"{secure_filename(base)}_cleaned{ext}"

        return send_from_directory(
            directory=app.config['CLEANED_FOLDER'],
            path=os.path.basename(file_path),
            as_attachment=True,
            download_name=download_name # Use the modified name for download
        )
    except Exception as e:
        flash(f"Error preparing file for download: {str(e)}", "danger")
        return redirect(url_for('clean_data'))

# --- Cleanup Old Files (Optional - Basic Example) ---
# A more robust solution would use a background task or scheduler
# This basic example cleans on startup (not ideal for production)
def cleanup_old_files(folder, max_age_seconds=3600): # Clean files older than 1 hour
    now = time.time()
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if os.stat(file_path).st_mtime < now - max_age_seconds:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # Create folders if they don't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(CLEANED_FOLDER):
        os.makedirs(CLEANED_FOLDER)
    if not os.path.exists('static/css'):
         os.makedirs('static/css', exist_ok=True)

    # Optional: Basic cleanup on start (consider moving to a scheduled task)
    # import time
    # cleanup_old_files(app.config['UPLOAD_FOLDER'])
    # cleanup_old_files(app.config['CLEANED_FOLDER'])

    app.run(debug=True) # Set debug=False for production