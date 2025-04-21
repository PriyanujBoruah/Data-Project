import os
import uuid
import pandas as pd
import numpy as np
from flask import (Flask, request, render_template, redirect, url_for,
                   send_from_directory, flash, session, make_response, Response) # Added Response
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import scipy.stats as stats
# import chardet # Not needed if only using paste/Supabase for intermediate
from collections import Counter
import time
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype, is_object_dtype, is_bool_dtype, is_categorical_dtype
import io # Still needed for StringIO and BytesIO
from thefuzz import fuzz
import re
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation
import plotly.express as px
import plotly.io as pio
import traceback # For detailed error logging

# --- Supabase Integration ---
from supabase import create_client, Client
from dotenv import load_dotenv # For local development

load_dotenv() # Load environment variables from .env file if it exists

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Keep for potential future hybrid use
CLEANED_FOLDER = 'cleaned_data' # Base name, not used for primary storage anymore
# ALLOWED_EXTENSIONS = {'csv', 'xlsx'} # Keep if hybrid needed
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB limit

app = Flask(__name__)
# Increase session cookie size if needed (object paths can be long)
# app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
# IMPORTANT: Use a strong, random secret key! Read from env var.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-this-in-production-env')
if app.secret_key == 'change-this-in-production-env':
    print("WARNING: Using default FLASK_SECRET_KEY. Set a strong secret in environment variables for production!")


# --- Supabase Client Initialization ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use service role key
BUCKET_NAME = "cleaned-files" # <<<--- REPLACE WITH YOUR BUCKET NAME ---<<<

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Supabase client: {e}")
        # Optional: Exit or disable features if Supabase is essential
        # sys.exit("Exiting due to Supabase connection failure.")
else:
    print("WARNING: Supabase URL or Key not found in environment variables. External storage will fail.")

# --- Plotting Configuration ---
# Consistent plot appearance
pio.templates.default = "plotly_white"
DEFAULT_PLOT_LAYOUT = {
    'title_x': 0.5, # Center titles
    'height': 375,  # Default height
    'margin': dict(l=50, r=30, t=60, b=40) # Consistent margins
}


# --- NEW DATA ANALYSIS HELPER FUNCTIONS ---

def update_layout(fig, title):
    """Helper to apply default layout updates."""
    fig.update_layout(title=title, **DEFAULT_PLOT_LAYOUT)
    return fig

# --- Column Type Identification (Slightly enhanced) ---
def get_column_types(df):
    """Categorizes columns into types for UI dropdowns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Treat boolean as categorical for grouping/plotting
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category', 'boolean']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    low_cardinality_cols = []
    high_cardinality_cols = []
    for col in categorical_cols:
        nunique = df[col].nunique()
        # Define thresholds (adjust as needed)
        if nunique <= 1: # Constant columns (might already be handled by cleaning)
             pass # Or add to a separate list if needed
        elif nunique <= 50:
            low_cardinality_cols.append(col)
        else:
            # Flag likely IDs (very high unique count relative to rows) separately?
            # if nunique / len(df) > 0.95: # Example heuristic
            #     pass # Treat as ID
            # else:
            high_cardinality_cols.append(col)

    other_cols = df.columns.difference(numeric_cols + categorical_cols + datetime_cols).tolist()

    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols, # All object/string/cat/bool
        'low_cardinality_categorical': low_cardinality_cols, # Suitable for x-axis/color
        'high_cardinality_categorical': high_cardinality_cols, # Maybe only for info
        'datetime': datetime_cols,
        'other': other_cols
    }

def analyze_numeric_column(df, col_name):
    """Generates enhanced stats and plots for a single numeric column."""
    if col_name not in df.columns or not is_numeric_dtype(df[col_name]):
        flash(f"'{col_name}' is not a valid numeric column.", "warning")
        return None, None, None

    col_data = df[col_name].dropna()
    if col_data.empty:
        flash(f"Numeric column '{col_name}' contains no non-missing values.", "info")
        return {'count': 0, 'missing': df[col_name].isnull().sum()}, None, None

    # Calculate Enhanced Stats
    stats_dict = {}
    desc = col_data.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(4)
    stats_dict.update(desc.to_dict())
    stats_dict['median'] = round(col_data.median(), 4) # Explicit median
    try: stats_dict['skewness'] = round(col_data.skew(), 4)
    except Exception: stats_dict['skewness'] = 'N/A'
    try: stats_dict['kurtosis'] = round(col_data.kurt(), 4)
    except Exception: stats_dict['kurtosis'] = 'N/A'
    if stats_dict.get('mean') and abs(stats_dict['mean']) > 1e-9: # Avoid division by zero
        stats_dict['coeff_variation'] = round(stats_dict.get('std', 0) / stats_dict['mean'], 4)
    else:
         stats_dict['coeff_variation'] = 'N/A'
    try: # Median Absolute Deviation
        mad = median_abs_deviation(col_data, scale='normal', nan_policy='omit')
        stats_dict['mad'] = round(mad, 4)
    except Exception: stats_dict['mad'] = 'N/A'
    stats_dict['num_zeros'] = int((col_data == 0).sum())
    stats_dict['missing'] = int(df[col_name].isnull().sum())
    stats_dict['count_total'] = len(df[col_name]) # Total including missing

    # Generate Plots
    hist_div, box_div = None, None
    try:
        fig_hist = px.histogram(col_data, x=col_name, marginal="box") # Histogram with box plot
        hist_div = pio.to_html(update_layout(fig_hist, f"Distribution of {col_name}"), full_html=False, include_plotlyjs=False)
    except Exception as e: print(f"Error plot hist {col_name}: {e}")

    try:
        fig_box = px.box(col_data, y=col_name, points="outliers") # Separate Box Plot
        box_div = pio.to_html(update_layout(fig_box, f"Box Plot of {col_name}"), full_html=False, include_plotlyjs=False)
    except Exception as e: print(f"Error plot box {col_name}: {e}")

    return stats_dict, hist_div, box_div

def analyze_categorical_column(df, col_name, top_n=20):
    """Generates value counts table and bar chart for categorical columns."""
    if col_name not in df.columns: # Basic check
        flash(f"Column '{col_name}' not found.", "warning")
        return None, None
    # Check if it's treatable as categorical
    if not (is_string_dtype(df[col_name]) or is_object_dtype(df[col_name]) or pd.api.types.is_categorical_dtype(df[col_name]) or is_bool_dtype(df[col_name])):
         flash(f"Column '{col_name}' is not a recognized categorical type.", "warning")
         return None, None

    col_data = df[col_name].astype(str).fillna("(Missing)") # Treat uniformly as string
    counts = col_data.value_counts()

    if counts.empty:
        flash(f"Categorical column '{col_name}' contains no values.", "info")
        return "No data available.", None

    # Prepare data for table (Top N + Other)
    if len(counts) > top_n:
        top_counts = counts.head(top_n)
        other_count = counts.iloc[top_n:].sum()
        counts_display = pd.concat([top_counts, pd.Series({'*Other*': other_count})]) if other_count > 0 else top_counts
    else:
        counts_display = counts

    counts_df = counts_display.reset_index()
    counts_df.columns = ['Value', 'Count']
    counts_df['Percentage'] = (counts_df['Count'] / len(col_data) * 100).round(2)
    counts_html = counts_df.to_html(classes='table table-sm table-striped', index=False, border=0)

    # Generate Bar Chart
    plot_div = None
    try:
        fig = px.bar(counts_df.head(top_n + 1), x='Value', y='Count', text='Percentage') # Show percentage on bars
        fig.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
        fig = update_layout(fig, f"Top {len(counts_df)} Value Counts for {col_name}")
        if len(counts_df) > 10: fig.update_xaxes(tickangle=-45) # Angle labels if many cats
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    except Exception as e: print(f"Error plot bar {col_name}: {e}")

    return counts_html, plot_div


def analyze_datetime_column(df, col_name):
    """Generates info and timeline plot for datetime columns."""
    if col_name not in df.columns or not is_datetime64_any_dtype(df[col_name]):
        flash(f"'{col_name}' is not a datetime column.", "warning")
        return None, None

    col_data = df[col_name].dropna()
    stats_dict = {
        'missing_count': int(df[col_name].isnull().sum()),
        'unique_count': 0, 'min': 'N/A', 'max': 'N/A', 'duration': 'N/A'
    }
    plot_div = None

    if not col_data.empty:
        stats_dict['min'] = str(col_data.min())
        stats_dict['max'] = str(col_data.max())
        stats_dict['unique_count'] = col_data.nunique()
        duration = col_data.max() - col_data.min()
        stats_dict['duration'] = str(duration)

        # Generate Timeline Plot (Counts per time period)
        try:
            # Dynamically choose resampling period (or let histogram auto-bin)
            time_range_days = (col_data.max() - col_data.min()).days
            if time_range_days > 365 * 2: freq = 'M' # Monthly if > 2 years
            elif time_range_days > 60: freq = 'W' # Weekly if > 2 months
            else: freq = 'D' # Daily otherwise
            # Example using histogram auto-binning - often good enough
            fig = px.histogram(col_data, x=col_name)
            fig = update_layout(fig, f"Record Count Over Time for {col_name}")
            plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        except Exception as e: print(f"Error plot time {col_name}: {e}")
    else:
        flash(f"Datetime column '{col_name}' has no non-missing values.", "info")

    return stats_dict, plot_div

def analyze_numeric_vs_numeric(df, col_x, col_y):
    """Generates correlation, scatter plot for two numeric columns."""
    if col_x not in df.columns or not is_numeric_dtype(df[col_x]) or \
       col_y not in df.columns or not is_numeric_dtype(df[col_y]):
       flash("Both selected columns must be numeric.", "warning")
       return None, None, None

    results = {'correlation_pearson': 'N/A', 'correlation_spearman': 'N/A', 'p_value_pearson': 'N/A'}
    plot_div = None
    df_pair = df[[col_x, col_y]].dropna()

    if len(df_pair) > 1:
        try: # Pearson (Linear)
            corr_p, p_val_p = stats.pearsonr(df_pair[col_x], df_pair[col_y])
            results['correlation_pearson'] = round(corr_p, 4)
            results['p_value_pearson'] = f"{p_val_p:.3g}" # Scientific notation if small
        except Exception as e: print(f"Pearson corr fail {col_x}v{col_y}: {e}")

        try: # Spearman (Monotonic)
            corr_s, p_val_s = stats.spearmanr(df_pair[col_x], df_pair[col_y])
            results['correlation_spearman'] = round(corr_s, 4)
            # results['p_value_spearman'] = f"{p_val_s:.3g}" # Optional to show
        except Exception as e: print(f"Spearman corr fail {col_x}v{col_y}: {e}")

        try: # Scatter Plot
            fig = px.scatter(df_pair, x=col_x, y=col_y, trendline="ols", trendline_color_override="red", opacity=0.7)
            fig = update_layout(fig, f"Scatter Plot: {col_x} vs {col_y}")
            fig.update_layout(height=400) # Slightly taller scatter
            plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        except Exception as e: print(f"Scatter plot fail {col_x}v{col_y}: {e}")
    else:
        flash("Not enough non-missing data points for analysis.", "info")


    return results, plot_div

def analyze_numeric_vs_categorical(df, num_col, cat_col):
    """Generates box/violin plots for numeric grouped by categorical."""
    if num_col not in df.columns or not is_numeric_dtype(df[num_col]) or cat_col not in df.columns:
       flash("Invalid columns selected.", "warning"); return None, None

    box_div, violin_div = None, None
    df_pair = df[[num_col, cat_col]].dropna(subset=[num_col])
    df_pair[cat_col] = df_pair[cat_col].astype(str).fillna("(Missing)")
    num_categories = df_pair[cat_col].nunique()

    if not df_pair.empty and num_categories > 0:
        # Only generate plots if categories are manageable
        if num_categories > 50:
            flash(f"Too many categories ({num_categories}) in '{cat_col}' for Box/Violin plots.", "warning")
            return None, None
        try: # Box Plot
            fig_box = px.box(df_pair, x=cat_col, y=num_col, points="outliers")
            fig_box = update_layout(fig_box, f"Box Plot of {num_col} by {cat_col}")
            if num_categories > 8: fig_box.update_xaxes(tickangle=-45)
            box_div = pio.to_html(fig_box, full_html=False, include_plotlyjs=False)
        except Exception as e: print(f"Box plot fail {num_col}v{cat_col}: {e}")

        try: # Violin Plot
            fig_violin = px.violin(df_pair, x=cat_col, y=num_col, box=True, points="outliers") # Show box inside
            fig_violin = update_layout(fig_violin, f"Violin Plot of {num_col} by {cat_col}")
            if num_categories > 8: fig_violin.update_xaxes(tickangle=-45)
            violin_div = pio.to_html(fig_violin, full_html=False, include_plotlyjs=False)
        except Exception as e: print(f"Violin plot fail {num_col}v{cat_col}: {e}")
    else:
         flash("Not enough data for numeric vs categorical analysis.", "info")

    return box_div, violin_div

def analyze_categorical_vs_categorical(df, cat_col1, cat_col2):
    """Generates cross-tab, Chi-squared test, and optional heatmap."""
    if cat_col1 not in df.columns or cat_col2 not in df.columns:
        flash("Invalid columns selected.", "warning"); return None, None, None

    crosstab_html = None
    heatmap_div = None
    test_results = {'chi2_statistic': 'N/A', 'p_value': 'N/A', 'dof': 'N/A', 'interpretation': 'N/A', 'warning': None}

    try:
        # Treat as string and handle missing explicitly
        cat1_series = df[cat_col1].astype(str).fillna("(Missing)")
        cat2_series = df[cat_col2].astype(str).fillna("(Missing)")

        # Crosstab with totals
        crosstab_df = pd.crosstab(cat1_series, cat2_series, margins=True, margins_name="Total")
        crosstab_html = crosstab_df.to_html(classes='table table-sm table-bordered', border=1)

        # Chi-Squared Test (only if enough data and categories)
        observed = pd.crosstab(cat1_series, cat2_series) # No margins for test
        if observed.shape[0] > 1 and observed.shape[1] > 1 and observed.sum().sum() > 0:
             try:
                 chi2, p, dof, expected = stats.chi2_contingency(observed)
                 test_results['chi2_statistic'] = round(chi2, 4)
                 test_results['p_value'] = f"{p:.3g}"
                 test_results['dof'] = dof
                 alpha = 0.05
                 if p < alpha:
                     test_results['interpretation'] = f"Significant association found (p < {alpha})."
                 else:
                     test_results['interpretation'] = f"No significant association found (p >= {alpha})."
                 # Check for low expected frequencies (assumption of Chi-Squared)
                 if (expected < 5).any().any():
                     test_results['warning'] = "Warning: Some expected frequencies are low (< 5), Chi-Squared test results may be unreliable."
                     flash(test_results['warning'], "warning")
             except ValueError as ve: # e.g., if table has zeros in places preventing calculation
                  test_results['interpretation'] = f"Chi-squared test could not be performed ({ve})."
                  print(f"Chi2 fail {cat_col1}v{cat_col2}: {ve}")
             except Exception as e:
                  test_results['interpretation'] = "Error during Chi-squared test."
                  print(f"Chi2 fail {cat_col1}v{cat_col2}: {e}")
        else:
            test_results['interpretation'] = "Chi-squared test requires at least 2 rows and 2 columns in the contingency table."


        # Optional: Heatmap for counts (if not too large)
        if observed.shape[0] <= 30 and observed.shape[1] <= 30: # Limit heatmap size
            try:
                fig_heatmap = px.imshow(observed, text_auto=True, aspect="auto", title=f"Heatmap of Counts: {cat_col1} vs {cat_col2}")
                fig_heatmap = update_layout(fig_heatmap, f"Counts: {cat_col1} vs {cat_col2}")
                heatmap_div = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False)
            except Exception as e: print(f"Crosstab heatmap fail: {e}")

    except Exception as e:
        print(f"Error generating crosstab for {cat_col1} vs {cat_col2}: {e}")
        flash(f"Could not generate cross-tabulation: {e}", "danger")

    return crosstab_html, heatmap_div, test_results


def generate_correlation_heatmap(df, method='pearson'):
    """Generates correlation heatmap for numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    plot_div = None
    if len(numeric_cols) < 2:
        flash("Need at least two numeric columns for correlation.", "info"); return None
    if method not in ['pearson', 'spearman', 'kendall']:
        flash("Invalid correlation method specified.", "warning"); return None

    try:
        corr_matrix = df[numeric_cols].corr(method=method).round(2)
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig = update_layout(fig, f"{method.capitalize()} Correlation Heatmap")
        fig.update_layout(height=max(450, len(numeric_cols)*25 + 50)) # Dynamic height
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    except Exception as e: print(f"Corr heatmap fail: {e}"); flash("Could not generate correlation heatmap.", "warning")
    return plot_div


def generate_grouped_summary(df, group_by_cols, value_cols, agg_funcs):
    """Performs groupby aggregation and returns HTML table."""
    if not group_by_cols or not value_cols or not agg_funcs:
        flash("Ensure Group By, Value, and Function fields are selected.", "warning"); return None

    valid_group_by = [col for col in group_by_cols if col in df.columns]
    valid_value_cols = [col for col in value_cols if col in df.columns and is_numeric_dtype(df[col])]
    # Simple validation for agg funcs - more robust checking might be needed
    valid_agg_funcs = [f for f in list(set(agg_funcs)) if f in ['mean', 'median', 'sum', 'count', 'std', 'min', 'max', 'nunique']]

    if not valid_group_by or not valid_value_cols or not valid_agg_funcs:
         missing = [item for sublist in [group_by_cols, value_cols, agg_funcs] for item in sublist if item not in valid_group_by + valid_value_cols + valid_agg_funcs]
         flash(f"Invalid selections for Group By operation. Check columns/functions. Missing/Invalid: {missing}", "warning"); return None

    try:
        agg_dict = {val_col: valid_agg_funcs for val_col in valid_value_cols}
        summary_df = df.groupby(valid_group_by, observed=False).agg(agg_dict).round(4) # observed=False is safer default

        if isinstance(summary_df.columns, pd.MultiIndex):
             summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]

        summary_html = summary_df.reset_index().to_html(classes='table table-sm table-striped', index=False, border=0, na_rep='-')
        return summary_html
    except Exception as e: print(f"Groupby fail: {e}"); flash(f"Group By operation failed: {e}", "danger"); return None


def perform_ttest_ind(df, num_col, bin_cat_col):
    """Performs independent T-test."""
    results = {'statistic': 'N/A', 'p_value': 'N/A', 'interpretation': 'N/A', 'groups_compared': 'N/A', 'warning': None}
    if num_col not in df.columns or not is_numeric_dtype(df[num_col]) or \
       bin_cat_col not in df.columns:
        results['interpretation'] = "Invalid columns selected."; return results

    # Ensure categorical column is truly binary (after dropping NaNs)
    valid_data = df[[num_col, bin_cat_col]].dropna()
    groups = valid_data[bin_cat_col].unique()
    if len(groups) != 2:
         results['interpretation'] = f"Categorical column '{bin_cat_col}' must have exactly 2 unique non-missing values for T-test (found {len(groups)})."; return results

    group1_data = valid_data[valid_data[bin_cat_col] == groups[0]][num_col]
    group2_data = valid_data[valid_data[bin_cat_col] == groups[1]][num_col]
    results['groups_compared'] = f"{groups[0]} vs {groups[1]}"

    if len(group1_data) < 2 or len(group2_data) < 2:
        results['interpretation'] = "Not enough data in one or both groups for T-test."; return results

    try:
        # Welch's t-test (doesn't assume equal variances) is generally safer
        stat, p = stats.ttest_ind(group1_data, group2_data, equal_var=False, nan_policy='omit')
        results['statistic'] = round(stat, 4)
        results['p_value'] = f"{p:.3g}"
        alpha = 0.05
        if p < alpha: results['interpretation'] = f"Means are significantly different (p < {alpha})."
        else: results['interpretation'] = f"No significant difference in means found (p >= {alpha})."
    except Exception as e: print(f"T-test fail: {e}"); results['interpretation'] = f"Error during T-test: {e}"
    return results

def perform_anova(df, num_col, cat_col):
    """Performs One-Way ANOVA."""
    results = {'statistic': 'N/A', 'p_value': 'N/A', 'interpretation': 'N/A', 'num_groups': 0, 'warning': None}
    if num_col not in df.columns or not is_numeric_dtype(df[num_col]) or \
       cat_col not in df.columns:
        results['interpretation'] = "Invalid columns selected."; return results

    valid_data = df[[num_col, cat_col]].dropna()
    unique_groups = valid_data[cat_col].unique()
    results['num_groups'] = len(unique_groups)

    if len(unique_groups) < 2:
        results['interpretation'] = "ANOVA requires at least 2 groups."; return results

    # Prepare data as a list of arrays, one for each group
    group_data = [valid_data[valid_data[cat_col] == group][num_col] for group in unique_groups]
    # Filter out groups with insufficient data (<2) for ANOVA calculation
    group_data_filtered = [g for g in group_data if len(g) >= 2]

    if len(group_data_filtered) < 2: # Need at least two valid groups
         results['interpretation'] = "Not enough groups with sufficient data (>1 sample) for ANOVA."; return results
    if len(group_data_filtered) < len(group_data):
         results['warning'] = f"Warning: {len(group_data) - len(group_data_filtered)} group(s) excluded from ANOVA due to insufficient data."
         flash(results['warning'], 'warning')

    try:
        stat, p = stats.f_oneway(*group_data_filtered)
        results['statistic'] = round(stat, 4)
        results['p_value'] = f"{p:.3g}"
        alpha = 0.05
        if p < alpha: results['interpretation'] = f"At least one group mean is significantly different (p < {alpha})."
        else: results['interpretation'] = f"No significant difference between group means found (p >= {alpha})."
        # Add reminder about post-hoc tests if significant?
        # if p < alpha: results['interpretation'] += " Consider post-hoc tests for pairwise comparisons."
    except Exception as e: print(f"ANOVA fail: {e}"); results['interpretation'] = f"Error during ANOVA: {e}"
    return results


# --- END OF NEW DATA ANALYSIS HELPER FUNCTIONS ---



# --- Helper Functions ---

def load_data(file_path=None, file_extension=None, encoding=None, text_data=None):
    """
    Loads data from a text string (assumed CSV) into a pandas DataFrame.
    File path/extension parameters are kept for potential future compatibility
    but are not used in the current paste-only workflow.
    """
    df = None
    used_encoding = None # Track encoding used

    try:
        # --- Handle pasted text data (assumed CSV) ---
        if text_data:
            used_encoding = encoding if encoding else 'utf-8' # Default for pasted text
            print(f"Attempting to read pasted CSV data with encoding: {used_encoding}")
            try:
                # Use io.StringIO to treat the string as a file
                string_io = io.StringIO(text_data)
                df = pd.read_csv(string_io, encoding=used_encoding)
                print(f"Successfully read DataFrame from pasted text.")
                # Optional: Add specific flash message here, or handle it in the calling route
                # flash(f"Read pasted data using encoding: {used_encoding}", "info")

            except UnicodeDecodeError as e:
                error_msg = f"Error decoding pasted CSV data with '{used_encoding}'. Common alternatives: 'utf-8', 'latin-1', 'cp1252'. Please specify encoding if default failed. Error: {e}"
                print(error_msg)
                flash(error_msg, "danger")
                return None, used_encoding # Return encoding tried

            except pd.errors.EmptyDataError:
                error_msg = "Error parsing pasted data: The provided text was empty or contained no data."
                print(error_msg)
                flash(error_msg, "danger")
                return None, used_encoding

            except Exception as e:
                # Catch other pandas parsing errors (e.g., CParserError for malformed CSV)
                error_msg = f"Error parsing pasted CSV data: {str(e)}. Check format (e.g., headers, delimiter consistency)."
                print(error_msg)
                flash(error_msg, "danger")
                return None, used_encoding

        # --- Placeholder for File path handling (Currently not used) ---
        elif file_path and file_extension:
             # This block would contain the logic for reading CSV/XLSX from disk
             # if you were to re-implement file uploads alongside pasting.
             # For now, it's unreachable in the paste-only workflow.
             print(f"Warning: load_data called with file_path ('{file_path}'), but only paste input is handled.")
             flash("Internal configuration error: File upload path used unexpectedly.", "danger")
             return None, None
        else:
             # This case should ideally not happen if called correctly from /upload
             print("Error: load_data called without text_data.")
             flash("Internal Error: No data provided to load.", "danger")
             return None, None

        # --- Common post-load processing ---
        if df is not None:
            if df.empty:
                flash("Loaded data is empty after parsing. Check the source text.", "warning")
                # Return empty df, let subsequent steps handle it
            else:
                 # Attempt basic type inference improvement
                 print("Inferring object types...")
                 df = df.infer_objects()
                 print("DataFrame loaded successfully and types inferred.")
        return df, used_encoding

    except Exception as e: # Catch-all for unexpected issues during load attempt
        error_msg = f"Unexpected error during data loading: {str(e)}"
        print(error_msg)
        traceback.print_exc() # Print full traceback for debugging
        flash(error_msg, "danger")
        return None, used_encoding

# Function to save data to Supabase Storage
def save_data_to_supabase(df, original_filename_base):
    """Saves the DataFrame to Supabase Storage as a CSV."""
    if not supabase:
        flash("Supabase client not initialized. Cannot save data.", "danger")
        return None

    secure_base = secure_filename(original_filename_base)
    if not secure_base: secure_base = "data"
    unique_id = uuid.uuid4().hex
    # Define the path within the bucket (e.g., using a subdirectory)
    object_path = f"intermediate/{unique_id}_{secure_base}.csv"

    try:
        # Convert DataFrame to CSV string in memory
        csv_string = df.to_csv(index=False)
        # Encode to bytes for upload
        csv_bytes = csv_string.encode('utf-8')

        # Upload to Supabase Storage
        print(f"Attempting Supabase upload: Bucket='{BUCKET_NAME}', Path='{object_path}'")
        # Use storage_options for cache control if desired, e.g., {'cacheControl': '3600'}
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=object_path,
            file=csv_bytes,
            file_options={"content-type": "text/csv", "cache-control": "no-cache"} # No cache for intermediate files
        )
        print(f"Supabase upload finished for {object_path}.")
        # In supabase-py v2, success means no exception. Response has metadata.
        # We just need the object_path to store in the session.
        return object_path

    except Exception as e:
        print(f"ERROR uploading to Supabase: {type(e).__name__} - {e}")
        traceback.print_exc()
        flash(f"Error saving intermediate data to cloud storage: {e}", "danger")
        return None

# Function to read data from Supabase Storage
def read_data_from_supabase(object_path):
    """Reads CSV data from Supabase Storage and returns a DataFrame."""
    if not supabase:
        flash("Supabase client not initialized. Cannot read data.", "danger")
        return None
    if not object_path:
         flash("Invalid intermediate data path provided.", "danger")
         return None

    print(f"Attempting Supabase download: Bucket='{BUCKET_NAME}', Path='{object_path}'")
    try:
        # Download the file content as bytes
        response_bytes = supabase.storage.from_(BUCKET_NAME).download(object_path)
        print(f"Supabase download finished for {object_path}. Bytes received: {len(response_bytes)}")

        # Decode bytes to string
        decoded_content = response_bytes.decode('utf-8')

        # Read string into DataFrame
        string_io = io.StringIO(decoded_content)
        df = pd.read_csv(string_io)
        df = df.infer_objects() # Ensure types are inferred
        print(f"Successfully read DataFrame from Supabase object: {object_path}")
        return df

    except Exception as e:
        # Handle common errors (e.g., file not found which might raise StorageApiError or similar)
        print(f"ERROR downloading or parsing from Supabase: {type(e).__name__} - {e}")
        # Check if the error message indicates "Not Found" or similar
        if "Not Found" in str(e) or "OBJECT_NOT_FOUND" in str(e):
             flash(f"Error: Intermediate data file not found in cloud storage ('{object_path}'). Session might be old or file was deleted.", "danger")
        else:
             flash(f"Error reading intermediate data from cloud storage: {e}", "danger")
        traceback.print_exc()
        return None


# --- auto_explore_data function remains the same (as modified previously) ---
def auto_explore_data(df):
    # (Keep the version of this function from previous steps that adds 'affected_columns')
    """
    Analyzes the DataFrame and returns a list of findings/suggestions
    with more specific actions recommended and affected columns.
    """
    findings = []
    if df is None or df.empty:
        findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'No data loaded or DataFrame is empty.', 'details': [], 'suggestion': 'Upload or paste valid CSV data.'})
        return findings

    num_rows, num_cols = df.shape
    if num_rows == 0 or num_cols == 0:
         findings.append({'issue_type': 'Data Error', 'severity': 'High', 'message': 'DataFrame has zero rows or columns.', 'details': [], 'suggestion': 'Check the pasted data or uploaded file content.'})
         return findings

    def format_cols_for_suggestion(col_list, max_cols=3):
        if not col_list: return ""
        cols_to_show = [f"'{c}'" for c in col_list[:max_cols]]
        suffix = "..." if len(col_list) > max_cols else ""
        return f" on columns like [{', '.join(cols_to_show)}{suffix}]"

    # Finding 1: Missing Data Analysis
    missing_summary = df.isnull().sum()
    missing_cols_series = missing_summary[missing_summary > 0]
    if not missing_cols_series.empty:
        total_missing = missing_cols_series.sum()
        pct_total_missing = (total_missing / (num_rows * num_cols)) * 100 if num_rows * num_cols > 0 else 0
        severity = 'High' if pct_total_missing > 10 else ('Medium' if pct_total_missing > 1 else 'Low')
        affected_cols_list = missing_cols_series.index.tolist()
        affected_cols_str = format_cols_for_suggestion(affected_cols_list)
        suggestion = f"Use '1. Clean Missing Data'{affected_cols_str} to handle NaNs (e.g., Action: 'Fill with Median/Mode' or 'Drop Rows')."
        findings.append({
            'issue_type': 'Missing Data',
            'severity': severity,
            'message': f"Found {total_missing} missing values ({pct_total_missing:.2f}% of total cells) across {len(affected_cols_list)} column(s).",
            'details': [f"'{col}': {count} missing ({ (count/num_rows)*100:.1f}%)" for col, count in missing_cols_series.items()],
            'suggestion': suggestion,
            'affected_columns': affected_cols_list
        })

    # Finding 2: Data Type Overview
    dtypes = df.dtypes
    object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    findings.append({
        'issue_type': 'Data Types',
        'severity': 'Info',
        'message': f"Dataset has {num_cols} columns: {len(numeric_cols)} numeric, {len(object_cols)} text/object, {len(datetime_cols)} datetime.",
        'details': [f"'{col}': {str(dtype)}" for col, dtype in dtypes.items()],
        'suggestion': "Review data types using 'Data Profile' or '8. Convert Data Type'."
    })

    # Finding 3: Potential Outliers (IQR)
    outlier_details_list = []
    outlier_columns_affected = []
    for col in numeric_cols:
        if df[col].isnull().all() or df[col].nunique() < 2: continue
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or pd.isna(IQR): continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if pd.isna(lower_bound) or pd.isna(upper_bound): continue
            outliers = df.loc[df[col].notna(), col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                outlier_count = outliers.count()
                non_na_count = df[col].notna().sum()
                pct_outliers = (outlier_count / non_na_count) * 100 if non_na_count > 0 else 0
                if pct_outliers > 0.1:
                     outlier_details_list.append(f"'{col}': {outlier_count} potential outliers ({pct_outliers:.2f}%)")
                     outlier_columns_affected.append(col)
        except Exception as e: print(f"Outlier check failed for column '{col}': {e}")
    if outlier_details_list:
         affected_cols_str = format_cols_for_suggestion(outlier_columns_affected)
         suggestion = f"Use '2. Clean Outlier Data'{affected_cols_str} (e.g., Method: 'IQR', Action: 'Cap Value' or 'Remove Row')."
         findings.append({
            'issue_type': 'Potential Outliers',
            'severity': 'Medium',
            'message': f"Potential outliers detected in {len(outlier_columns_affected)} numeric column(s) using IQR method.",
            'details': outlier_details_list,
            'suggestion': suggestion,
            'affected_columns': outlier_columns_affected
        })

    # Finding 4: Duplicate Rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        pct_duplicates = (duplicate_count / num_rows) * 100 if num_rows > 0 else 0
        severity = 'Medium' if pct_duplicates > 5 else 'Low'
        suggestion = f"Use '5. Deduplicate Records' (e.g., Action: 'Keep First') to handle {duplicate_count} duplicate rows."
        findings.append({
            'issue_type': 'Duplicate Records',
            'severity': severity,
            'message': f"Found {duplicate_count} duplicate rows ({pct_duplicates:.2f}% of total).",
            'details': [],
            'suggestion': suggestion
            # No 'affected_columns' needed for auto-clean
        })

    # Finding 5: Low Variance / Constant Columns
    constant_cols = []
    low_variance_cols = []
    low_var_cols_for_finding = [] # Store names for finding
    if num_rows > 0:
        for col in df.columns:
            nunique = df[col].nunique(dropna=False)
            if nunique <= 1:
                 constant_cols.append(col)
                 low_var_cols_for_finding.append(col)
            elif nunique > 1 and num_rows > 1 and (nunique / num_rows) < 0.01:
                 if not (nunique > num_rows * 0.95):
                     low_variance_cols.append(col)
                     low_var_cols_for_finding.append(col)
    low_var_msgs = []
    if constant_cols: low_var_msgs.append(f"Constant columns: {', '.join([f'`{c}`' for c in constant_cols])}")
    if low_variance_cols: low_var_msgs.append(f"Low variance columns (<1% unique): {', '.join([f'`{c}`' for c in low_variance_cols])}")
    if low_var_msgs:
        affected_cols_str = format_cols_for_suggestion(low_var_cols_for_finding)
        suggestion = f"Consider using '14. Remove Variable (Column)'{affected_cols_str} if these are uninformative."
        findings.append({
            'issue_type': 'Low Variance',
            'severity': 'Low',
            'message': "Found columns with very few unique values.",
            'details': low_var_msgs,
            'suggestion': suggestion,
            'affected_columns': low_var_cols_for_finding # Store for hard-clean
        })

    # Finding 6: Potential Date Columns
    potential_date_cols_details = []
    potential_date_cols_names = []
    for col in object_cols:
        if df[col].isnull().all(): continue
        non_na_series = df[col].dropna().astype(str)
        if non_na_series.empty: continue
        actual_sample_size = min(100, len(non_na_series))
        if actual_sample_size == 0: continue
        try:
            sample = non_na_series.sample(actual_sample_size, random_state=1)
            contains_digit_sep = sample.str.contains(r'[\d/\-:]', na=False).mean() > 0.6
            parseable_fraction = 0.0
            if contains_digit_sep:
                try:
                    parsed_dates = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                    parseable_fraction = parsed_dates.notna().mean()
                except Exception: parseable_fraction = 0.0
            if parseable_fraction > 0.7 and contains_digit_sep:
                potential_date_cols_details.append(f"'{col}' (Sample parse rate: {parseable_fraction:.1%})")
                potential_date_cols_names.append(col)
        except ValueError as ve: print(f"Sample size issue for date check on column '{col}': {ve}")
        except Exception as e: print(f"Date check failed for column '{col}': {e}")
    if potential_date_cols_details:
         affected_cols_str = format_cols_for_suggestion(potential_date_cols_names)
         suggestion = f"Use '8. Convert Data Type'{affected_cols_str} selecting Target Type: 'Date/Time'."
         findings.append({
            'issue_type': 'Potential Dates',
            'severity': 'Info',
            'message': f"Found {len(potential_date_cols_names)} text column(s) that might contain dates.",
            'details': potential_date_cols_details,
            'suggestion': suggestion,
            'affected_columns': potential_date_cols_names
        })

    # Finding 7: Text Issues (Case & Whitespace)
    text_issue_cols_details = []
    text_issue_cols_names = []
    sample_size = min(500, num_rows)
    if sample_size > 0:
        actual_sample_size = min(sample_size, num_rows)
        if actual_sample_size > 0:
            df_sample = df.sample(actual_sample_size, random_state=1)
            for col in object_cols:
                if col not in df_sample.columns or df_sample[col].isnull().all(): continue
                try:
                    col_str = df_sample[col].astype('string').dropna()
                    if col_str.empty: continue
                    has_whitespace = (col_str != col_str.str.strip()).any()
                    unique_vals = col_str.unique()
                    has_mixed_case = False
                    if len(unique_vals) > 1:
                         if col_str.str.lower().nunique() < len(unique_vals): has_mixed_case = True
                    if has_whitespace or has_mixed_case:
                        issues = []
                        if has_whitespace: issues.append("whitespace")
                        if has_mixed_case: issues.append("casing")
                        text_issue_cols_details.append(f"'{col}': Contains {'/'.join(issues)} (based on sample)")
                        text_issue_cols_names.append(col)
                except Exception as e: print(f"Text check failed for column '{col}': {e}")
    if text_issue_cols_details:
         affected_cols_str = format_cols_for_suggestion(text_issue_cols_names)
         suggestion = f"Use '7. Case & Whitespace'{affected_cols_str} (e.g., Action: 'Strip Leading/Trailing Whitespace' and 'Convert to Lowercase')."
         findings.append({
            'issue_type': 'Text Formatting',
            'severity': 'Low',
            'message': f"Potential text formatting issues found in {len(text_issue_cols_names)} column(s).",
            'details': text_issue_cols_details,
            'suggestion': suggestion,
            'affected_columns': text_issue_cols_names
        })

    # Finding 8: High Cardinality Text Columns
    high_cardinality_cols = []
    if num_rows > 0:
        for col in object_cols:
            try:
                unique_count = df[col].nunique()
                is_high_card = unique_count > 50 and (unique_count / num_rows) > 0.10
                is_likely_id = unique_count > num_rows * 0.9
                if is_high_card and not is_likely_id:
                    high_cardinality_cols.append(f"'{col}': {unique_count} unique values")
            except Exception as e: print(f"Cardinality check failed for column '{col}': {e}")
    if high_cardinality_cols:
         suggestion = "Review if these need cleaning (e.g., '10. Fuzzy Match Text Values') or are identifiers/free text."
         findings.append({
            'issue_type': 'High Cardinality Text',
            'severity': 'Info',
            'message': f"Found {len(high_cardinality_cols)} text column(s) with many unique values (excluding likely IDs).",
            'details': high_cardinality_cols,
            'suggestion': suggestion
        })

    return findings
# --- END auto_explore_data ---


# --- generate_profile function remains the same ---
def generate_profile(df):
    # (Keep the existing generate_profile logic)
    """Generates detailed statistics for each column."""
    profile = {}
    if df is None or df.empty:
        return {"error": "DataFrame is empty or not loaded."}
    total_rows = len(df)
    for col in df.columns:
        column_data = df[col]
        stats = {}
        stats['dtype'] = str(column_data.dtype)
        stats['count'] = int(column_data.count())
        stats['missing_count'] = int(column_data.isnull().sum())
        stats['missing_percent'] = round((stats['missing_count'] / total_rows) * 100, 2) if total_rows > 0 else 0
        stats['unique_count'] = int(column_data.nunique())
        stats['unique_percent'] = round((stats['unique_count'] / total_rows) * 100, 2) if total_rows > 0 else 0

        if is_numeric_dtype(column_data):
            stats['type'] = 'Numeric'
            desc = column_data.describe(percentiles=[.25, .5, .75])
            stats['mean'] = round(desc.get('mean', np.nan), 4)
            stats['std'] = round(desc.get('std', np.nan), 4)
            stats['min'] = round(desc.get('min', np.nan), 4)
            stats['25%'] = round(desc.get('25%', np.nan), 4)
            stats['50%'] = round(desc.get('50%', np.nan), 4)
            stats['75%'] = round(desc.get('75%', np.nan), 4)
            stats['max'] = round(desc.get('max', np.nan), 4)
            try:
                stats['skewness'] = round(column_data.skew(skipna=True), 4)
                stats['kurtosis'] = round(column_data.kurt(skipna=True), 4)
            except Exception: stats['skewness'] = stats['kurtosis'] = np.nan
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
        elif is_string_dtype(column_data) or is_object_dtype(column_data):
            stats['type'] = 'Text/Object'
            try:
                value_counts = column_data.value_counts().head(5)
                stats['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
            except Exception as e: stats['top_values'] = {"Error": f"Could not get counts ({type(e).__name__})"}
            try:
                str_series = column_data.astype('string').dropna()
                if not str_series.empty:
                    str_lengths = str_series.str.len()
                    stats['min_length'] = int(str_lengths.min())
                    stats['mean_length'] = round(str_lengths.mean(), 2)
                    stats['max_length'] = int(str_lengths.max())
                else: stats['min_length'] = stats['mean_length'] = stats['max_length'] = 0
            except Exception: stats['min_length'] = stats['mean_length'] = stats['max_length'] = 'Error'
        else:
            stats['type'] = 'Other'
        profile[col] = stats
    return profile
# --- END generate_profile ---


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the initial upload/paste page."""
    # Clear potentially large session items
    keys_to_clear = ['current_file_path', 'original_filename', 'source_format',
                     'source_encoding_info', 'exploration_results',
                     'profile_results', 'fuzzy_results']
    for key in keys_to_clear:
        session.pop(key, None)
    return render_template('index.html')


# --- Route for Initial Data Load (Paste) ---
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles pasted text input, loads data, saves to Supabase, redirects."""
    if not supabase: # Check if client initialized
         flash("Cloud storage client not available. Cannot process data.", "danger")
         return redirect(url_for('index'))

    csv_text = request.form.get('csv_data')
    if not csv_text or not csv_text.strip():
        flash('No data pasted into the text area.', 'warning')
        return redirect(url_for('index'))

    filename_base = "pasted_data"
    file_extension = 'csv'
    user_encoding = request.form.get('encoding') or 'utf-8' # Default to utf-8

    # Use load_data helper (which now assumes text is CSV)
    df, source_info = load_data(text_data=csv_text,
                                file_extension=file_extension, # Pass 'csv'
                                encoding=user_encoding)
    if df is None:
        # load_data should have flashed an error message
        return redirect(url_for('index'))

    # Save the initial loaded state to Supabase
    object_path = save_data_to_supabase(df, filename_base)
    if object_path is None:
        # save_data_to_supabase flashes error
        return redirect(url_for('index'))

    # Store Supabase object path and other info in session
    session['current_file_path'] = object_path # Store Supabase path now
    session['original_filename'] = f"{filename_base}.csv"
    session['source_format'] = file_extension
    session['source_encoding_info'] = source_info # Store encoding used

    # Clear analysis results from any previous run
    session.pop('exploration_results', None)
    session.pop('profile_results', None)
    session.pop('fuzzy_results', None)

    flash(f"Pasted data processed successfully. Ready for cleaning.", "success")
    return redirect(url_for('clean_data'))


# --- Analysis Routes (Explore, Profile, Fuzzy) ---
# These now use read_data_from_supabase
@app.route('/explore', methods=['POST'])
def explore_data():
    object_path = session.get('current_file_path')
    if not object_path: flash("No data loaded.", "warning"); return redirect(url_for('index'))
    df = read_data_from_supabase(object_path)
    if df is None: return redirect(url_for('clean_data')) # read helper flashes error
    try:
        start_time = time.time()
        flash("Running Auto Explore analysis...", "info")
        exploration_results = auto_explore_data(df)
        session['exploration_results'] = exploration_results; session.pop('profile_results', None)
        duration = time.time() - start_time
        flash(f"Auto Explore completed ({duration:.2f}s). Found {len(exploration_results)} potential areas.", "success")
    except Exception as e: flash(f"Error during Auto Explore: {e}", "danger"); session.pop('exploration_results', None); traceback.print_exc()
    return redirect(url_for('clean_data'))

@app.route('/profile', methods=['POST'])
def profile_data():
    object_path = session.get('current_file_path')
    if not object_path: flash("No data loaded.", "warning"); return redirect(url_for('index'))
    df = read_data_from_supabase(object_path)
    if df is None: return redirect(url_for('clean_data'))
    try:
        start_time = time.time()
        flash("Generating Data Profile...", "info")
        profile_results = generate_profile(df)
        session['profile_results'] = profile_results; session.pop('exploration_results', None)
        duration = time.time() - start_time
        flash(f"Data Profile generated ({duration:.2f}s).", "success")
    except Exception as e: flash(f"Error during profiling: {e}", "danger"); session.pop('profile_results', None); traceback.print_exc()
    return redirect(url_for('clean_data'))

# --- Fuzzy Analysis Route (Corrected) ---
@app.route('/fuzzy_analyze', methods=['POST'])
def fuzzy_analyze():
    """Analyzes a column to find potentially similar groups using fuzzy matching."""
    object_path = session.get('current_file_path')
    if not object_path:
        flash("No data loaded to analyze.", "warning")
        return redirect(url_for('index')) # Redirect home if no data

    # Get and validate inputs
    col = request.form.get('fuzzy_column')
    try:
        threshold = int(request.form.get('fuzzy_threshold', 85)) # Default 85
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100.")
    except (ValueError, TypeError):
        flash("Invalid similarity threshold (must be 0-100 integer).", "danger")
        return redirect(url_for('clean_data')) # Redirect back to clean page

    if not col:
         flash("Please select a column to analyze for fuzzy matching.", "warning")
         return redirect(url_for('clean_data'))

    # Read data
    df = read_data_from_supabase(object_path)
    if df is None:
        # read_data_from_supabase flashes error
        return redirect(url_for('clean_data')) # Go back if read fails

    # Validate column exists
    if col not in df.columns:
        flash(f"Column '{col}' not found in the current dataset.", "warning")
        return redirect(url_for('clean_data'))

    try:
        start_time = time.time()
        # Ensure column is treated as string and get unique non-null values
        unique_vals = df[col].dropna().astype(str).unique()

        if len(unique_vals) < 2:
            flash(f"Not enough unique text values in column '{col}' to perform fuzzy matching.", "info")
            session.pop('fuzzy_results', None) # Clear any previous results
            return redirect(url_for('clean_data'))

        flash(f"Analyzing column '{col}' for values with similarity >= {threshold}% (this may take time for many unique values)...", "info")

        groups = []
        processed_indices = set() # Use indices for faster lookup
        unique_vals_list = sorted(list(unique_vals)) # Sort for minor efficiency gain & consistent order
        n_unique = len(unique_vals_list)

        # O(n^2) comparison - performance bottleneck for high cardinality
        for i in range(n_unique):
            if i in processed_indices:
                continue

            # ***** CORRECTED INITIALIZATION *****
            # Initialize current_group for *this specific starting value*
            current_group = [unique_vals_list[i]]
            processed_indices.add(i)
            # ************************************

            for j in range(i + 1, n_unique):
                if j in processed_indices:
                    continue

                # Ensure comparison is between strings
                val_i = str(unique_vals_list[i])
                val_j = str(unique_vals_list[j])

                # Calculate similarity ratio
                ratio = fuzz.ratio(val_i, val_j)

                if ratio >= threshold:
                    # Append the *original* value from the list
                    current_group.append(unique_vals_list[j])
                    processed_indices.add(j)

            # Only store groups with more than one member (i.e., matches found)
            if len(current_group) > 1:
                # Sort group members alphabetically for consistent display
                groups.append(sorted(current_group))

        # Process results
        duration = time.time() - start_time
        if groups:
            session['fuzzy_results'] = {'column': col, 'threshold': threshold, 'groups': groups}
            flash(f"Fuzzy analysis completed ({duration:.2f}s). Found {len(groups)} potential groups. Review below.", "success")
        else:
            session.pop('fuzzy_results', None) # Clear if no groups found
            flash(f"Fuzzy analysis completed ({duration:.2f}s). No groups found meeting the {threshold}% similarity threshold.", "info")

    except Exception as e:
        flash(f"An error occurred during fuzzy analysis: {type(e).__name__} - {e}", "danger")
        session.pop('fuzzy_results', None) # Clear results on error
        traceback.print_exc() # Log detailed error

    return redirect(url_for('clean_data')) # Redirect back to show results/messages


# --- Route to Display Data and Cleaning Options ---
@app.route('/clean')
def clean_data():
    """Displays the data preview and cleaning options, reading from Supabase."""
    object_path = session.get('current_file_path')
    if not object_path:
        flash("No data loaded. Please paste data on the home page.", "warning")
        session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None)
        return redirect(url_for('index'))

    # Read data using the helper
    df = read_data_from_supabase(object_path)

    if df is None:
        # Helper function flashed the error, maybe redirect or just show empty page
        # Redirecting might be better if the file is truly gone/unreadable
        flash("Failed to load data for display. Session might be invalid.", "warning")
        return redirect(url_for('index'))

    try:
        columns = df.columns.tolist()
        preview_rows = 100
        actual_rows_shown = min(preview_rows, len(df))
        # Use pandas Styler for better HTML rendering options if needed later
        df_preview_html = df.head(preview_rows).to_html(
            classes='table table-hover table-sm',
            border=0, index=False,
            na_rep='<span class="text-muted fst-italic">NaN</span>' # Render NaN nicely
        )
        preview_text = f"Data Preview ({actual_rows_shown} of {len(df)} Rows)"

        exploration_results = session.get('exploration_results')
        profile_results = session.get('profile_results')
        fuzzy_results = session.get('fuzzy_results')

        return render_template('clean.html',
                               columns=columns,
                               df_preview=df_preview_html,
                               preview_text=preview_text,
                               exploration_results=exploration_results,
                               profile_results=profile_results,
                               fuzzy_results=fuzzy_results)
    except Exception as e:
        flash(f"Error preparing data display: {str(e)}", "danger")
        traceback.print_exc()
        # Attempt to redirect back gracefully
        return redirect(url_for('index'))


# --- Manual Cleaning Route ---
@app.route('/apply/<operation>', methods=['POST'])
def apply_cleaning(operation):
    """Applies the selected manual cleaning operation, reads/saves via Supabase."""
    object_path = session.get('current_file_path')
    original_filename = session.get('original_filename', 'data.csv')
    original_filename_base = os.path.splitext(original_filename)[0]

    # Clear analysis results from session
    session.pop('exploration_results', None)
    session.pop('profile_results', None)
    if operation != 'fuzzy_apply': session.pop('fuzzy_results', None) # Keep fuzzy results if applying them

    if not object_path:
        flash("Session expired or data path missing. Please start over.", "warning")
        return redirect(url_for('index'))

    # Read current data state from Supabase
    df = read_data_from_supabase(object_path)
    if df is None:
        # read_data_from_supabase already flashed an error
        return redirect(url_for('clean_data')) # Redirect back to clean page

    try:
        original_shape = df.shape
        cols_affected = [] # Track columns directly modified by value changes
        specific_op_success_msg = None # Track if op logic flashed success

        # ===============================================
        # --- Apply Manual Operations (Logic Unchanged) ---
        # (Paste the entire block for operations 'missing' through 'remove' here)
        # The internal logic remains the same, operating on the 'df' DataFrame.
        # ===============================================
        if operation == 'missing':
            cols = request.form.getlist('columns')
            method = request.form.get('missing_method')
            fill_val = request.form.get('fill_value')
            target_cols = cols if cols else df.columns.tolist()
            if method == 'drop_row':
                subset_param = target_cols if cols else None
                initial_rows = df.shape[0]
                df.dropna(subset=subset_param, inplace=True)
                rows_removed = initial_rows - df.shape[0]
                if rows_removed > 0: specific_op_success_msg = f"Removed {rows_removed} rows with missing values."
                else: specific_op_success_msg = ("No rows with missing values found based on selection.", "info")
            elif method == 'drop_col':
                cols_to_drop = [col for col in target_cols if col in df.columns and df[col].isnull().any()]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    cols_affected = cols_to_drop
                    specific_op_success_msg = f"Dropped columns with missing values: {', '.join(cols_to_drop)}"
                else: specific_op_success_msg = ("No columns found with missing values.", "info")
            else: # Fill methods
                applied_fill_cols = []
                for col in target_cols:
                    if col in df.columns and df[col].isnull().any():
                        original_dtype = df[col].dtype; fill_applied_to_col = False
                        try:
                            if method == 'fill_mean':
                                if is_numeric_dtype(df[col]):
                                    mean_val = df[col].mean()
                                    if pd.notna(mean_val): df[col].fillna(mean_val, inplace=True); fill_applied_to_col = True
                                    else: flash(f"Cannot calculate mean for '{col}' (all NaN?).", "warning")
                                else: flash(f"'{col}' is not numeric. Cannot fill with mean.", "warning")
                            elif method == 'fill_median':
                                if is_numeric_dtype(df[col]):
                                    median_val = df[col].median()
                                    if pd.notna(median_val): df[col].fillna(median_val, inplace=True); fill_applied_to_col = True
                                    else: flash(f"Cannot calculate median for '{col}' (all NaN?).", "warning")
                                else: flash(f"'{col}' is not numeric. Cannot fill with median.", "warning")
                            elif method == 'fill_mode':
                                mode_val = df[col].mode();
                                if not mode_val.empty: df[col].fillna(mode_val[0], inplace=True); fill_applied_to_col = True
                                else: flash(f"Cannot determine mode for '{col}'.", "warning")
                            elif method == 'fill_value':
                                if fill_val is None: flash("Fill value required.", "warning"); continue
                                try: typed_fill_val = pd.Series([fill_val]).astype(original_dtype)[0]; df[col].fillna(typed_fill_val, inplace=True)
                                except Exception: df[col].fillna(str(fill_val), inplace=True); flash(f"Filled '{col}' with '{fill_val}' as string (type conversion failed).", "info")
                                fill_applied_to_col = True
                            if fill_applied_to_col: applied_fill_cols.append(col); cols_affected.append(col)
                        except Exception as e: flash(f"Error filling column '{col}': {e}", "danger")
                if applied_fill_cols: specific_op_success_msg = f"Applied fill '{method}' to columns: {', '.join(list(set(applied_fill_cols)))}"
                elif method not in ['drop_row', 'drop_col']: specific_op_success_msg = ("No missing values found to fill, or fill failed.", "info")
        elif operation == 'outlier':
            col = request.form.get('column'); method = request.form.get('outlier_method'); threshold_str = request.form.get('threshold', ''); action = request.form.get('action')
            if not col or col not in df.columns: specific_op_success_msg = (f"Column '{col}' not found.", "warning")
            elif not is_numeric_dtype(df[col]): specific_op_success_msg = (f"'{col}' is not numeric.", "warning")
            else:
                numeric_col_data = df[col].dropna()
                if numeric_col_data.empty or numeric_col_data.nunique() < 2: specific_op_success_msg = (f"Not enough valid numeric data in '{col}'.", "info")
                else:
                    lower_bound, upper_bound, param_value = None, None, None; outlier_indices = pd.Index([])
                    try: # Parameter Parsing
                        if method in ['iqr', 'zscore', 'modified_zscore']: param_value = float(threshold_str); assert param_value > 0
                        elif method == 'percentile': param_value = float(threshold_str); assert 0 < param_value < 50
                        elif method == 'isolation_forest': param_value = 'auto' if threshold_str.lower() == 'auto' else float(threshold_str); assert param_value=='auto' or 0.0 < param_value <= 0.5
                        else: raise ValueError(f"Invalid method: {method}")
                    except Exception as e: specific_op_success_msg = (f"Invalid parameter '{threshold_str}' for '{method}': {e}", "danger"); method=None
                    if method: # Outlier Detection
                        try:
                            if method == 'iqr':
                                Q1, Q3 = numeric_col_data.quantile([0.25, 0.75]); IQR = Q3 - Q1
                                if pd.notna(IQR) and IQR > 0: lower_bound, upper_bound = Q1 - param_value * IQR, Q3 + param_value * IQR; outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                                else: specific_op_success_msg = (f"IQR zero/NaN for '{col}'.", "info"); outlier_mask = pd.Series(False, index=df.index)
                            elif method == 'zscore':
                                mean, std = numeric_col_data.mean(), numeric_col_data.std()
                                if pd.notna(std) and std > 0: z_scores = np.abs((df[col] - mean) / std); lower_bound, upper_bound = mean - param_value*std, mean + param_value*std; outlier_mask = z_scores > param_value
                                else: specific_op_success_msg = (f"Std Dev zero/NaN for '{col}'.", "info"); outlier_mask = pd.Series(False, index=df.index)
                            elif method == 'percentile':
                                lower_p, upper_p = param_value / 100.0, 1.0 - (param_value / 100.0); lower_bound, upper_bound = numeric_col_data.quantile([lower_p, upper_p])
                                if pd.notna(lower_bound) and pd.notna(upper_bound): outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound); flash(f"Using bounds: < {lower_bound:.4f} / > {upper_bound:.4f}.", "info")
                                else: specific_op_success_msg = (f"Cannot get percentile bounds for '{col}'.", "warning"); outlier_mask = pd.Series(False, index=df.index)
                            elif method == 'modified_zscore':
                                median = numeric_col_data.median(); mad = median_abs_deviation(numeric_col_data, nan_policy='omit', scale='normal')
                                if pd.notna(mad) and mad > 0: mod_z = 0.6745 * (df[col] - median) / mad; lower_bound, upper_bound = median - param_value*mad/0.6745, median + param_value*mad/0.6745; outlier_mask = np.abs(mod_z) > param_value
                                else: specific_op_success_msg = (f"MAD zero/NaN for '{col}'.", "info"); outlier_mask = pd.Series(False, index=df.index)
                            elif method == 'isolation_forest':
                                data_for_iforest = numeric_col_data.values.reshape(-1, 1); iforest = IsolationForest(contamination=param_value, random_state=42); iforest.fit(data_for_iforest)
                                predictions = iforest.predict(df.loc[numeric_col_data.index, [col]].values); outlier_indices_if = numeric_col_data.index[predictions == -1]; lower_bound, upper_bound = None, None
                                if action == 'cap': flash("Capping not recommended for Isolation Forest.", "warning")
                                outlier_mask = df.index.isin(outlier_indices_if)
                            outlier_indices = df.index[outlier_mask & df[col].notna()]
                        except Exception as detect_err: specific_op_success_msg = (f"Error during outlier detection: {detect_err}", "danger"); outlier_indices = pd.Index([])
                        # Apply Action
                        num_outliers = len(outlier_indices)
                        if num_outliers == 0 and specific_op_success_msg is None: specific_op_success_msg = (f"No outliers detected in '{col}' using {method}.", "info")
                        elif action == 'remove' and num_outliers > 0: df.drop(index=outlier_indices, inplace=True); specific_op_success_msg = f"Removed {num_outliers} outliers from '{col}'."
                        elif action == 'cap' and num_outliers > 0:
                            if lower_bound is not None and upper_bound is not None: cap_mask = df.index.isin(outlier_indices); capped_count = cap_mask.sum(); df.loc[cap_mask, col] = df.loc[cap_mask, col].clip(lower=lower_bound, upper=upper_bound); cols_affected.append(col); specific_op_success_msg = f"Capped {capped_count} outliers in '{col}'."
                            elif method == 'isolation_forest': specific_op_success_msg = ("Capping skipped for Isolation Forest.", "warning")
                            else: specific_op_success_msg = ("Cannot cap, bounds not determined.", "warning")
                        elif action not in ['remove', 'cap']: specific_op_success_msg = ("Invalid outlier action.", "danger")
        elif operation == 'smooth':
            col = request.form.get('column'); method = request.form.get('smooth_method'); window = None
            try: window = int(request.form.get('window', 3)); assert window >= 2
            except Exception as e: specific_op_success_msg = (f"Invalid window size: {e}", "danger")
            if window:
                if not col or col not in df.columns: specific_op_success_msg = (f"Column '{col}' not found.", "warning")
                elif not is_numeric_dtype(df[col]): specific_op_success_msg = (f"'{col}' is not numeric.", "warning")
                else:
                    original_series = df[col].copy()
                    if method == 'moving_average': df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean(); cols_affected.append(col); specific_op_success_msg = f"Applied Moving Average (win={window}) to '{col}'."
                    else: specific_op_success_msg = (f"Invalid smooth method: {method}.", "danger")
                    if cols_affected and df[col].equals(original_series): specific_op_success_msg = (f"Smoothing applied to '{col}', but values unchanged.", "info")
        elif operation == 'normalize':
            cols = request.form.getlist('columns'); method = request.form.get('normalize_method'); applied_norm_cols = []
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist(); target_cols = [c for c in (cols if cols else numeric_cols) if c in numeric_cols]
            if not target_cols: specific_op_success_msg = ("No numeric columns selected/found.", "warning")
            else:
                for col in target_cols:
                    data_to_scale = df[[col]].dropna()
                    if data_to_scale.empty or data_to_scale.nunique()[0] < 2: flash(f"Skipping scaling for '{col}'.", "info"); continue
                    try:
                        if method == 'min_max': scaler = MinMaxScaler()
                        elif method == 'z_score': scaler = StandardScaler()
                        else: flash(f"Invalid scaling method: {method}.", "danger"); continue
                        scaled_data = scaler.fit_transform(data_to_scale); df.loc[data_to_scale.index, col] = scaled_data.flatten(); cols_affected.append(col); applied_norm_cols.append(col)
                    except Exception as e: flash(f"Error scaling '{col}': {e}", "danger")
                if applied_norm_cols: specific_op_success_msg = f"Applied {method} scaling to: {', '.join(applied_norm_cols)}"
                elif target_cols: specific_op_success_msg = ("Scaling not applied.", "info")
        elif operation == 'deduplicate':
            cols = request.form.getlist('columns'); keep = request.form.get('keep')
            if keep not in ['first', 'last', 'none']: specific_op_success_msg = ("Invalid 'keep' parameter.", "danger")
            else:
                keep_param = keep if keep != 'none' else False; subset_param = cols if cols else None; initial_rows = df.shape[0]
                df.drop_duplicates(subset=subset_param, keep=keep_param, inplace=True); rows_removed = initial_rows - df.shape[0]
                if rows_removed > 0: specific_op_success_msg = f"Removed {rows_removed} duplicate rows."
                else: specific_op_success_msg = ("No duplicate rows found.", "info")
        elif operation == 'date':
            cols = request.form.getlist('columns'); output_format = request.form.get('date_format') or None; errors_coerce = request.form.get('errors_coerce') == 'true'; converted_cols, formatted_cols = [], []
            if not cols: specific_op_success_msg = ("Select columns for date standardization.", "warning")
            else:
                for col in cols:
                    if col not in df.columns: flash(f"Column '{col}' not found.", "warning"); continue
                    original_series = df[col].copy()
                    try:
                        errors_param = 'coerce' if errors_coerce else 'raise'; converted_series = pd.to_datetime(df[col], errors=errors_param, infer_datetime_format=True)
                        if not converted_series.equals(original_series) or converted_series.dtype != original_series.dtype: df[col] = converted_series; cols_affected.append(col); converted_cols.append(col)
                        if output_format and is_datetime64_any_dtype(df[col]):
                            try: is_nat = df[col].isna(); df.loc[~is_nat, col] = df.loc[~is_nat, col].dt.strftime(output_format); formatted_cols.append(col); cols_affected.append(col)
                            except Exception as fmt_err: flash(f"Invalid date format '{output_format}' for '{col}': {fmt_err}", "danger")
                    except ValueError as e: flash(f"Error parsing date in '{col}': {e}. Try coercion.", "danger")
                    except Exception as e: flash(f"Unexpected date error for '{col}': {e}", "danger")
                msg_parts = []
                if converted_cols: msg_parts.append(f"Converted/Validated: {', '.join(list(set(converted_cols)))}")
                if formatted_cols: msg_parts.append(f"Formatted: {', '.join(list(set(formatted_cols)))}")
                if msg_parts: specific_op_success_msg = ". ".join(msg_parts)
                elif cols: specific_op_success_msg = ("Date op applied, but no columns changed.", "info")
        elif operation == 'case':
            cols = request.form.getlist('columns'); case_type = request.form.get('case_type'); applied_case_cols = []
            if not cols: specific_op_success_msg = ("Select columns.", "warning")
            else:
                for col in cols:
                    if col not in df.columns: flash(f"'{col}' not found.", "warning"); continue
                    if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                        original_series = df[col].copy(); df[col] = df[col].astype("string")
                        if case_type == 'lower': df[col] = df[col].str.lower()
                        elif case_type == 'upper': df[col] = df[col].str.upper()
                        elif case_type == 'title': df[col] = df[col].str.title()
                        elif case_type == 'strip': df[col] = df[col].str.strip()
                        else: flash(f"Invalid case type: {case_type}.", "danger"); continue
                        if not df[col].equals(original_series): applied_case_cols.append(col); cols_affected.append(col)
                    else: flash(f"'{col}' not text type.", "warning")
                if applied_case_cols: specific_op_success_msg = f"Applied '{case_type}' to: {', '.join(list(set(applied_case_cols)))}"
                elif cols: specific_op_success_msg = ("Case/whitespace op applied, no changes.", "info")
        elif operation == 'convert_type':
            col = request.form.get('column'); target_type = request.form.get('target_type'); converted_series = None; conversion_error = None
            if not col or col not in df.columns: specific_op_success_msg = (f"Column '{col}' not found/selected.", "warning")
            elif not target_type: specific_op_success_msg = ("Select target data type.", "warning")
            else:
                original_series = df[col].copy(); original_nan_count = original_series.isnull().sum()
                try:
                    if target_type == 'string': converted_series = df[col].astype('string')
                    elif target_type == 'Int64': numeric_temp = pd.to_numeric(df[col], errors='coerce'); assert not (numeric_temp.notna() & (numeric_temp % 1 != 0)).any(), "Non-integers found"; converted_series = numeric_temp.astype('Int64')
                    elif target_type == 'float64': converted_series = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    elif target_type == 'boolean': map_dict = {'true': True,'false': False,'1': True,'0': False,'yes': True,'no': False,'t': True,'f': False,'y': True,'n': False}; str_series = df[col].astype(str).str.lower().str.strip(); mapped_series = str_series.replace(r'^(?!(true|false|1|0|yes|no|t|f|y|n)$).*$', np.nan, regex=True).map(map_dict); converted_series = mapped_series.astype('boolean')
                    elif target_type == 'datetime64[ns]': converted_series = pd.to_datetime(df[col], errors='coerce')
                    else: conversion_error = f"Unsupported target type: {target_type}"
                except AssertionError as e: conversion_error = f"Cannot convert '{col}' to Int64: {e}"
                except Exception as e: conversion_error = f"Error converting '{col}' to {target_type}: {e}"

                if conversion_error: specific_op_success_msg = (conversion_error, "danger")
                elif converted_series is not None:
                    if not converted_series.equals(original_series) or converted_series.dtype != original_series.dtype:
                        df[col] = converted_series; cols_affected.append(col); final_nan_count = df[col].isnull().sum(); coerced_count = max(0, final_nan_count - original_nan_count)
                        if coerced_count > 0: specific_op_success_msg = (f"Converted '{col}' to {target_type}. {coerced_count} value(s) became missing.", "warning")
                        else: specific_op_success_msg = f"Successfully converted '{col}' to {target_type}."
                    else: specific_op_success_msg = (f"'{col}' already compatible with {target_type}.", "info")
        elif operation == 'regex_replace':
            col = request.form.get('regex_column'); pattern = request.form.get('regex_pattern'); replacement = request.form.get('regex_replacement', '')
            if not col or col not in df.columns: specific_op_success_msg = (f"Column '{col}' not found.", "warning")
            elif pattern is None: specific_op_success_msg = ("Regex pattern required.", "warning")
            else:
                if not is_object_dtype(df[col]) and not is_string_dtype(df[col]): flash(f"Converting '{col}' to string for Regex.", "info"); df[col] = df[col].astype('string')
                try:
                    original_series = df[col].copy(); df[col] = df[col].astype('string').str.replace(pattern, replacement, regex=True)
                    if not df[col].equals(original_series): cols_affected.append(col); specific_op_success_msg = f"Applied Regex replace on '{col}'."
                    elif original_series.astype('string').str.contains(pattern, regex=True, na=False).any(): specific_op_success_msg = ("Regex found, but no change after replace.", "info")
                    else: specific_op_success_msg = ("Regex pattern not found.", "info")
                except re.error as regex_err: specific_op_success_msg = (f"Invalid Regex: {regex_err}", "danger")
                except Exception as e: specific_op_success_msg = (f"Error during Regex replace: {e}", "danger")
        elif operation == 'fuzzy_apply':
            fuzzy_results = session.get('fuzzy_results')
            if not fuzzy_results or 'column' not in fuzzy_results or 'groups' not in fuzzy_results: specific_op_success_msg = ("No fuzzy results in session.", "warning")
            else:
                col = fuzzy_results['column']; groups = fuzzy_results['groups']
                if col not in df.columns: specific_op_success_msg = (f"Fuzzy column '{col}' not found.", "warning"); session.pop('fuzzy_results', None)
                else:
                    replacement_map = {}; total_replacements_made = 0
                    try:
                        df[col] = df[col].astype('string'); original_series = df[col].copy()
                        for i, group in enumerate(groups):
                            canon = request.form.get(f'canonical_value_{i}')
                            if canon is not None and canon in group:
                                for orig in group:
                                    if orig != canon and orig in original_series.unique(): replacement_map[orig] = canon
                        if replacement_map: df[col].replace(replacement_map, inplace=True); changes = (df[col] != original_series) & original_series.notna(); total_replacements_made = changes.sum()
                        if total_replacements_made > 0: cols_affected.append(col); specific_op_success_msg = f"Applied {total_replacements_made} fuzzy replacements in '{col}'."
                        elif replacement_map: specific_op_success_msg = ("Selected values resulted in no changes.", "info")
                        else: specific_op_success_msg = ("No valid changes selected.", "info")
                    except Exception as e: specific_op_success_msg = (f"Error applying fuzzy changes: {e}", "danger")
                    session.pop('fuzzy_results', None)
        elif operation == 'constraint':
            col = request.form.get('column'); min_s = request.form.get('min_val','').strip(); max_s = request.form.get('max_val','').strip(); numeric_col = None
            if not col or col not in df.columns: specific_op_success_msg = (f"Column '{col}' not found.", "warning")
            elif not min_s and not max_s: specific_op_success_msg = ("Provide min or max.", "warning")
            else:
                try: numeric_col = pd.to_numeric(df[col], errors='coerce'); assert not numeric_col.isnull().all()
                except Exception: specific_op_success_msg = (f"'{col}' has no valid numeric data.", "danger")
                if numeric_col is not None:
                    cond = pd.Series(True, index=df.index); valid = True
                    try:
                        if min_s: cond &= (numeric_col >= float(min_s))
                        if max_s: cond &= (numeric_col <= float(max_s))
                    except ValueError: specific_op_success_msg = ("Invalid number for min/max.", "danger"); valid = False
                    if valid: cond &= numeric_col.notna(); initial = df.shape[0]; df = df[cond].copy(); removed = initial - df.shape[0]
                    if removed > 0: specific_op_success_msg = f"Numeric filter on '{col}' removed {removed} rows."
                    elif valid: specific_op_success_msg = (f"Numeric filter on '{col}' removed no rows.", "info")
        elif operation == 'sort':
            cols = request.form.getlist('columns'); ascending = request.form.get('ascending', 'True') == 'True'
            valid_cols = [c for c in cols if c in df.columns]; invalid_cols = [c for c in cols if c not in df.columns]
            if invalid_cols: flash(f"Cols not found: {', '.join(invalid_cols)}", "warning")
            if not valid_cols: specific_op_success_msg = ("Select valid cols.", "warning")
            else:
                try: df.sort_values(by=valid_cols, ascending=ascending, inplace=True, ignore_index=True, na_position='last'); specific_op_success_msg = f"Sorted by: {', '.join(valid_cols)} ({'Asc' if ascending else 'Desc'})."
                except Exception as e: specific_op_success_msg = (f"Error sorting: {e}", "danger")
        elif operation == 'rename':
            old = request.form.get('old_name'); new = request.form.get('new_name', '').strip()
            if not old or old not in df.columns: specific_op_success_msg = (f"Original col '{old}' not found.", "warning")
            elif not new: specific_op_success_msg = ("New name empty.", "warning")
            elif new in df.columns and new != old: specific_op_success_msg = (f"'{new}' already exists.", "warning")
            elif not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new): specific_op_success_msg = (f"Invalid new name '{new}'.", "warning")
            else: df.rename(columns={old: new}, inplace=True); specific_op_success_msg = f"Renamed '{old}' to '{new}'."
        elif operation == 'remove':
            cols = request.form.getlist('columns'); exist = [c for c in cols if c in df.columns]; not_exist = [c for c in cols if c not in df.columns]
            if not_exist: flash(f"Cols not found: {', '.join(not_exist)}", "warning")
            if exist: df.drop(columns=exist, inplace=True); specific_op_success_msg = f"Removed columns: {', '.join(exist)}"
            elif not not_exist: specific_op_success_msg = ("No columns selected.", "info")
        else:
            flash(f"Unknown operation: {operation}", "danger")
            return redirect(url_for('clean_data'))

        # --- Save state to Supabase & Final Flash ---
        # Save the modified DataFrame state back to Supabase
        new_object_path = save_data_to_supabase(df, original_filename_base)

        if new_object_path:
            # **Important:** Remove the *previous* object from Supabase to avoid clutter
            # This requires careful error handling in case removal fails
            if object_path != new_object_path: # Check if path actually changed
                try:
                    print(f"Attempting to remove old Supabase object: {object_path}")
                    # Ensure supabase client is available
                    if supabase:
                         supabase.storage.from_(BUCKET_NAME).remove([object_path]) # remove expects a list
                         print(f"Successfully removed old Supabase object: {object_path}")
                    else:
                         print("Skipping removal of old object: Supabase client not available.")
                except Exception as remove_err:
                    # Log error but don't necessarily fail the whole request
                    print(f"WARNING: Failed to remove previous Supabase object '{object_path}': {remove_err}")
                    flash(f"Warning: Could not remove old intermediate data file '{os.path.basename(object_path)}' from cloud storage.", "warning")

            # Update session with the NEW Supabase object path
            session['current_file_path'] = new_object_path
            final_shape = df.shape

            # Flash specific message from operation logic if available
            if specific_op_success_msg:
                 if isinstance(specific_op_success_msg, tuple): flash(specific_op_success_msg[0], specific_op_success_msg[1])
                 else: flash(specific_op_success_msg, "success")
            else: # Generate generic message
                 rows_ch = original_shape[0] - final_shape[0]; cols_ch = original_shape[1] - final_shape[1]; msg = ""
                 if rows_ch > 0: msg += f" {rows_ch} rows removed."
                 if cols_ch > 0: msg += f" {cols_ch} columns removed."
                 unique_aff = sorted(list(set(cols_affected)))
                 if not msg and unique_aff: msg = f" Values potentially modified in: {', '.join(unique_aff)}."
                 if not msg and not unique_aff: msg = " Operation applied, no apparent changes."
                 flash(f"Operation '{operation}' applied.{msg}", "success")
        else:
            # save_data_to_supabase failed and should have flashed an error
            flash("Failed to save changes to cloud storage. Modifications not persisted.", "danger")

        return redirect(url_for('clean_data'))

    # --- Consolidated Error Handling for /apply ---
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error during '/apply/{operation}': {error_type} - {error_msg}")
        traceback.print_exc()
        user_msg = f"Unexpected error during '{operation}': {error_type}. Check inputs or data state."
        # Add more specific user messages based on error types if needed
        if isinstance(e, KeyError): user_msg = f"Error applying '{operation}': Missing column/parameter '{error_msg}'. Check inputs."
        elif isinstance(e, ValueError): user_msg = f"Error applying '{operation}': Invalid value or data type mismatch - {error_msg}. Check inputs/types."
        elif isinstance(e, MemoryError): user_msg = f"Error applying '{operation}': Insufficient memory."
        # Consider adding specific Supabase error types if needed
        flash(user_msg, "danger")
        return redirect(url_for('clean_data')) # Redirect back, state might be inconsistent


# --- AUTO CLEAN ROUTES ---

# --- SAFE AUTO CLEAN ---
@app.route('/safe_auto_clean', methods=['POST'])
def safe_auto_clean():
    object_path = session.get('current_file_path')
    original_filename = session.get('original_filename'); original_filename_base = os.path.splitext(original_filename or 'data.csv')[0]
    session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None)
    if not object_path: flash("Session expired.", "warning"); return redirect(url_for('index'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))

    df = read_data_from_supabase(object_path)
    if df is None: return redirect(url_for('clean_data'))

    try:
        start_time = time.time(); actions_taken = []
        findings = auto_explore_data(df.copy()); findings_dict = {f['issue_type']: f for f in findings}
        # Order: Duplicates -> Text Format -> Dates -> Missing Impute -> Outlier Cap
        if 'Duplicate Records' in findings_dict:
            initial_rows = df.shape[0]; df.drop_duplicates(keep='first', inplace=True); rows_removed = initial_rows - df.shape[0];
            if rows_removed > 0: actions_taken.append(f"Removed {rows_removed} duplicates")
        if 'Text Formatting' in findings_dict:
            cols = findings_dict['Text Formatting'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if is_string_dtype(df[col]) or is_object_dtype(df[col]): df[col] = df[col].astype('string').str.strip().str.lower(); count += 1
                if count > 0: actions_taken.append(f"Formatted {count} text cols")
        if 'Potential Dates' in findings_dict:
            cols = findings_dict['Potential Dates'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    try: df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True); count += 1
                    except Exception as e: print(f"Safe date fail {col}: {e}")
                if count > 0: actions_taken.append(f"Converted {count} date cols")
        if 'Missing Data' in findings_dict:
            cols = findings_dict['Missing Data'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if df[col].isnull().any():
                        try:
                            if is_numeric_dtype(df[col]): median = df[col].median(); df[col].fillna(median, inplace=True) if pd.notna(median) else print(f"Safe: No median {col}")
                            elif is_string_dtype(df[col]) or is_object_dtype(df[col]): mode = df[col].mode(); df[col].fillna(mode[0] if not mode.empty else "Unknown", inplace=True)
                            count +=1
                        except Exception as e: print(f"Safe fill fail {col}: {e}")
                if count > 0: actions_taken.append(f"Imputed missing in {count} cols")
        if 'Potential Outliers' in findings_dict:
            cols = findings_dict['Potential Outliers'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if is_numeric_dtype(df[col]):
                        try: 
                            Q1, Q3 = df[col].quantile([0.25, 0.75]); IQR = Q3 - Q1
                            if pd.notna(IQR) and IQR > 0: low, up = Q1 - 1.5*IQR, Q3 + 1.5*IQR; df[col].clip(lower=low, upper=up, inplace=True); count += 1
                        except Exception as e: print(f"Safe cap fail {col}: {e}")
                if count > 0: actions_taken.append(f"Capped outliers in {count} cols")

        new_object_path = save_data_to_supabase(df, original_filename_base)
        if new_object_path:
            if object_path != new_object_path: 
                try: 
                    supabase.storage.from_(BUCKET_NAME).remove([object_path]); 
                except Exception as e: print(f"Warn: Failed remove old {object_path}: {e}")
            session['current_file_path'] = new_object_path; duration = time.time() - start_time
            if actions_taken: flash(f"Safe Auto-Clean: {'; '.join(actions_taken)}. ({duration:.2f}s)", "success")
            else: flash(f"Safe Auto-Clean ran, no actions taken. ({duration:.2f}s)", "info")
        else: flash("Safe Auto-Clean failed to save.", "danger")
        return redirect(url_for('clean_data'))
    except Exception as e: print(f"Error Safe Auto-Clean: {e}"); traceback.print_exc(); flash(f"Error during Safe Auto-Clean: {type(e).__name__}.", "danger"); return redirect(url_for('clean_data'))


# --- HARD AUTO CLEAN (Combined Row/Col) ---
@app.route('/hard_auto_clean', methods=['POST'])
def hard_auto_clean():
    object_path = session.get('current_file_path'); original_filename = session.get('original_filename'); original_filename_base = os.path.splitext(original_filename or 'data.csv')[0]
    session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None)
    if not object_path: flash("Session expired.", "warning"); return redirect(url_for('index'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))
    df = read_data_from_supabase(object_path);
    if df is None: return redirect(url_for('clean_data'))

    try:
        start_time = time.time(); original_shape = df.shape; actions_taken = []; rows_dropped_indices = pd.Index([]); cols_dropped_list = []
        MISSING_COL_DROP_THRESHOLD = 0.80
        findings = auto_explore_data(df.copy()); findings_dict = {f['issue_type']: f for f in findings}
        # Order: Duplicates(R) -> TextFormat(V) -> Dates(V/T) -> LowVariance(C) -> MissingColDrop(C) -> OutlierRowDrop(R) -> MissingRowDrop(R)
        if 'Duplicate Records' in findings_dict:
            dup_indices = df.index[df.duplicated(keep='first')];
            if not dup_indices.empty: df.drop(index=dup_indices, inplace=True); rows_dropped_indices = rows_dropped_indices.union(dup_indices); actions_taken.append(f"Removed {len(dup_indices)} duplicates")
        if 'Text Formatting' in findings_dict:
            cols = findings_dict['Text Formatting'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if is_string_dtype(df[col]) or is_object_dtype(df[col]): df[col] = df[col].astype('string').str.strip().str.lower(); count += 1
                if count > 0: actions_taken.append(f"Formatted {count} text cols")
        if 'Potential Dates' in findings_dict:
            cols = findings_dict['Potential Dates'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    try: df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True); count += 1
                    except Exception as e: print(f"Hard date fail {col}: {e}")
                if count > 0: actions_taken.append(f"Converted {count} date cols (errors->NaN)")
        if 'Low Variance' in findings_dict:
            cols = findings_dict['Low Variance'].get('affected_columns', []); valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                try: df.drop(columns=valid_cols, inplace=True); cols_dropped_list.extend(valid_cols); actions_taken.append(f"Removed {len(valid_cols)} low-var cols")
                except Exception as e: print(f"Hard low-var drop fail: {e}")
        cols_to_drop_missing = [col for col in df.columns if col not in cols_dropped_list and df[col].isnull().mean() > MISSING_COL_DROP_THRESHOLD]
        if cols_to_drop_missing:
             try: df.drop(columns=cols_to_drop_missing, inplace=True); cols_dropped_list.extend(cols_to_drop_missing); actions_taken.append(f"Removed {len(cols_to_drop_missing)} high-NaN cols")
             except Exception as e: print(f"Hard high-NaN drop fail: {e}")
        if 'Potential Outliers' in findings_dict:
            cols = findings_dict['Potential Outliers'].get('affected_columns', []); outlier_indices = pd.Index([]); count = 0; valid_cols = [c for c in cols if c in df.columns and is_numeric_dtype(df[c])]
            if valid_cols:
                for col in valid_cols:
                    try:
                        Q1, Q3 = df[col].quantile([0.25, 0.75]); IQR = Q3 - Q1
                        if pd.notna(IQR) and IQR > 0: low, up = Q1 - 1.5*IQR, Q3 + 1.5*IQR; mask = ((df[col] < low) | (df[col] > up)) & df[col].notna(); outlier_indices = outlier_indices.union(df.index[mask]); count +=1
                    except Exception as e: print(f"Hard outlier check fail {col}: {e}")
            if not outlier_indices.empty: actual_drop = outlier_indices.difference(rows_dropped_indices);
            if not actual_drop.empty: df.drop(index=actual_drop, inplace=True); rows_dropped_indices = rows_dropped_indices.union(actual_drop); actions_taken.append(f"Removed {len(actual_drop)} outlier rows ({count} cols)")
        missing_indices = df.index[df.isnull().any(axis=1)]
        if not missing_indices.empty: actual_drop = missing_indices.difference(rows_dropped_indices);
        if not actual_drop.empty: df.drop(index=actual_drop, inplace=True); rows_dropped_indices = rows_dropped_indices.union(actual_drop); actions_taken.append(f"Removed {len(actual_drop)} rows with NaN")

        final_shape = df.shape; new_object_path = save_data_to_supabase(df, original_filename_base)
        if new_object_path:
            if object_path != new_object_path: 
                try: 
                    supabase.storage.from_(BUCKET_NAME).remove([object_path]); 
                except Exception as e: print(f"Warn: Failed remove old {object_path}: {e}")
            session['current_file_path'] = new_object_path; duration = time.time() - start_time; total_rows_dropped = len(rows_dropped_indices); cols_dropped = len(cols_dropped_list); change_summary = ""
            if total_rows_dropped > 0: change_summary += f"{total_rows_dropped} rows removed. "
            if cols_dropped > 0: change_summary += f"{cols_dropped} cols removed. "
            if actions_taken: flash(f"Hard Auto-Clean: {change_summary}[{'; '.join(actions_taken)}]. Final: {final_shape[0]}R, {final_shape[1]}C. ({duration:.2f}s)", "danger")
            else: flash(f"Hard Auto-Clean ran, no actions taken. ({duration:.2f}s)", "info")
        else: flash("Hard Auto-Clean failed to save.", "danger")
        return redirect(url_for('clean_data'))
    except Exception as e: print(f"Error Hard Auto-Clean: {e}"); traceback.print_exc(); flash(f"Error during Hard Auto-Clean: {type(e).__name__}.", "danger"); return redirect(url_for('clean_data'))


# --- HARD AUTO ROW CLEAN ---
@app.route('/hard_auto_row_clean', methods=['POST'])
def hard_auto_row_clean():
    object_path = session.get('current_file_path'); original_filename = session.get('original_filename'); original_filename_base = os.path.splitext(original_filename or 'data.csv')[0]
    session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None)
    if not object_path: flash("Session expired.", "warning"); return redirect(url_for('index'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))
    df = read_data_from_supabase(object_path);
    if df is None: return redirect(url_for('clean_data'))

    try:
        start_time = time.time(); original_shape = df.shape; actions_taken = []; rows_dropped_indices = pd.Index([])
        findings = auto_explore_data(df.copy()); findings_dict = {f['issue_type']: f for f in findings}
        # Order: Duplicates(R) -> TextFormat(V) -> Dates(V/T) -> OutlierRowDrop(R) -> MissingRowDrop(R)
        if 'Duplicate Records' in findings_dict:
            dup_indices = df.index[df.duplicated(keep='first')];
            if not dup_indices.empty: df.drop(index=dup_indices, inplace=True); rows_dropped_indices = rows_dropped_indices.union(dup_indices); actions_taken.append(f"Removed {len(dup_indices)} duplicates")
        if 'Text Formatting' in findings_dict:
            cols = findings_dict['Text Formatting'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if is_string_dtype(df[col]) or is_object_dtype(df[col]): df[col] = df[col].astype('string').str.strip().str.lower(); count += 1
                if count > 0: actions_taken.append(f"Formatted {count} text cols")
        if 'Potential Dates' in findings_dict:
            cols = findings_dict['Potential Dates'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    try: df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True); count += 1
                    except Exception as e: print(f"HardRow date fail {col}: {e}")
                if count > 0: actions_taken.append(f"Converted {count} date cols (errors->NaN)")
        if 'Potential Outliers' in findings_dict:
            cols = findings_dict['Potential Outliers'].get('affected_columns', []); outlier_indices = pd.Index([]); count = 0; valid_cols = [c for c in cols if c in df.columns and is_numeric_dtype(df[c])]
            if valid_cols:
                for col in valid_cols:
                    try:
                        Q1, Q3 = df[col].quantile([0.25, 0.75]); IQR = Q3 - Q1
                        if pd.notna(IQR) and IQR > 0: low, up = Q1 - 1.5*IQR, Q3 + 1.5*IQR; mask = ((df[col] < low) | (df[col] > up)) & df[col].notna(); outlier_indices = outlier_indices.union(df.index[mask]); count +=1
                    except Exception as e: print(f"HardRow outlier check fail {col}: {e}")
            if not outlier_indices.empty: actual_drop = outlier_indices.difference(rows_dropped_indices);
            if not actual_drop.empty: df.drop(index=actual_drop, inplace=True); rows_dropped_indices = rows_dropped_indices.union(actual_drop); actions_taken.append(f"Removed {len(actual_drop)} outlier rows ({count} cols)")
        missing_indices = df.index[df.isnull().any(axis=1)]
        if not missing_indices.empty: actual_drop = missing_indices.difference(rows_dropped_indices);
        if not actual_drop.empty: df.drop(index=actual_drop, inplace=True); rows_dropped_indices = rows_dropped_indices.union(actual_drop); actions_taken.append(f"Removed {len(actual_drop)} rows with NaN")

        final_shape = df.shape; new_object_path = save_data_to_supabase(df, original_filename_base)
        if new_object_path:
            if object_path != new_object_path: 
                try: 
                    supabase.storage.from_(BUCKET_NAME).remove([object_path]); 
                except Exception as e: print(f"Warn: Failed remove old {object_path}: {e}")
            session['current_file_path'] = new_object_path; duration = time.time() - start_time; total_rows_dropped = len(rows_dropped_indices); cols_dropped = original_shape[1] - final_shape[1]; change_summary = ""
            if total_rows_dropped > 0: change_summary += f"{total_rows_dropped} rows removed. "
            if cols_dropped > 0: change_summary += f"{cols_dropped} cols removed (Unexpected!). " # Should be 0
            if actions_taken: flash(f"Hard Row Auto-Clean: {change_summary}[{'; '.join(actions_taken)}]. Final: {final_shape[0]}R, {final_shape[1]}C. ({duration:.2f}s)", "warning")
            else: flash(f"Hard Row Auto-Clean ran, no actions taken. ({duration:.2f}s)", "info")
        else: flash("Hard Row Auto-Clean failed to save.", "danger")
        return redirect(url_for('clean_data'))
    except Exception as e: print(f"Error Hard Row Auto-Clean: {e}"); traceback.print_exc(); flash(f"Error during Hard Row Auto-Clean: {type(e).__name__}.", "danger"); return redirect(url_for('clean_data'))


# --- HARD AUTO COLUMN CLEAN ---
@app.route('/hard_auto_column_clean', methods=['POST'])
def hard_auto_column_clean():
    object_path = session.get('current_file_path'); original_filename = session.get('original_filename'); original_filename_base = os.path.splitext(original_filename or 'data.csv')[0]
    session.pop('exploration_results', None); session.pop('profile_results', None); session.pop('fuzzy_results', None)
    if not object_path: flash("Session expired.", "warning"); return redirect(url_for('index'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))
    df = read_data_from_supabase(object_path);
    if df is None: return redirect(url_for('clean_data'))

    try:
        start_time = time.time(); original_shape = df.shape; actions_taken = []; cols_dropped_list = []
        MISSING_COL_DROP_THRESHOLD = 0.80
        findings = auto_explore_data(df.copy()); findings_dict = {f['issue_type']: f for f in findings}
        # Order: Duplicates(R) -> TextFormat(V) -> Dates(V/T) -> LowVariance(C) -> MissingColDrop(C)
        if 'Duplicate Records' in findings_dict:
            initial_rows = df.shape[0]; df.drop_duplicates(keep='first', inplace=True); rows_removed = initial_rows - df.shape[0];
            if rows_removed > 0: actions_taken.append(f"Removed {rows_removed} duplicates")
        if 'Text Formatting' in findings_dict:
            cols = findings_dict['Text Formatting'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    if is_string_dtype(df[col]) or is_object_dtype(df[col]): df[col] = df[col].astype('string').str.strip().str.lower(); count += 1
                if count > 0: actions_taken.append(f"Formatted {count} text cols")
        if 'Potential Dates' in findings_dict:
            cols = findings_dict['Potential Dates'].get('affected_columns', []); count = 0; valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                for col in valid_cols:
                    try: df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True); count += 1
                    except Exception as e: print(f"HardCol date fail {col}: {e}")
                if count > 0: actions_taken.append(f"Converted {count} date cols (errors->NaN)")
        if 'Low Variance' in findings_dict:
            cols = findings_dict['Low Variance'].get('affected_columns', []); valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                try: df.drop(columns=valid_cols, inplace=True); cols_dropped_list.extend(valid_cols); actions_taken.append(f"Removed {len(valid_cols)} low-var cols")
                except Exception as e: print(f"HardCol low-var drop fail: {e}")
        cols_to_drop_missing = [col for col in df.columns if col not in cols_dropped_list and df[col].isnull().mean() > MISSING_COL_DROP_THRESHOLD]
        if cols_to_drop_missing:
             try: df.drop(columns=cols_to_drop_missing, inplace=True); cols_dropped_list.extend(cols_to_drop_missing); actions_taken.append(f"Removed {len(cols_to_drop_missing)} high-NaN cols")
             except Exception as e: print(f"HardCol high-NaN drop fail: {e}")
        # Skip Row Drops

        final_shape = df.shape; new_object_path = save_data_to_supabase(df, original_filename_base)
        if new_object_path:
            if object_path != new_object_path: 
                try: 
                    supabase.storage.from_(BUCKET_NAME).remove([object_path]); 
                except Exception as e: print(f"Warn: Failed remove old {object_path}: {e}")
            session['current_file_path'] = new_object_path; duration = time.time() - start_time; rows_dropped = original_shape[0] - final_shape[0]; cols_dropped = len(cols_dropped_list); change_summary = ""
            if rows_dropped > 0: change_summary += f"{rows_dropped} dup rows removed. "
            if cols_dropped > 0: change_summary += f"{cols_dropped} cols removed. "
            if actions_taken: flash(f"Hard Column Auto-Clean: {change_summary}[{'; '.join(actions_taken)}]. Final: {final_shape[0]}R, {final_shape[1]}C. ({duration:.2f}s)", "warning")
            else: flash(f"Hard Column Auto-Clean ran, no actions taken. ({duration:.2f}s)", "info")
        else: flash("Hard Column Auto-Clean failed to save.", "danger")
        return redirect(url_for('clean_data'))
    except Exception as e: print(f"Error Hard Column Auto-Clean: {e}"); traceback.print_exc(); flash(f"Error during Hard Column Auto-Clean: {type(e).__name__}.", "danger"); return redirect(url_for('clean_data'))


# Add this updated route function to your app.py, replacing the previous /analysis route

@app.route('/analysis', methods=['GET']) # Only handles GET requests
def analyze_data():
    """Displays the analysis page and handles requests for specific analyses."""
    object_path = session.get('current_file_path')
    original_filename = session.get('original_filename', 'Pasted Data')

    # --- Basic Checks ---
    if not object_path:
        flash("No data loaded. Please paste data on the home page first.", "warning")
        return redirect(url_for('index'))
    if not supabase:
        flash("Cloud storage client not available. Cannot perform analysis.", "danger")
        return redirect(url_for('clean_data'))

    # --- Load Data ---
    df = read_data_from_supabase(object_path)
    if df is None:
        # read_data_from_supabase should have flashed an error
        return redirect(url_for('clean_data')) # Redirect as data is inaccessible

    # --- Get Column Info for UI ---
    try:
        column_types = get_column_types(df)
        all_columns = df.columns.tolist()
        current_shape = df.shape
    except Exception as e:
        flash(f"Error getting column information: {e}", "danger")
        return redirect(url_for('clean_data'))

    # --- Initialize variables for analysis results ---
    # Use more specific names and include slots for new results
    analysis_results = {
        "action_performed": None,
        "error": None,
        # Single Variable Results
        "single_col_name": None,
        "single_num_stats": None,          # Dict
        "single_num_hist_div": None,       # Plotly Div HTML
        "single_num_box_div": None,        # Plotly Div HTML
        "single_cat_table_html": None,     # HTML Table
        "single_cat_bar_div": None,        # Plotly Div HTML
        "single_dt_stats": None,           # Dict
        "single_dt_plot_div": None,        # Plotly Div HTML
        # Bivariate Results
        "scatter_col_x": None,
        "scatter_col_y": None,
        "scatter_results": None,           # Dict (corr_p, corr_s, p_val_p)
        "scatter_plot_div": None,          # Plotly Div HTML
        "num_cat_num_col": None,
        "num_cat_cat_col": None,
        "num_cat_box_div": None,           # Plotly Div HTML
        "num_cat_violin_div": None,      # Plotly Div HTML
        "cat_cat_col1": None,
        "cat_cat_col2": None,
        "crosstab_html": None,             # HTML Table
        "crosstab_heatmap_div": None,    # Plotly Div HTML
        "chi2_test_results": None,         # Dict
        # Overview Results
        "correlation_method": None,
        "correlation_heatmap_div": None, # Plotly Div HTML
        # GroupBy Results
        "groupby_table_html": None,        # HTML Table
        # Hypothesis Test Results
        "ttest_num_col": None,
        "ttest_cat_col": None,
        "ttest_results": None,             # Dict
        "anova_num_col": None,
        "anova_cat_col": None,
        "anova_results": None              # Dict
    }

    # --- Check for requested analysis action ---
    action = request.args.get('action')
    analysis_results["action_performed"] = action

    if action:
        print(f"Analysis action requested: {action} with args: {request.args}")
        try:
            # --- Single Variable Analysis ---
            if action == 'single_variable':
                col_name = request.args.get('single_column')
                analysis_results["single_col_name"] = col_name
                if col_name and col_name in all_columns:
                    if col_name in column_types['numeric']:
                        stats_dict, hist_div, box_div = analyze_numeric_column(df, col_name)
                        analysis_results["single_num_stats"] = stats_dict
                        analysis_results["single_num_hist_div"] = hist_div
                        analysis_results["single_num_box_div"] = box_div
                    elif col_name in column_types['categorical']:
                         table_html, bar_div = analyze_categorical_column(df, col_name)
                         analysis_results["single_cat_table_html"] = table_html
                         analysis_results["single_cat_bar_div"] = bar_div
                    elif col_name in column_types['datetime']:
                        stats_dict, plot_div = analyze_datetime_column(df, col_name)
                        analysis_results["single_dt_stats"] = stats_dict
                        analysis_results["single_dt_plot_div"] = plot_div
                    else:
                        flash(f"Analysis for column type of '{col_name}' not implemented.", "info")
                elif col_name:
                     flash(f"Selected column '{col_name}' not found in the dataset.", "warning")
                else:
                    flash("Please select a column for single variable analysis.", "warning")

            # --- Bivariate Analysis ---
            elif action == 'numeric_vs_numeric':
                col_x = request.args.get('scatter_x')
                col_y = request.args.get('scatter_y')
                analysis_results["scatter_col_x"] = col_x
                analysis_results["scatter_col_y"] = col_y
                if col_x and col_y and col_x in column_types['numeric'] and col_y in column_types['numeric']:
                    if col_x == col_y: flash("Please select two different numeric columns.", "warning")
                    else:
                        results_dict, plot_div = analyze_numeric_vs_numeric(df, col_x, col_y)
                        analysis_results["scatter_results"] = results_dict
                        analysis_results["scatter_plot_div"] = plot_div
                elif col_x or col_y : # Only flash if one or both were selected but invalid
                    flash("Please select two valid numeric columns.", "warning")

            elif action == 'numeric_vs_categorical':
                num_col = request.args.get('num_cat_numeric')
                cat_col = request.args.get('num_cat_categorical')
                analysis_results["num_cat_num_col"] = num_col
                analysis_results["num_cat_cat_col"] = cat_col
                if num_col and cat_col and num_col in column_types['numeric'] and cat_col in all_columns:
                     # Helper checks cardinality
                     box_div, violin_div = analyze_numeric_vs_categorical(df, num_col, cat_col)
                     analysis_results["num_cat_box_div"] = box_div
                     analysis_results["num_cat_violin_div"] = violin_div
                elif num_col or cat_col:
                     flash("Please select a valid numeric and a categorical column.", "warning")

            elif action == 'categorical_vs_categorical':
                cat_col1 = request.args.get('cat_cat_1')
                cat_col2 = request.args.get('cat_cat_2')
                analysis_results["cat_cat_col1"] = cat_col1
                analysis_results["cat_cat_col2"] = cat_col2
                if cat_col1 and cat_col2 and cat_col1 in all_columns and cat_col2 in all_columns:
                    if cat_col1 == cat_col2: flash("Please select two different categorical columns.", "warning")
                    else:
                        crosstab_html, heatmap_div, test_results = analyze_categorical_vs_categorical(df, cat_col1, cat_col2)
                        analysis_results["crosstab_html"] = crosstab_html
                        analysis_results["crosstab_heatmap_div"] = heatmap_div
                        analysis_results["chi2_test_results"] = test_results
                elif cat_col1 or cat_col2:
                     flash("Please select two valid categorical columns.", "warning")

            # --- Overview Analysis ---
            elif action == 'correlation_heatmap':
                 corr_method = request.args.get('corr_method', 'pearson') # Default to pearson
                 analysis_results["correlation_method"] = corr_method
                 heatmap_div = generate_correlation_heatmap(df, method=corr_method)
                 analysis_results["correlation_heatmap_div"] = heatmap_div

            # --- GroupBy Analysis ---
            elif action == 'grouped_summary':
                 group_cols = request.args.getlist('groupby_cols')
                 val_cols = request.args.getlist('groupby_vals')
                 agg_funcs = request.args.getlist('groupby_funcs')
                 summary_html = generate_grouped_summary(df, group_cols, val_cols, agg_funcs)
                 # Store inputs for potential display/confirmation in template if needed
                 # analysis_results["groupby_cols"] = group_cols
                 # analysis_results["groupby_vals"] = val_cols
                 # analysis_results["groupby_funcs"] = agg_funcs
                 analysis_results["groupby_table_html"] = summary_html

            # --- Hypothesis Testing ---
            elif action == 'ttest':
                 num_col = request.args.get('ttest_numeric')
                 cat_col = request.args.get('ttest_cat_binary')
                 analysis_results["ttest_num_col"] = num_col
                 analysis_results["ttest_cat_col"] = cat_col
                 if num_col and cat_col and num_col in column_types['numeric'] and cat_col in all_columns:
                     # Helper function performs validation on binary nature
                     test_results = perform_ttest_ind(df, num_col, cat_col)
                     analysis_results["ttest_results"] = test_results
                 elif num_col or cat_col:
                     flash("Please select a numeric and a binary categorical column for T-test.", "warning")

            elif action == 'anova':
                 num_col = request.args.get('anova_numeric')
                 cat_col = request.args.get('anova_cat_multi')
                 analysis_results["anova_num_col"] = num_col
                 analysis_results["anova_cat_col"] = cat_col
                 if num_col and cat_col and num_col in column_types['numeric'] and cat_col in all_columns:
                      # Helper function performs validation
                     test_results = perform_anova(df, num_col, cat_col)
                     analysis_results["anova_results"] = test_results
                 elif num_col or cat_col:
                     flash("Please select a numeric and a categorical column (>=2 groups) for ANOVA.", "warning")

            else:
                 flash(f"Unknown analysis action requested: {action}", "warning")

        except Exception as e:
            error_msg = f"Error performing analysis action '{action}': {type(e).__name__} - {e}"
            print(error_msg)
            traceback.print_exc()
            flash(error_msg, "danger")
            analysis_results["error"] = error_msg # Store error for display

    # --- Render the analysis page template ---
    # Pass common info and unpack the results dictionary
    return render_template(
        'analysis.html',
        filename=original_filename,
        current_shape=current_shape,
        all_columns=all_columns,
        numeric_cols=column_types['numeric'],
        categorical_cols=column_types['categorical'],
        low_cardinality_cols=column_types['low_cardinality_categorical'],
        datetime_cols=column_types['datetime'],
        # Unpack all analysis results (plot divs, tables, stats dicts, etc.)
        **analysis_results
    )

# --- END of /analysis route ---


# --- Visualization Route ---
@app.route('/visualize')
def visualize_data():
    object_path = session.get('current_file_path'); original_filename = session.get('original_filename', 'Pasted Data')
    if not object_path: flash("No data loaded.", "warning"); return redirect(url_for('index'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))
    df = read_data_from_supabase(object_path)
    if df is None: return redirect(url_for('clean_data'))
    if df.empty: flash("Cannot visualize empty dataset.", "warning"); return redirect(url_for('clean_data'))

    try:
        plots = {'histograms': [], 'bar_charts': [], 'scatter_plots': []}; max_cats=20; max_scatters=10
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        for col in numeric_cols: # Histograms
            if df[col].nunique(dropna=True) < 2: continue
            try: fig = px.histogram(df, x=col, title=f'Distribution of {col}', template='plotly_white', marginal="box"); fig.update_layout(bargap=0.1, title_x=0.5, height=350, margin=dict(l=40, r=20, t=50, b=30)); plots['histograms'].append({'div': pio.to_html(fig, full_html=False, include_plotlyjs=False)})
            except Exception as e: print(f"Vis fail hist {col}: {e}"); flash(f"Plot failed: Hist '{col}'.", "warning")
        for col in cat_cols: # Bar Charts
            unique_non_na = df[col].nunique(dropna=True)
            if 0 < unique_non_na <= max_cats:
                 try: counts = df[col].value_counts(dropna=False).reset_index(); counts.columns = [col, 'count']; counts[col] = counts[col].astype(str).fillna('(Missing)'); fig = px.bar(counts.head(max_cats), x=col, y='count', title=f'Top {max_cats} Counts: {col}', template='plotly_white'); fig.update_layout(title_x=0.5, height=350, margin=dict(l=40, r=20, t=50, b=30)); plots['bar_charts'].append({'div': pio.to_html(fig, full_html=False, include_plotlyjs=False)})
                 except Exception as e: print(f"Vis fail bar {col}: {e}"); flash(f"Plot failed: Bar '{col}'.", "warning")
            elif unique_non_na > max_cats: flash(f"Skip bar '{col}': >{max_cats} unique values.", "info")
        scatter_count = 0 # Scatter Plots
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if scatter_count >= max_scatters: break; col1, col2 = numeric_cols[i], numeric_cols[j]
                    if df[col1].nunique(dropna=True) < 2 or df[col2].nunique(dropna=True) < 2: continue
                    try: fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}', template='plotly_white', opacity=0.6, trendline="ols", trendline_color_override="red"); fig.update_layout(title_x=0.5, height=400, margin=dict(l=40, r=20, t=50, b=30)); plots['scatter_plots'].append({'div': pio.to_html(fig, full_html=False, include_plotlyjs=False)}); scatter_count += 1
                    except Exception as e: print(f"Vis fail scatter {col1}v{col2}: {e}"); flash(f"Plot failed: Scatter '{col1}' vs '{col2}'.", "warning")
                if scatter_count >= max_scatters: flash(f"Stopped scatter plots at {max_scatters}.", "info"); break
        return render_template('visualize.html', filename=original_filename, plots=plots)
    except Exception as e: flash(f"Error generating visualizations: {e}", "danger"); traceback.print_exc(); return redirect(url_for('clean_data'))


# --- Download Route ---
@app.route('/download')
def download_file():
    """Provides the current cleaned file from Supabase for download."""
    object_path = session.get('current_file_path')
    session_filename = session.get('original_filename', 'data.csv')
    requested_format = request.args.get('format', 'csv').lower()

    if not object_path: flash("No file available for download.", "warning"); return redirect(url_for('clean_data'))
    if not supabase: flash("Cloud storage client not available.", "danger"); return redirect(url_for('clean_data'))

    base, _ = os.path.splitext(session_filename); secure_base = secure_filename(base) or "cleaned_data"

    try:
        if requested_format == 'csv':
            output_filename = f"{secure_base}.csv"
            # Download bytes directly from Supabase
            file_bytes = supabase.storage.from_(BUCKET_NAME).download(object_path)
            # Return as a Flask response
            return Response(
                file_bytes,
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={output_filename}"}
            )
        elif requested_format == 'xlsx':
            output_filename = f"{secure_base}.xlsx"
            # Need to read into DataFrame first
            df = read_data_from_supabase(object_path)
            if df is None:
                # read_data already flashed error
                return redirect(url_for('clean_data'))
            # Convert DF to Excel in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                 df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            excel_buffer.seek(0)
            # Return as Flask response
            return Response(
                excel_buffer.getvalue(),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-disposition": f"attachment; filename={output_filename}"}
            )
        else:
            flash(f"Unsupported download format: {requested_format}", "warning")
            return redirect(url_for('clean_data'))

    except Exception as e:
        # Handle Supabase download errors etc.
        error_type = type(e).__name__
        print(f"Error during download preparation: {error_type} - {e}")
        traceback.print_exc()
        if "Not Found" in str(e) or "OBJECT_NOT_FOUND" in str(e):
             flash(f"Error: Could not find intermediate data file '{object_path}' in cloud storage for download.", "danger")
        else:
             flash(f"Error preparing file for download: {error_type}", "danger")
        return redirect(url_for('clean_data'))


# --- Local File Cleanup (No longer needed for intermediate data) ---
# def cleanup_old_files(folder, max_age_seconds=...): ...


# --- Main Execution ---
if __name__ == '__main__':
    # Create folders if they don't exist (Upload folder might still be useful)
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists('static/css'): os.makedirs('static/css', exist_ok=True)

    # No local cleanup needed for CLEANED_FOLDER anymore

    port = int(os.environ.get('PORT', 5000))
    # Set debug=False for production! Use threaded=True for dev convenience.
    print(f"Starting Flask app on port {port} with Supabase configured: {'YES' if supabase else 'NO'}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
