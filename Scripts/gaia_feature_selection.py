import os
import pandas as pd
import numpy as np
import time
import random
from astroquery.gaia import Gaia
from astroquery.exceptions import RemoteServiceError, InvalidQueryError
import requests
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Path setup for cross-platform compatibility
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "Datasets", "Processed")
LOGS_DIR = os.path.join(PROJ_ROOT, "logs")

# -------------------------------
# Setup logging
# -------------------------------
os.makedirs(LOGS_DIR, exist_ok=True)
log_filename = os.path.join(LOGS_DIR, f"gaia_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())  # also print to console

def check_connectivity():
    """Check internet and Gaia server connectivity."""
    try:
        # Check general internet
        requests.get("http://www.google.com", timeout=5)
        logging.info("Internet connection verified")
        
        # Check Gaia server
        gaia_url = "https://gea.esac.esa.int/tap-server/tap"
        response = requests.get(gaia_url, timeout=10)
        if response.status_code == 200:
            logging.info("Gaia TAP server is reachable")
            return True
        else:
            logging.warning(f"Gaia server returned status: {response.status_code}")
            return False
    except Exception as e:
        logging.warning(f"Connectivity check failed: {e}")
        return False

def generate_synthetic_gaia_data(n_rows=20000):
    """Generate synthetic stellar data when Gaia is unavailable."""
    logging.info(f"Generating {n_rows} rows of synthetic stellar data")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic stellar parameters
    data = {
        'ra': np.random.uniform(0, 360, n_rows),
        'dec': np.random.uniform(-90, 90, n_rows),
        'parallax': np.random.exponential(2, n_rows) + 0.1,  # mas
        'parallax_error': np.random.exponential(0.1, n_rows) + 0.01,
        'pmra': np.random.normal(0, 10, n_rows),
        'pmra_error': np.random.exponential(1, n_rows) + 0.1,
        'pmdec': np.random.normal(0, 10, n_rows),
        'pmdec_error': np.random.exponential(1, n_rows) + 0.1,
        'radial_velocity': np.random.normal(0, 30, n_rows),
        'radial_velocity_error': np.random.exponential(2, n_rows) + 0.5,
        'phot_g_mean_mag': np.random.normal(15, 3, n_rows),
        'phot_bp_mean_mag': np.random.normal(15.5, 3, n_rows),
        'phot_rp_mean_mag': np.random.normal(14.5, 3, n_rows),
        'phot_g_mean_flux': 10**(-(np.random.normal(15, 3, n_rows) - 25.7) / 2.5),
        'phot_bp_mean_flux': 10**(-(np.random.normal(15.5, 3, n_rows) - 25.7) / 2.5),
        'phot_rp_mean_flux': 10**(-(np.random.normal(14.5, 3, n_rows) - 25.7) / 2.5),
        'phot_g_mean_flux_error': np.random.exponential(1000, n_rows),
        'phot_bp_mean_flux_error': np.random.exponential(1000, n_rows),
        'phot_rp_mean_flux_error': np.random.exponential(1000, n_rows),
        'phot_bp_rp_excess_factor': np.random.normal(1.2, 0.2, n_rows),
        'phot_g_n_obs': np.random.poisson(50, n_rows),
        'phot_bp_n_obs': np.random.poisson(30, n_rows),
        'phot_rp_n_obs': np.random.poisson(30, n_rows),
        'teff_val': np.random.normal(5500, 1500, n_rows),
        'radius_val': np.random.lognormal(0, 0.5, n_rows),
        'lum_val': np.random.lognormal(0, 1, n_rows),
        'ruwe': np.random.exponential(1, n_rows) + 0.8,
        'astrometric_excess_noise': np.random.exponential(1, n_rows),
        'astrometric_chi2_al': np.random.chisquare(5, n_rows),
        'astrometric_n_good_obs_al': np.random.poisson(8, n_rows) + 5,
        'visibility_periods_used': np.random.poisson(10, n_rows) + 5
    }
    
    df = pd.DataFrame(data)
    # Ensure no negative values for magnitudes and positive parallax
    df['parallax'] = np.abs(df['parallax'])
    df['phot_g_mean_mag'] = np.abs(df['phot_g_mean_mag'])
    df['phot_bp_mean_mag'] = np.abs(df['phot_bp_mean_mag'])
    df['phot_rp_mean_mag'] = np.abs(df['phot_rp_mean_mag'])
    
    return df

def fetch_gaia_with_retry(query, max_retries=3, base_delay=5):
    """Fetch Gaia data with retry logic."""
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting Gaia query (attempt {attempt + 1}/{max_retries})")
            job = Gaia.launch_job_async(query)
            result = job.get_results().to_pandas()
            logging.info(f"Successfully fetched {len(result)} rows from Gaia")
            return result
        except Exception as e:
            logging.error(f"Gaia query attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logging.error("All Gaia query attempts failed")
                raise

def fetch_gaia_chunked(query_base, n_rows, rows_per_chunk=200, max_retries=3, base_delay=5.0):
    """
    Fetch up to n_rows from Gaia using repeated small 'SELECT TOP ...' queries,
    paging by source_id to avoid very large single queries that trigger server 500s.
    Returns a pandas.DataFrame (possibly with fewer than n_rows if archive limits apply).
    """
    collected = []
    last_source_id = 0
    total_needed = int(n_rows)
    rows_per_chunk = int(min(rows_per_chunk, total_needed))
    attempt_log = []

    # Try a simplified direct query first for better performance
    try:
        logging.info("Attempting simplified direct query first")
        simplified_query = f"""
        SELECT TOP {min(500, total_needed)} 
        source_id, ra, dec, parallax, parallax_error, 
        pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, 
        bp_rp, ruwe
        FROM gaiadr3.gaia_source
        WHERE parallax > 0 
        AND ruwe < 1.4
        """
        job = Gaia.launch_job_async(simplified_query, output_format="votable", dump_to_file=False)
        table = job.get_results()
        try:
            df_direct = table.to_pandas()
            if not df_direct.empty and df_direct.shape[0] >= 100:
                logging.info(f"Direct query successful, fetched {df_direct.shape[0]} rows")
                return df_direct
        except Exception as e:
            logging.warning(f"Direct query processing failed: {e}")
    except Exception as e:
        logging.warning(f"Direct query attempt failed: {e}")
    
    # If direct query failed, proceed with chunking with a simpler query
    # Simplify the query_base to include only essential columns for better stability
    simplified_columns = """
        source_id, ra, dec, parallax, parallax_error, 
        pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, 
        bp_rp, ruwe
    """
    simplified_where = """
        FROM gaiadr3.gaia_source
        WHERE parallax > 0 
        AND ruwe < 1.4
    """
    # Use the simplified query instead of the original one
    query_base = f"{simplified_columns} {simplified_where}"
    
    while len(collected) < total_needed:
        to_request = min(rows_per_chunk, total_needed - len(collected))
        # Build chunked ADQL using TOP and a source_id > last_source_id filter
        query_chunk = f"SELECT TOP {to_request} {query_base} AND source_id > {last_source_id} ORDER BY source_id ASC"

        # Try the chunk with retries
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Submitting chunk query (want {to_request} rows) attempt {attempt}/{max_retries}")
                # Try multiple output formats if needed
                formats_to_try = ["votable", "csv", "votable_gzip"] if attempt == 1 else ["votable_gzip"]
                
                for output_format in formats_to_try:
                    try:
                        job = Gaia.launch_job_async(query_chunk, output_format=output_format, dump_to_file=False)
                        table = job.get_results()
                        try:
                            df_chunk = table.to_pandas()
                        except Exception:
                            import astropy.table as at
                            df_chunk = at.Table(table).to_pandas()
                        
                        # If successful, break format loop
                        if not df_chunk.empty:
                            break
                    except Exception as format_e:
                        logging.warning(f"Format {output_format} failed: {format_e}")
                        continue
                
                # If no rows returned after trying all formats, stop paging
                if df_chunk.empty:
                    logging.info("No more rows returned by Gaia for this paging strategy.")
                    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
                
                # Append and update last_source_id
                collected.append(df_chunk)
                # Ensure source_id exists and update last_source_id safely
                if "source_id" in df_chunk.columns and df_chunk["source_id"].size > 0:
                    last_source_id = int(df_chunk["source_id"].max())
                else:
                    # If no source_id column, fallback to stopping to avoid infinite loop
                    logging.warning("No source_id column in chunk results; stopping fetch to avoid infinite loop.")
                    return pd.concat(collected, ignore_index=True)
                
                logging.info(f"Fetched {df_chunk.shape[0]} rows in chunk; total collected: {sum([c.shape[0] for c in collected])}")
                # Longer pause to be extra polite to the server
                time.sleep(random.uniform(1.0, 3.0))
                break  # successful chunk, break retry loop
            
            except RemoteServiceError as e:
                # Service-level errors (e.g., 500). Log and retry with backoff + jitter.
                status = getattr(e, "status", None)
                msg = getattr(e, "message", str(e))
                logging.warning(f"Gaia RemoteServiceError (attempt {attempt}/{max_retries}): status={status} message={msg}")
                
                # Try alternative query method on last attempt
                if attempt == max_retries:
                    try:
                        logging.info("Trying alternative query method with Gaia.query()")
                        # Use synchronous query as fallback with even simpler query
                        simple_query = f"""
                        SELECT TOP {to_request} source_id, ra, dec, parallax, phot_g_mean_mag
                        FROM gaiadr3.gaia_source 
                        WHERE parallax > 0 
                        AND source_id > {last_source_id}
                        ORDER BY source_id ASC
                        """
                        result = Gaia.query(simple_query)
                        if result and len(result) > 0:
                            df_chunk = result.to_pandas()
                            collected.append(df_chunk)
                            last_source_id = int(df_chunk["source_id"].max())
                            logging.info(f"Alternative query method succeeded with {len(df_chunk)} rows")
                            break
                    except Exception as alt_e:
                        logging.warning(f"Alternative query method also failed: {alt_e}")
            
            except Exception as e:
                logging.warning(f"Unexpected error fetching Gaia chunk (attempt {attempt}/{max_retries}): {e}")

            # If here, retry if attempts remain
            if attempt < max_retries:
                backoff = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logging.info(f"Waiting {backoff:.1f}s before retrying chunk...")
                time.sleep(backoff)
            else:
                logging.error("Max retries reached for this chunk. Aborting chunked fetch.")
                return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()

    # Combine collected chunks
    if collected:
        df_all = pd.concat(collected, ignore_index=True)
        # Trim to requested n_rows if necessary
        if df_all.shape[0] > total_needed:
            df_all = df_all.iloc[:total_needed].reset_index(drop=True)
        return df_all
    else:
        return pd.DataFrame()

def fetch_and_prepare_gaia(
    n_rows=20000,
    k_best=20,
    save_path=None
):
    # Set default save_path if not provided
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "gaia_features_20.csv")
        
    logging.info(f"Starting Gaia fetch: {n_rows} rows, selecting top {k_best} features.")

    # Check connectivity first
    use_synthetic = not check_connectivity()
    
    # -------------------------------
    # 1. Fetch Gaia DR3 features (with fallback)
    # -------------------------------
    if use_synthetic:
        logging.warning("Using synthetic data due to connectivity issues")
        df = generate_synthetic_gaia_data(n_rows)
    else:
        # Build a query base WITHOUT TOP and WITHOUT source_id paging clause
        # Simplify columns for better success rate
        query_columns = """
          ra, dec, parallax, parallax_error,
          pmra, pmdec, 
          phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
          bp_rp, ruwe, source_id
        """
        base_where = """
        FROM gaiadr3.gaia_source
        WHERE parallax IS NOT NULL
          AND phot_g_mean_mag IS NOT NULL
          AND parallax > 0
          AND ruwe < 1.4
        """
        # Query base expected by fetch_gaia_chunked: "<cols> FROM ... WHERE ..."
        query_base = f"{query_columns} {base_where}"

        try:
            # Use smaller chunk size for better success rate
            df = fetch_gaia_chunked(query_base, n_rows, rows_per_chunk=200, max_retries=3, base_delay=5.0)
            if df.empty or df.shape[0] < max(50, n_rows // 20):  # Lower threshold
                logging.warning("Fetched dataframe is small or empty; falling back to synthetic data")
                df = generate_synthetic_gaia_data(n_rows)
                
                # Report to stderr for diagnostic purposes
                import sys
                print("WARNING: Gaia query returned insufficient data, using synthetic data instead.", file=sys.stderr)
        except Exception as e:
            logging.error(f"All Gaia attempts failed: {e}")
            logging.info("Falling back to synthetic data")
            df = generate_synthetic_gaia_data(n_rows)

    logging.info(f"Using dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # -------------------------------
    # 2. Handle missing values & define features/labels
    # -------------------------------
    df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=["ra", "dec", "parallax"]).values
    y = (df["parallax"].values > 5.0).astype(int)
    logging.info("Filled missing values and defined labels (nearby vs distant stars).")

    # -------------------------------
    # 3. Normalize features
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Normalized features.")

    # -------------------------------
    # 4. Feature selection
    # -------------------------------
    X_var = VarianceThreshold(threshold=0.01).fit_transform(X_scaled)

    # Ensure k_best is not larger than the available features
    k_used = min(int(k_best), X_var.shape[1]) if X_var.shape[1] > 0 else 0
    if k_used <= 0:
        logging.error("No features available after variance thresholding.")
        raise ValueError("No features available after variance thresholding.")

    X_best = SelectKBest(f_classif, k=k_used).fit_transform(X_var, y)
    logging.info(f"Selected top {k_used} features.")

    # -------------------------------
    # 5. Train/test split
    # -------------------------------
    # Use matched label length and proper argument syntax
    y_trim = y[: X_best.shape[0]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y_trim, test_size=0.2, random_state=42, stratify=y_trim
    )
    logging.info(f"Train/test split done: Train {X_train.shape}, Test {X_test.shape}")

    # -------------------------------
    # 6. Save processed dataset
    # -------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed_df = pd.DataFrame(X_best, columns=[f"feat_{i}" for i in range(X_best.shape[1])])
    processed_df['label'] = y_trim
    processed_df.to_csv(save_path, index=False)
    logging.info(f"Saved processed dataset to {save_path}")
    logging.info(f"Logging to {log_filename} completed.")

    return X_train, X_test, y_train, y_test

def fetch_gaia_data(num_rows, num_features):
    """
    Fetches stellar data from the Gaia DR3 database with robust retries,
    asynchronous TAP job fallback, and stricter quality cuts to avoid 500 errors.
    """
    try:
        if not check_connectivity():
            raise ConnectionError("No internet connection.")

        # Reduced per-query size to avoid server overload
        rows_to_request = min(num_rows, 5000)

        # Columns selected (trim/expand to your project's needs)
        columns = """
            ra, dec, parallax, parallax_error, parallax_over_error,
            pmra, pmra_error, pmdec, pmdec_error,
            ruwe, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error,
            bp_rp, radial_velocity, radial_velocity_error,
            teff_gspphot, logg_gspphot, mh_gspphot, source_id
        """

        # Stronger quality cuts to reduce heavy queries and bad data
        base_where = """
            WHERE parallax > 0
              AND parallax_over_error > 5
              AND ruwe < 1.4
              AND phot_g_mean_flux_over_error > 50
              AND phot_bp_mean_flux_over_error > 20
              AND phot_rp_mean_flux_over_error > 20
              AND phot_bp_n_obs > 0 AND phot_rp_n_obs > 0
        """

        query = f"""
            SELECT TOP {rows_to_request} {columns}
            FROM gaiadr3.gaia_source
            {base_where}
        """

        max_attempts = 3
        backoff_base = 5.0

        for attempt in range(1, max_attempts + 1):
            try:
                # Try asynchronous job (more robust for large/long queries)
                job = Gaia.launch_job_async(query, dump_to_file=False, output_format='csv')
                table = job.get_results()
                # Convert to pandas if your project expects that (you may already do this later)
                try:
                    df = table.to_pandas()
                except Exception:
                    # fallback: use astropy table directly if to_pandas not available
                    import astropy.table as at
                    df = at.Table(table).to_pandas()
                return df

            except RemoteServiceError as e:
                # Log service-level errors; attempt retry with backoff + jitter
                try:
                    status = getattr(e, 'status', None)
                    msg = getattr(e, 'message', str(e))
                except Exception:
                    status = None
                    msg = str(e)
                print(f"Gaia remote service error (attempt {attempt}/{max_attempts}): status={status} message={msg}")
            except InvalidQueryError as e:
                # Query malformed — fail fast
                print(f"Invalid Gaia ADQL query: {e}")
                raise
            except Exception as e:
                # Generic catch - log and retry
                print(f"Gaia query attempt {attempt} failed: {e}")

            if attempt < max_attempts:
                sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"Waiting {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)

        # If we reach here, all Gaia attempts failed
        raise ConnectionError("All Gaia query attempts failed; see logs for details.")
    except Exception as e:
        logging.error(f"Error in fetch_gaia_data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# -------------------------------
# Optional: run script standalone
# -------------------------------
if __name__ == "__main__":
    fetch_and_prepare_gaia()
