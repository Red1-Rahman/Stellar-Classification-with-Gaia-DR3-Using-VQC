import os
import pandas as pd
from astroquery.gaia import Gaia
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# -------------------------------
# Setup logging
# -------------------------------
os.makedirs("../logs", exist_ok=True)
log_filename = f"../logs/gaia_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())  # also print to console

def fetch_and_prepare_gaia(
    n_rows=10000,
    k_best=20,
    save_path="../Datasets/Processed/gaia_features_20.csv"
):
    logging.info(f"Starting Gaia fetch: {n_rows} rows, selecting top {k_best} features.")

    # -------------------------------
    # 1. Fetch Gaia DR3 features
    # -------------------------------
    query = f"""
    SELECT TOP {n_rows}
      ra, dec, parallax, parallax_error,
      pmra, pmra_error, pmdec, pmdec_error,
      radial_velocity, radial_velocity_error,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
      phot_g_mean_flux, phot_bp_mean_flux, phot_rp_mean_flux,
      phot_g_mean_flux_error, phot_bp_mean_flux_error, phot_rp_mean_flux_error,
      phot_bp_rp_excess_factor,
      phot_g_n_obs, phot_bp_n_obs, phot_rp_n_obs,
      teff_val, radius_val, lum_val,
      ruwe, astrometric_excess_noise, astrometric_chi2_al, astrometric_n_good_obs_al,
      visibility_periods_used
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    AND bp_rp IS NOT NULL
    """

    logging.info("Querying Gaia DR3...")
    job = Gaia.launch_job_async(query)
    df = job.get_results().to_pandas()
    logging.info(f"Fetched {df.shape[0]} rows and {df.shape[1]} columns.")

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
    X_best = SelectKBest(f_classif, k=k_best).fit_transform(X_var, y)
    logging.info(f"Selected top {k_best} features.")

    # -------------------------------
    # 5. Train/test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y[:len(X_best)], test_size=0.2, random_state=42, stratify=y[:len(X_best)]
    )
    logging.info(f"Train/test split done: Train {X_train.shape}, Test {X_test.shape}")

    # -------------------------------
    # 6. Save processed dataset
    # -------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed_df = pd.DataFrame(X_best, columns=[f"feat_{i}" for i in range(X_best.shape[1])])
    processed_df['label'] = y[:len(X_best)]
    processed_df.to_csv(save_path, index=False)
    logging.info(f"Saved processed dataset to {save_path}")
    logging.info(f"Logging to {log_filename} completed.")

    return X_train, X_test, y_train, y_test

# -------------------------------
# Optional: run script standalone
# -------------------------------
if __name__ == "__main__":
    fetch_and_prepare_gaia()
