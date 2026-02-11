
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """
    Gaussian function: a * exp(-(x - x0)^2 / (2 * sigma^2))

    Args:
        x (np.ndarray): Input values.
        a (float): Amplitude.
        x0 (float): Center.
        sigma (float): Standard deviation.

    Returns:
        np.ndarray: Calculated Gaussian values.
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def process_calibration_files(wmx_path: str, srf_path: str, output_path: str) -> None:
    """
    Reads HyTES WMX and SRF files, fits Gaussians to each band,
    and saves the extracted Center Wavelength and FWHM.

    Args:
        wmx_path (str): Path to Wavelength Matrix file.
        srf_path (str): Path to Spectral Response Function file.
        output_path (str): Path to save the processed CSV.
    """
    print(f"Reading WMX file: {wmx_path}")
    # Skip header row (line 1), so header is on line 2 (0-indexed logic depends on pandas)
    # The file has a description line first.
    df_wmx = pd.read_csv(wmx_path, skiprows=1)
    
    print(f"Reading SRF file: {srf_path}")
    df_srf = pd.read_csv(srf_path, skiprows=1)
    
    # Verify alignment
    if len(df_wmx) != len(df_srf):
        raise ValueError(f"Mismatch in number of bands: WMX={len(df_wmx)}, SRF={len(df_srf)}")
    
    results = []
    
    # Iterate over bands
    for idx in range(len(df_wmx)):
        # Extract row data (cols wvl1 to wvl7)
        # Assuming format: BND, wvl1...wvl7
        row_wmx = df_wmx.iloc[idx]
        row_srf = df_srf.iloc[idx]
        
        band_id = row_wmx['BND']
        
        # Get the 7 points
        x_data = row_wmx[['wvl1', 'wvl2', 'wvl3', 'wvl4', 'wvl5', 'wvl6', 'wvl7']].values.astype(float)
        y_data = row_srf[['wvl1', 'wvl2', 'wvl3', 'wvl4', 'wvl5', 'wvl6', 'wvl7']].values.astype(float) # Columns are named wvl1..7 in SRF file too? Check file structure.
        # Yes, viewing the file showed header: BND,wvl1,wvl2... for both files.
        
        # Initial guess for optimization
        # Amplitude (a) = max Y
        # Center (x0) = X where Y is max
        # Sigma = approx width
        max_y_idx = np.argmax(y_data)
        init_a = y_data[max_y_idx]
        init_x0 = x_data[max_y_idx]
        init_sigma = 0.02 # Rough guess for HyTES (~15-20nm FWHM -> sigma ~ 0.008, bit wider for safety)
        
        try:
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=[init_a, init_x0, init_sigma], maxfev=5000)
            
            a_fit, x0_fit, sigma_fit = popt
            
            # Calculate FWHM
            # FWHM = 2 * sqrt(2 * ln(2)) * sigma approx 2.35482 * sigma
            fwhm_fit = 2.35482 * abs(sigma_fit)
            
            results.append({
                'band_idx': int(band_id),
                'center_wavelength': x0_fit,
                'fwhm': fwhm_fit,
                'amplitude': a_fit,
                'sigma': sigma_fit
            })
            
        except RuntimeError:
            print(f"Warning: Gaussian fit failed for band {band_id}. Using simple stats.")
            # Fallback: Weighted Average
            x0_est = np.sum(x_data * y_data) / np.sum(y_data)
            # Estimate width roughly
            results.append({
                'band_idx': int(band_id),
                'center_wavelength': x0_est,
                'fwhm': 0.05, # Default fallback
                'amplitude': np.max(y_data),
                'sigma': 0.05/2.355
            })

    output_df = pd.DataFrame(results)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed parameters to: {output_path}")
    output_df.to_csv(output_path, index=False)
    
    print("Preview of processed data:")
    print(output_df[['band_idx', 'center_wavelength', 'fwhm']].head())
    print(output_df[['band_idx', 'center_wavelength', 'fwhm']].tail())

if __name__ == "__main__":
    # Define paths based on project structure
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    WMX_FILE = os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_WMX_v6.csv')
    SRF_FILE = os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_SRF_v6.csv')
    OUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'hytes_real_parameters.csv')
    
    if not os.path.exists(WMX_FILE) or not os.path.exists(SRF_FILE):
        print("Error: Input files not found.")
        print(f"Searched: {WMX_FILE}")
    else:
        process_calibration_files(WMX_FILE, SRF_FILE, OUT_FILE)
