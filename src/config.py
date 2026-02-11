"""
Centralized configuration for IATIR project.
Contains sensor parameters (HyTES, ASTER) and path definitions.
"""
import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENSOR_CONFIG = {
    'hytes': {
        'name': 'hytes',
        'n_bands': 256,
        'wavelength_range': (7.5, 12.0),
        'calibration_file_wmx': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_WMX_v6.csv'),
        'calibration_file_srf': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_SRF_v6.csv'),
        'processed_params_file': os.path.join(PROJECT_ROOT, 'data', 'processed', 'hytes_real_parameters.csv'),
        'description': 'Hyperspectral Thermal Emission Spectrometer (256 bands)'
    },
    'aster': {
        'name': 'aster',
        'n_bands': 5,
        'band_indices': [10, 11, 12, 13, 14], # TIR Bands
        'wavelength_range': (8.0, 12.0),
        'srf_dir': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'aster'),
        # Fallback FWHMs if SRF files are missing (Approximate values in microns)
        'fallback_params': {
            'centers': [8.3, 8.65, 9.1, 10.6, 11.3],
            'fwhms': [0.35, 0.35, 0.35, 0.7, 0.7] 
        },
        'description': 'Advanced Spaceborne Thermal Emission and Reflection Radiometer (5 TIR bands)'
    }
}
