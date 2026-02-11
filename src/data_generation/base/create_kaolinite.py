
import os
import numpy as np

def create_kaolinite_spectrum():
    """
    Creates a synthetic Kaolinite spectrum.
    Base Emissivity: 0.98
    Absorption Feature: Centered at 10.5 um, Sigma=0.5, Depth=0.15.
    range: 8.0 to 12.0 um (step 0.1) similar to quartz example.
    """
    
    # Define wavelength grid similar to quartz (8.0 to 12.0)
    # The quartz example had 0.5 steps, let's use 0.1 for better resolution as requested implied by "sigma=0.5"
    wavelengths = np.arange(8.0, 12.1, 0.1)
    
    base_emis = 0.98
    center = 10.5
    sigma = 0.5
    depth = 0.15
    
    # Gaussian absorption: Depth * exp(-0.5 * ((x - mu) / sigma)^2)
    absorption = depth * np.exp(-0.5 * ((wavelengths - center) / sigma)**2)
    
    emissivity = base_emis - absorption
    
    # Clip to be safe
    emissivity = np.clip(emissivity, 0.0, 1.0)
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'data', 'raw', 'spectral_libraries')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'kaolinite_synthetic.txt')
    
    print(f"Generating Kaolinite spectrum at {output_file}...")
    
    with open(output_file, 'w') as f:
        for w, e in zip(wavelengths, emissivity):
            f.write(f"{w:.2f} {e:.4f}\n")
            
    print("Done.")

if __name__ == "__main__":
    create_kaolinite_spectrum()
