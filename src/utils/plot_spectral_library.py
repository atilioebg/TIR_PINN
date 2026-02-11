import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_library() -> None:
    """
    Plots a random sample of spectral curves from the library.
    
    Reads data from 'data/raw/spectral_libraries', selects up to 50 random files,
    and saves the plot to 'outputs/graphs/spectral_library_curves.png'.
    """
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LIB_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'spectral_libraries')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(LIB_DIR, "*.txt"))
    if not files:
        print(f"No files found in {LIB_DIR}")
        return

    MAX_PLOTS = 50
    # Shuffle or just take first N? Let's take a sample if too many
    if len(files) > MAX_PLOTS:
        print(f"Too many files ({len(files)}). Plotting a random sample of {MAX_PLOTS}.")
        files = np.random.choice(files, MAX_PLOTS, replace=False)
    
    plt.figure(figsize=(10, 6))
    
    for fpath in files:
        name = os.path.basename(fpath).replace('.txt', '')
        
        # Robust read
        try:
            # Handle new format with header "Wavelength Emissivity"
            # skiprows=1 is risky if header is missing, so use comment='#' and pandas which handles headers better
            df = pd.read_csv(fpath, sep=r'\s+', comment='#')
            
            # If header is text, it's used as columns. If header is numbers, it's row 0.
            # Our curated files have "Wavelength Emissivity" as header.
            # If pd.read_csv sees text header, columns are strings.
            # If we enforce numeric, we drop non-numeric rows.
            
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            # If empty after dropna (maybe entire file was header?), try reading with header=None?
            # But curated files are consistent.
            
            if df.shape[1] < 2:
                 # Try skipping first row manualy via skiprows=1 equivalent (reading as header=0 did this effectively if columns are names)
                 pass

            wavs = df.iloc[:, 0].values
            emis = df.iloc[:, 1].values
            
            plt.plot(wavs, emis, label=name, linewidth=2, alpha=0.7)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            
    plt.title("Sample Spectral Library Curves (Emissivity)", fontsize=14, fontweight='bold')
    plt.xlabel(r"Wavelength ($\mu m$)")
    plt.ylabel("Emissivity")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if len(files) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        print("Legend disabled due to high number of plots.")
    plt.ylim(0, 1.05)
    
    out_path = os.path.join(OUTPUT_DIR, "spectral_library_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")
    
    # Open interactive window
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    plot_library()
