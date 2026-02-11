import os
import pandas as pd

# --- Constants ---
# Project Root is 4 levels up: src/data_generation/base/clean.py -> base -> data_gen -> src -> IATIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LIBRARY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'spectral_libraries')
SUMMARY_LOG_PATH = os.path.join(LIBRARY_DIR, 'summary_log.csv')
CLEANED_LOG_PATH = os.path.join(LIBRARY_DIR, 'summary_log_cleaned.csv')

# Validation List (Golden Targets - Endmembers Puros)
LISTA_DE_VALIDACAO = [
    'quartz_mineral_silicate_tectosilicate_coarse_tir_quartz_1_jhu_nicolet_spectrum.txt',
    'calcite_mineral_carbonate_none_coarse_tir_calcite_1_jhu_nicolet_spectrum.txt',
    'kaolinite_mineral_silicate_phyllosilicate_coarse_tir_kaolinite_2_jhu_nicolet_spectrum.txt',
    'feldspar_mineral_silicate_tectosilicate_coarse_tir_anorthite_1_jhu_nicolet_spectrum.txt',
    'urban_concrete_manmade_concrete_constructionconcrete_solid_all_0598uuucnc_jhu_becknic_spectrum.txt',
    'urban_asphalt_manmade_road_pavingasphalt_solid_all_0095uuuasp_jhu_becknic_spectrum.txt',
    'vegetation_green_leaf_nonphotosyntheticvegetation_bark_abies_concolor_tir_vh311_ucsb_nicolet_spectrum.txt'
]

def clean_and_tag_library() -> None:
    """
    Cleans the spectral library log and tags samples for Training or Validation.
    
    Processing:
    1. Removes files marked as 'Error' or 'SKIPPED'.
    2. Identifies 'Golden Targets' (Endmembers) and tags them as 'teste/validacao'.
    3. Tags all remaining files as 'treino'.
    4. Saves the cleaned log to 'summary_log_cleaned.csv'.
    """
    if not os.path.exists(SUMMARY_LOG_PATH):
        print(f"Error: {SUMMARY_LOG_PATH} not found.")
        return

    print(f"Loading log from: {SUMMARY_LOG_PATH}")
    df = pd.read_csv(SUMMARY_LOG_PATH)
    
    # Add new column 'situacao'
    if 'situacao' not in df.columns:
        df['situacao'] = ''

    deleted_count = 0
    validation_count = 0
    train_count = 0

    for index, row in df.iterrows():
        status = row['Status']
        filename = row['New']
        
        # Check if filename is valid string (not NaN)
        if not isinstance(filename, str) or filename.lower() == 'skipped':
             # Try to delete original if possible, but mainly mark as deleted/skipped
             df.at[index, 'situacao'] = 'deletado'
             continue

        file_path = os.path.join(LIBRARY_DIR, filename)

        if status != 'Success':
            # Try to delete the file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted error file: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
            df.at[index, 'situacao'] = 'deletado'
        else:
            # Check if it's in validation list
            if filename in LISTA_DE_VALIDACAO:
                df.at[index, 'situacao'] = 'teste/validacao'
                validation_count += 1
            else:
                df.at[index, 'situacao'] = 'treino'
                train_count += 1

    print(f"Saving cleaned log to: {CLEANED_LOG_PATH}")
    df.to_csv(CLEANED_LOG_PATH, index=False)
    print(f"Saved cleaned log to: {CLEANED_LOG_PATH}")
    print(f"Summary:")
    print(f"  Deleted/Skipped: {deleted_count}")
    print(f"  Validation (Golden Targets): {validation_count}")
    print(f"  Training: {train_count}")

if __name__ == "__main__":
    clean_and_tag_library()
