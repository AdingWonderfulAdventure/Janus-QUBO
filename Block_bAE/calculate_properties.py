import argparse
import h5py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import time

# Import PropertyCalculator
try:
    utils_Qmol_FM_DIR = str(Path(__file__).resolve().parent.parent / "Qmol_FM")
    sys.path.insert(0, utils_Qmol_FM_DIR)
    from utils_Qmol_FM import create_property_calculator
except ImportError:
    raise ImportError("Cannot import PropertyCalculator, please check utils_Qmol_FM.py path.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def deduplicate_h5(src_path: Path, out_path: Path):
    """Deduplicate and rebuild a new HDF5 file."""
    with h5py.File(src_path, "r") as hf:
        smiles_bytes = hf["smiles"][:]
        smiles_list = [s.decode("utf-8") for s in smiles_bytes]

        log.info(f"Total samples: {len(smiles_list):,}")
        unique_smiles, unique_indices = np.unique(smiles_list, return_index=True)
        log.info(f"Unique samples: {len(unique_smiles):,} (removed {len(smiles_list)-len(unique_smiles):,} duplicates)")

        # Create new file
        with h5py.File(out_path, "w") as hf_new:
            # Trim all top-level keys
            for key in hf.keys():
                if key == "properties":
                    continue  # properties will be recalculated
                log.info(f"Copying dataset: {key}")
                data = hf[key][:]
                hf_new.create_dataset(key, data=data[unique_indices], compression="gzip")

    return unique_smiles


def compute_and_write_properties(h5_path: Path, n_jobs: int, batch_size: int):
    log.info("=" * 60)
    log.info("Starting RDKit Property Calculation for HDF5 File (Large-Scale Mode)")
    log.info(f"Target HDF5 file: {h5_path}")
    log.info("=" * 60)

    # 1. Initialize PropertyCalculator
    property_calculator = create_property_calculator(n_jobs=n_jobs)

    # 2. Calculate properties directly from HDF5 file
    h5_path = Path(h5_path)
    log.info("Starting property calculation directly from HDF5 file...")
    start_time = time.time()

    try:
        properties_dict = property_calculator.calculate_from_h5(
            str(h5_path),
            batch_size=batch_size
        )
    except FileNotFoundError as e:
        log.error(f"Error: {e}")
        return
    except ValueError as e:
        log.error(f"HDF5 file content error: {e}")
        return

    elapsed = time.time() - start_time
    log.info(f"Property calculation complete! Elapsed: {elapsed:.2f} seconds.")

    if not properties_dict:
        log.warning("Calculation result is empty, no data written to HDF5 file.")
        return

    # 3. Write computed properties back to HDF5 file
    log.info(f"Preparing to write properties for {len(properties_dict)} samples back to HDF5 file...")

    # Re-read smiles to ensure order alignment
    try:
        with h5py.File(h5_path, "r") as hf:
            smiles_bytes = hf["smiles"][:]
        smiles_list_for_reindex = [s.decode("utf-8") for s in smiles_bytes]
    except Exception as e:
        log.error(f"Cannot re-read SMILES list for alignment: {e}")
        return

    try:
        props_df = pd.DataFrame.from_dict(properties_dict, orient="index")
        props_df = props_df.reindex(smiles_list_for_reindex)

        if props_df.isnull().values.any():
            log.warning("Some property values are missing (NaN) after alignment. This may indicate SMILES key mismatch.")
            props_df.fillna(value=property_calculator.PENALTY_VALUES, inplace=True)
    except Exception as e:
        log.error(f"Error creating or reindexing DataFrame: {e}")
        return

    log.info(f"Data aligned. Preparing to write {len(props_df.columns)} properties...")

    try:
        with h5py.File(h5_path, "a") as hf:
            if "properties" in hf:
                log.warning("'properties' group already exists, will be fully overwritten for data consistency.")
                del hf["properties"]

            props_group = hf.create_group("properties")

            for prop_name in props_df.columns:
                data_to_write = props_df[prop_name].values.astype(np.float32)
                log.info(f" - Writing property: '{prop_name}' (shape={data_to_write.shape}, dtype={data_to_write.dtype})")
                props_group.create_dataset(prop_name, data=data_to_write, compression="gzip")

    except Exception as e:
        log.error(f"Error writing to HDF5 file: {e}")
        return

    log.info("=" * 60)
    log.info("Process completed successfully!")
    log.info(f"All computed properties have been added/updated in the 'properties' group of '{h5_path}'.")
    log.info("=" * 60)


def main(args):
    src_path = Path(args.h5_path)
    out_path = Path(args.out_path)

    unique_smiles = deduplicate_h5(src_path, out_path)
    compute_and_write_properties(out_path, args.n_jobs, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate and rebuild HDF5 file (including property calculation)")
    parser.add_argument("--h5_path", type=str, required=True, help="Original HDF5 file")
    parser.add_argument("--out_path", type=str, required=True, help="Output path for deduplicated HDF5 file")
    parser.add_argument("--batch_size", type=int, default=50000, help="Number of SMILES per batch")
    parser.add_argument("--n_jobs", type=int, default=160, help="Number of parallel processes")
    args = parser.parse_args()

    main(args)
