import argparse
import json
import torch
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import selfies as sf
from rdkit import Chem, RDLogger

# Suppress verbose RDKit logs
RDLogger.DisableLog('rdApp.*')

# ========================= Key Import =========================
try:
    from train_gruencoder_transformerdecoder import LitBlockbAE_Transformer
except ImportError:
    print("Error: Cannot import 'LitBlockbAE_Transformer'...")
    # Assumes the file is in the same directory; if not, set PYTHONPATH correctly
    exit()
# ==========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ========================= Data Preprocessing Function =========================
# Function also returns the list of successfully processed cleaned_smi
def preprocess_smiles_to_ids(smiles_list: list[str], token_to_id: dict, max_len: int) -> tuple[list[torch.Tensor], list[int], list[str]]:
    tokenized_sequences = []
    valid_indices = []
    successful_cleaned_smiles = []

    sos_id, eos_id = token_to_id['<sos>'], token_to_id['<eos>']

    for i, smi in enumerate(smiles_list):
        cleaned_smi = None
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue

            cleaned_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            selfies_seq = sf.encoder(cleaned_smi)
            tokens = list(sf.split_selfies(selfies_seq))

            if len(tokens) > max_len - 2:
                tokens = tokens[:max_len - 2]

            token_ids = [sos_id]
            has_unknown = False
            for t in tokens:
                token_id = token_to_id.get(t)
                if token_id is None:
                    log.warning(f"Unknown token! SMILES: '{smi}' -> Cleaned: '{cleaned_smi}' -> Token: '{t}'")
                    has_unknown = True
                    break
                token_ids.append(token_id)

            if has_unknown:
                continue

            token_ids.append(eos_id)

            # Record results only after all checks pass
            tokenized_sequences.append(torch.tensor(token_ids, dtype=torch.long))
            valid_indices.append(i)
            successful_cleaned_smiles.append(cleaned_smi)

        except Exception as e:
            log.warning(f"Exception while processing '{smi}' (Cleaned: '{cleaned_smi}'): {e}")
            continue

    return tokenized_sequences, valid_indices, successful_cleaned_smiles

# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Encode SMILES into deterministic Latent Codes using a trained model.")
    parser.add_argument("--ckpt_path", required=True, help="Model checkpoint (.ckpt) file path.")
    parser.add_argument("--vocab_path", required=True, help="Vocabulary (vocabulary.json) file path.")
    parser.add_argument("--input_smiles_file", required=True, help="Text file containing SMILES (one per line).")
    parser.add_argument("--output_h5_path", required=True, help="Output HDF5 file path for latent codes.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding.")
    args = parser.parse_args()

    # --- 1. Load model and vocabulary ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading model to {device}...")

    model_module = LitBlockbAE_Transformer.load_from_checkpoint(args.ckpt_path, map_location=device)
    model = model_module.model
    model.eval()

    log.info(f"Loading vocabulary from {args.vocab_path}...")
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
    token_to_id = vocab_data['stoi']
    pad_id = token_to_id['<pad>']

    # --- 2. Read SMILES ---
    with open(args.input_smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    log.info(f"Read {len(smiles_list)} SMILES from {args.input_smiles_file}.")

    # Create a list of the same length as input to store all cleaned_smi; failed ones remain empty
    cleaned_smiles_full_list = [''] * len(smiles_list)

    all_latent_codes_list = []
    total_valid_smiles = 0
    log.info(f"Starting batch deterministic encoding (batch_size={args.batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="Encoding Batches"):
            batch_smiles = smiles_list[i : i + args.batch_size]

            tokenized_sequences, valid_indices_in_batch, cleaned_smiles_batch = preprocess_smiles_to_ids(
                batch_smiles, token_to_id, args.max_len
            )

            total_valid_smiles += len(valid_indices_in_batch)

            # Fill successful cleaned_smi into the correct positions of the full list
            for local_idx, original_batch_idx in enumerate(valid_indices_in_batch):
                global_idx = i + original_batch_idx
                cleaned_smiles_full_list[global_idx] = cleaned_smiles_batch[local_idx]

            latent_codes_batch_full = torch.zeros(len(batch_smiles), model.latent_dim, dtype=torch.float32)

            if tokenized_sequences:
                padded_batch = torch.nn.utils.rnn.pad_sequence(
                    tokenized_sequences, batch_first=True, padding_value=pad_id
                ).to(device)

                latent_codes_batch = model.get_deterministic_binary_representation(padded_batch)

                for local_idx, original_batch_idx in enumerate(valid_indices_in_batch):
                    latent_codes_batch_full[original_batch_idx] = latent_codes_batch[local_idx].cpu()

            all_latent_codes_list.append(latent_codes_batch_full)

    # --- 4. Save results ---
    log.info(f"All batches processed. {total_valid_smiles} out of {len(smiles_list)} SMILES were successfully processed.")

    if all_latent_codes_list:
        final_latents = torch.cat(all_latent_codes_list, dim=0).numpy()
        log.info(f"Encoding complete. Final latent codes tensor shape: {final_latents.shape}")

        output_path = Path(args.output_h5_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('latent_codes', data=final_latents.astype(np.int8), compression='gzip')

            # Save aligned cleaned_smi list instead of original smiles_list
            string_dt = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset('smiles', data=cleaned_smiles_full_list, dtype=string_dt, compression='gzip')

        log.info(f"Latent codes and {len(cleaned_smiles_full_list)} corresponding standardized SMILES saved to: {output_path}")
    else:
        log.warning("No successful encoding results, output file not created.")

if __name__ == '__main__':
    main()
