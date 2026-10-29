import os

import click
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from xlmexlab.embeddings import EmbeddingExtractor


@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_npz_file", type=str)

def image2embeddings(image_folder: str, output_npz_file: str):
    extractor = EmbeddingExtractor()

    embeddings = []
    filenames = []

    files = sorted(os.listdir(image_folder))
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            full_path = os.path.join(image_folder, fname)
            try:
                emb = extractor.extract_embedding(full_path)
                embeddings.append(emb)
                filenames.append(fname)
                print(f"Extracted embedding for {fname}")
            except Exception as e:
                print(f"Failed to extract embedding for {fname}: {e}")

    if len(embeddings) == 0:
        print("No valid images found. Exiting.")
        return

    embeddings_np = np.stack(embeddings)
    filenames_np = np.array(filenames)

    # Save both in one file
    np.savez(output_npz_file, embeddings=embeddings_np, filenames=filenames_np)
    print(f"\n Saved {len(embeddings)} embeddings to {output_npz_file}.npz")


def main():
    image2embeddings()


if __name__ == "__main__":
    main()
