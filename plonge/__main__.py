import argparse
import casanova
import sys
import torch
import tqdm

from sentence_transformers.sentence_transformer import SentenceTransformer

def compute(texts, model: SentenceTransformer, batch_size=32, device=None):
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        device=device,
        convert_to_numpy=True,
        convert_to_tensor=False,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def clear_device_cache(device):
    if not device:
        return

    device_type = str(device).split(":", 1)[0]

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def write_batch(rows, col, model, enricher, batch_size, device):
    if not rows:
        return 0

    vectors = compute([row[col] for row in rows], model, batch_size=batch_size, device=device)
    try:
        for row, vec in zip(rows, vectors):
            enricher.writerow(row, [vec])
    finally:
        del vectors
        clear_device_cache(device)

    return len(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("column", help="Column name to process")
    parser.add_argument("-m", "--model", help="Model name from HuggingFace")
    parser.add_argument("-d", "--device", help="Device to use for computation", default="cpu")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Enable model flag")
    parser.add_argument("-p", "--progress", type=int, help="For progress bar, need number of rows")
    
    args = parser.parse_args()
    
    file = sys.stdin
    output = sys.stdout

    model = SentenceTransformer(args.model, device=args.device)

    with casanova.enricher(file, output, add=["embedding"], prebuffer_bytes=1000) as enricher:
        col = enricher.fieldnames.index(args.column)
        total = enricher.total
        accu = []
        with tqdm.tqdm(total=args.progress if args.progress else total, desc="Computing embeddings") as progress:
            for row in enricher:
                accu.append(row)
                if len(accu) == args.batch:
                    written = write_batch(accu, col, model, enricher, args.batch, args.device)
                    accu = []

                    progress.update(written)
                
            written = write_batch(accu, col, model, enricher, args.batch, args.device)
            progress.update(written)

if __name__ == "__main__":
    main()
   