import argparse
import casanova
import sys
import tqdm

from sentence_transformers.sentence_transformer import SentenceTransformer

def compute(texts, model: SentenceTransformer, batch_size = 32):
    embeddings = model.encode(texts, batch_size=batch_size)
    embeddings = embeddings.to_device("cpu")
    embeddings = embeddings.tolist()
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("column", help="Column name to process")
    parser.add_argument("-m", "--model", help="Model name from HuggingFace")
    parser.add_argument("-d", "--device", help="Device to use for computation", default="cpu")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Enable model flag")
    parser.add_argument("-p", "--progress", type=int, help="For progress bar, need number of rows")
    
    args = parser.parse_args()

    column, model, batch, total = args.column, args.model, args.batch, args.progress
    
    file = sys.stdin
    output = sys.stdout

    model = SentenceTransformer(model, device="cpu")

    with casanova.enricher(file, output, add=["embedding"]) as enricher:
        col = enricher.fieldnames.index(column)
        
        accu = []
        with tqdm.tqdm(total=total, desc="Computing embeddings") as progress:
            for row in enricher:
                accu.append(row)
                if len(accu) == batch:
                    gen = compute([it[col] for it in accu], model, batch_size = batch)
                    for (row, vec) in zip(accu, gen):
                        enricher.writerow(row, [vec])
                    accu = []
                    
                    progress.update(batch)
                
            gen = compute([it[col] for it in accu], model, batch_size = batch)
            for (row, vec) in zip(accu, gen):
                enricher.writerow(row, [vec])
            progress.update(len(accu))

if __name__ == "__main__":
    main()
   