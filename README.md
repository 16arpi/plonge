# Plonge

Use `plonge` to compute the embeddings of all the cells of a given CSV file. Inspired from [`xan`](https://github.com/medialab/xan) and [`yuzu`](https://github.com/medialab/yuzu). Based on [`casanova`](https://github.com/medialab/casanova) and [`sentence-transformers`](https://sbert.net/)

```
usage: plonge [-h] [-m MODEL] [-d DEVICE] [-b BATCH] [-p PROGRESS] column

positional arguments:
  column                Column name to process

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name from HuggingFace
  -d DEVICE, --device DEVICE
                        Device to use for computation
  -b BATCH, --batch BATCH
                        Enable model flag
  -p PROGRESS, --progress PROGRESS
                        For progress bar, need number of rows
```