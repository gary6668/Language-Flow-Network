#!/bin/bash
echo "==== 🧠 Language Flow Network (Layer Version) ===="

echo "---- Step 1: Embedding ----"
PYTHONPATH=. python lang-dynamics/run_embed.py

echo "---- Step 2: MI original ----"
PYTHONPATH=. python -m lang-dynamics.run_mi \
  --emb outputs/imdb_local/emb_local.npz \
  --window 3 \
  --mode original \
  --out outputs/imdb_local/mi_original.txt

echo "---- Step 3: MI shuffle_layer ----"
PYTHONPATH=. python -m lang-dynamics.run_mi \
  --emb outputs/imdb_local/emb_local.npz \
  --window 3 \
  --mode shuffle_layer \
  --out outputs/imdb_local/mi_shuffle.txt

echo "---- ✅ DONE ----"
echo "Results:"
cat outputs/imdb_local/mi_original.txt
cat outputs/imdb_local/mi_shuffle.txt