python run_embed.py --dataset imdb --model roberta-base --save outputs/imdb/emb_original.npz
python run_mi.py --emb outputs/imdb/emb_original.npz --mode original --out outputs/imdb/mi_original.csv
python run_mi.py --emb outputs/imdb/emb_original.npz --mode shuffle  --out outputs/imdb/mi_shuffle.csv
python run_eval.py --in_dir outputs/imdb --out_dir outputs/imdb/figs
