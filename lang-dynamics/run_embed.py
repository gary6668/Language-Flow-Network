# Author: Longhao Gu (Zhejiang University)
# Original concept: Language Flow Network, Oct 30 2025

import os, re, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

# ==== 配置 ====
base_dir = "datasets/aclImdb"
output_dir = "outputs/imdb_local"
os.makedirs(output_dir, exist_ok=True)

sample_n = 200
# 本地模型路径（你现在的）
model_name = "models/distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_sentences(text):
    sents = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [s for s in sents if len(s.split()) >= 5]

def load_local_imdb(base_dir, n=200):
    texts = []
    for label in ['pos', 'neg']:
        folder = os.path.join(base_dir, 'train', label)
        files = random.sample(os.listdir(folder), n//2)
        for f in files:
            with open(os.path.join(folder, f), 'r', encoding='utf-8') as fp:
                texts.append(fp.read())
    return texts

def encode_sentence_list(sents, model, tokenizer, device):
    toks = tokenizer(sents, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, :, 0, :]  # (layers, B, dim)
    return hs.permute(1, 0, 2).cpu().numpy()  # (B, L, dim)

def main():
    print("📘 Loading IMDb texts ...")
    texts = load_local_imdb(base_dir, n=sample_n)
    print("📦 Loading local DistilBERT model ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_vecs = []
    print(f"🚀 Encoding {len(texts)} samples ...")
    for i, text in enumerate(texts):
        sents = split_sentences(text)
        if len(sents) < 1: continue
        emb = encode_sentence_list(sents, model, tokenizer, device)
        all_vecs.append(emb.mean(axis=0))
        print(f"[{i+1}/{len(texts)}] done ({len(sents)} sentences)")

    np.savez_compressed(os.path.join(output_dir, "emb_local.npz"), emb=np.array(all_vecs, dtype=object))
    print(f"\n✅ Done. Saved to {output_dir}/emb_local.npz")

if __name__ == "__main__":
    main()
