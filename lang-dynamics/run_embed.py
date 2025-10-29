from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np, argparse, os
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)
from nltk import sent_tokenize

def encode_sentences(sent_list, model, tokenizer):
    toks = tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt", max_length=128).to("cuda")
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, :, 0, :]  # (layers, B, d)
    return hs.permute(1, 0, 2).cpu().numpy()  # (B, L, d)

def process(dataset_name, model_name, save_path):
    dataset = load_dataset(dataset_name, split="train[:200]")  # 可改大小
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().cuda()

    all_vecs, all_layers = [], None
    for item in tqdm(dataset):
        text = item['text'] if 'text' in item else item['dialog']
        sents = sent_tokenize(text)
        if len(sents) < 3: continue
        emb = encode_sentences(sents, model, tokenizer)
        all_vecs.append(emb)
        all_layers = emb.shape[1]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, emb=all_vecs, layers=all_layers)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, default="roberta-base")
    p.add_argument("--save", type=str, required=True)
    args = p.parse_args()
    process(args.dataset, args.model, args.save)
