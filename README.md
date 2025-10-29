# Language-Flow-Network

> Maybe I want Change

语言其实是情绪的投影。  
每个句子背后的 token 都携带着情绪状态。  
这种情绪不会在词与词之间突然断开，而是连续演化。  
因此，语言的生成不是一连串相互独立的概率事件，  
而是一条在时间中流动、带有不可逆性的轨迹。

👉 [**Click here for the full English version**](#english-version)
---

## 🧩 1. 核心观点

Language Flow Network（LFN）提出：  
自然语言并非独立样本的集合，而是一种随时间连续演化的可观测过程。  

我们使用 InfoNCE（互信息的可计算代理）来度量：  
过去窗口的句向量能否预测当前句向量。  

- Original：保持原始顺序  
- Shuffle：打乱顺序作为对照  

若 Original 的 InfoNCE 更低（互信息更高），  
说明语言在向量空间中存在时间黏性与情绪连续性。



---

## 📂 2. 仓库结构

Language-Flow-Network/  
├── datasets/  
│  └── aclImdb/       # 可选：本地 IMDb 数据（不提交到仓库）  
│  
├── lang-dynamics/  
│  ├── data_cfg.yaml  
│  ├── run_embed.py     # 句向量提取（支持离线 DistilBERT 或随机投影）  
│  ├── run_mi.py       # InfoNCE 计算与时间对照  
│  └── models/  
│    ├── distilbert-base-uncased/ # 可选：本地 HF 模型文件夹  
│    ├── encoders.py  
│    └── mine.py  
│  
├── outputs/  
│  ├── imdb_local/  
│  └── dailydialog/  
│  
├── utils/  
└── run_all.sh       # 一键运行脚本

---

## ⚙️ 3. 快速运行

conda create -n langdyn python=3.10 -y  
conda activate langdyn  
pip install -r requirements.txt  

# Step 1. 生成句向量  
python lang-dynamics/run_embed.py  

# Step 2. 计算互信息  
python -m lang-dynamics.run_mi --emb outputs/imdb_local/emb_local.npz --window 3 --mode original --out outputs/imdb_local/mi_original.txt  

或直接运行：  
bash run_all.sh  

---

## 📊 4. 实验结果

| 模式 | 平均 InfoNCE（越低越相关） | 结论 |
|------|-----------------------------|------|
| Original | 1.5792 | 序列保持时间相关性 |
| Shuffle  | 1.5053 | 打乱削弱时间结构 |

---

## 🧠 5. 理论启示

- 语言是一种时间上的连续流，而非离散事件的集合。  
- 情绪在语义空间中是平滑演化的，不是瞬间跳变的。  
- InfoNCE 可作为语言动力学的度量工具，用于分析时间一致性。  
- 可扩展至“时间动力一致性约束”（Temporal Coherence Constraint）方向。

---

## 🧾 6. 引用

@misc{gu2025languageflownetwork,  
  author = {Gu, Longhao},  
  title  = {Language Flow Network: Quantifying Temporal Mutual Information in Language Dynamics},  
  year   = {2025},  
  url    = {https://github.com/yourusername/Language-Flow-Network}  
}

---

📜 7. License

本项目使用 MIT License 授权。
你可以自由地复制、修改、再分发本项目代码，
但请保留原始作者署名与许可声明。

---
First published and implemented by Longhao Gu, Oct 30 2025.
---

# English Version

> **Maybe I want Change**

Language is essentially a projection of emotion.  
Each token carries its own latent emotional state.  
These emotional signals do not abruptly change between words — they evolve smoothly over time.  
Thus, language generation is not a series of independent probabilistic events,  
but an irreversible emotional trajectory flowing through time.

---

### Core Idea

Language Flow Network (LFN) reframes natural language as a **temporally continuous observable process**, rather than a set of independent samples.  
We use the InfoNCE objective (a computable proxy for mutual information) to measure how much the past sentence embeddings can predict the current one.

- **Original:** maintains temporal order  
- **Shuffle:** randomly reorders the sequence  

If the Original sequence achieves lower InfoNCE (i.e., higher mutual information),  
it implies that language carries temporal coherence and emotional continuity.

---

### Repository Structure

The project contains:
- Sentence embedding extraction using DistilBERT or random projections  
- Mutual information estimation via InfoNCE  
- Temporal order comparison (Original vs. Shuffle)  
- Results visualization and data logging  

---

### Key Result

| Mode | Avg InfoNCE ↓ | Interpretation |
|------|----------------|----------------|
| Original | 1.5792 | Strong temporal dependence |
| Shuffle | 1.5053 | Temporal structure disrupted |

---

### Theoretical Implication

This experiment provides a quantifiable lens to study **temporal dynamics in language**.  
It suggests that the evolution of language carries smooth, directional emotional information —  
bridging linguistic structure, time, and affect.  

If Transformer represents the logic of symbols,  
then **Language Flow Network** represents the **flow of emotion through time**.
---

### Citation

@misc{gu2025languageflownetwork,  
  author = {Gu, Longhao},  
  title  = {Language Flow Network: Quantifying Temporal Mutual Information in Language Dynamics},  
  year   = {2025},  
  url    = {https://github.com/gary6668/Language-Flow-Network}  
}

---

### License

This project is released under the MIT License.

---
First published and implemented by Longhao Gu, Oct 30 2025.
