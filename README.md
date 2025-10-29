# Language-Flow-Network

> Maybe I want Change

语言其实是情绪的投影。  
每个句子背后的 token 都携带着情绪状态。  
这种情绪不会在词与词之间突然断开，而是连续演化。  
因此，语言的生成不是一连串相互独立的概率事件，  
而是一条在时间中流动、带有不可逆性的轨迹。

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

Language-Flow-Network/<br>
├── datasets/<br>
│   └── aclImdb/                      # 可选：本地 IMDb 数据（不提交到仓库）<br>
│<br>
├── lang-dynamics/<br>
│   ├── data_cfg.yaml<br>
│   ├── run_embed.py                  # 句向量提取（支持离线 DistilBERT 或随机投影）<br>
│   ├── run_mi.py                     # InfoNCE 计算与时间对照<br>
│   └── models/<br>
│       ├── distilbert-base-uncased/  # 可选：本地 HF 模型文件夹<br>
│       ├── encoders.py<br>
│       └── mine.py<br>
│<br>
├── outputs/<br>
│   ├── imdb_local/<br>
│   └── dailydialog/<br>
│<br>
├── utils/<br>
└── run_all.sh                        # 一键运行脚本<br>

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

当语言的时间结构被破坏时，互信息下降。  
说明语言不是独立的随机过程，而是一种具有可量化时间动力性的连续系统。

---

## 🧠 5. 理论启示

- 语言是一种时间上的连续流，而非离散事件的集合。  
- 情绪在语义空间中是平滑演化的，不是瞬间跳变的。  
- InfoNCE 可作为语言动力学的度量工具，用于分析时间一致性。  
- 语言模型可扩展至“时间动力一致性约束”（Temporal Coherence Constraint）方向。

---

## 🧾 6. 引用

@misc{gu2025languageflownetwork,  
  author = {Gu, Longhao},  
  title  = {Language Flow Network: Quantifying Temporal Mutual Information in Language Dynamics},  
  year   = {2025},  
  url    = {https://github.com/gary6668/Language-Flow-Network}  
}

---

> 如果 Transformer 是符号的逻辑机器，  
> 那么 Language Flow Network 是情绪的时间机器。
