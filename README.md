# SelfTok-o

SelfTok-o is a **multimodal interleaved data processing and alignment framework**, designed for **text–image generation, editing, and understanding**.  
It supports **scalable pipelines, configurable modules, and efficient storage** for large-scale research and production use cases.  

---



## 🔔 News

-   **[2025-09-15]** SFT training on BLIP3o 60K.
-   **[2025-09-08]** Start the **pre-training** of the 30m data for BLIP3o. Completed GPU multi-node training and resolved some bugs (qk_norm, the tie in LLM input and output)
-   **[2025-09-03]** Align the metrics of selftok-o (2115.14) and qwen2.5 vl 3B (2214.4) on the MME.
-   **[2025-08-30]** Completed the porting of qwenvl 2.5 to selftok-o, and fully inherit the **understanding capabilities** ().
-   **[2025-08-18]** Completed the selftok transplantation on the bagel and can **generate images** on the imagenet (31d790b).

---

## 🚀 Features

- **Unified Interleaved Pipeline**  
  Handle text, image, and multimodal sequences seamlessly.  

- **Flexible Data Preprocessing**  
  Support for raw `.zip`, `.csv`, `.mat`, and Parquet formats with robust cleaning and filtering modules.  

- **Configurable with YAML**  
  Modular design (e.g., `clip_similarity`, `maniqa`, `vlm_scoring`) controlled via simple YAML configs.  

- **High-Quality Data Filtering**  
  - CLIP-based similarity filtering  
  - Image quality scoring (MANIQA)  
  - VLM-based text–image alignment  

- **Parquet Output**  
  Standardized output format with efficient metadata storage for large-scale ML pipelines.  

- **Extensible**  
  Easy to integrate new scoring models, feature extraction modules, or storage backends.  

---

## 📦 Installation

```bash
git clone https://github.com/your-org/SelfTok-o.git
cd SelfTok-o

# Create environment
conda create -n selftoko python=3.10
conda activate selftoko

# Install dependencies
pip install -r requirements.txt

<!-- pip install flash attention -->

conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 make cmake
# 显式指向 CC/CXX（很重要，确保各 rank 都能找到）
export CC="$(which x86_64-conda-linux-gnu-cc || which gcc)"
export CXX="$(which x86_64-conda-linux-gnu-c++ || which g++)"

pip install diffusers
pip install easydict
pip install einx
pip install timm
pip install -U transformers

```

## 🔥 Train & Eval

### Train

```bash
bash scripts/train.sh
```

### Inference

```bash
python infer_selftok_o.py
```

### Evaluation - Gen
#### GenEval
```shell
source activate /opt/conda/envs/geneval

```

### Evaluation - Understanding
#### MME
```bash
scripts/eval/run_eval_vlm.sh
```




## Pull to github
```bash
git add .
git commit -m "SFT training"
git push origin main
```

