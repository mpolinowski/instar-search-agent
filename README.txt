cd ~/Projects/instar-search-agent

# Stage 1 — dataset creation (already built & green; re-run only if data/ or SPEC_Dataset_Creation.md changed)
.venv/bin/python build_dataset.py --data ./data --out ./hf_dataset

# Stage 2 — embedding build (first run downloads the model: HF primary → hf-mirror fallback)
.venv/bin/python run_embeddings.py \
    --dataset ./hf_dataset \
    --modality image_text \
    --multi-image primary \
    --batch-size 8 \
    --embed-file ./embeddings/instar_docs_v1_image_text.safetensors


+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.57.04              KMD Version: 610.57.04     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090 ...    Off |   00000000:65:00.0  On |                  N/A |
| N/A   73C    P0             94W /   95W |   21475MiB /  24463MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1797      G   Hyprland                                  6MiB |
|    0   N/A  N/A         2860978      C   .venv/bin/python                      21402MiB |
+-----------------------------------------------------------------------------------------+
