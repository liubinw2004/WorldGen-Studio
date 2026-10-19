# 模型准备

模型权重不在版本库内。这份文档说明每个文件是什么、放哪里、要多少空间，以及容易踩的坑。

先跑一次校验，它会告诉你当前缺什么：

```bat
venv\Scripts\python.exe scripts\check_models.py
```

## 清单

### 必需（文本生成场景）

| 模型 | 体积 | 来源 | 许可 | 落盘位置 |
|---|---|---|---|---|
| `LeoXie/WorldGen`（2 个 LoRA） | ≈ 80 MB | [HuggingFace](https://huggingface.co/LeoXie/WorldGen) | Apache-2.0 | `cache/hf/hub/models--LeoXie--WorldGen/` |
| `haodongli/DA-2`（全景深度） | ≈ 1.3 GB | [HuggingFace](https://huggingface.co/haodongli/DA-2) | 见仓库 | `cache/hf/hub/models--haodongli--DA-2/` |
| `mit-han-lab/svdq-int4-flux.1-dev` | ≈ 4 GB | [HuggingFace](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev) | 见仓库 | `cache/hf/hub/models--mit-han-lab--svdq-int4-flux.1-dev/` |
| `black-forest-labs/FLUX.1-dev`（**仅组件**） | ≈ 9.9 GB | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev)（gated） | **非商用** | `models/FLUX.1-dev/` |

### 可选

| 模型 | 用途 | 体积 | 落盘位置 |
|---|---|---|---|
| `mit-han-lab/svdq-int4-flux.1-fill-dev` | 图生场景 | ≈ 6.3 GB | `cache/hf/hub/` |
| `black-forest-labs/FLUX.1-Fill-dev`（组件） | 图生场景 | ≈ 0.6 GB | `models/FLUX.1-Fill-dev/` |
| `4x-UltraSharp.pth` | 超分（锐利） | 67 MB | `models/sr/` |
| `RealESRGAN_x4plus.pth` | 超分（柔和） | 67 MB | `models/sr/` |
| `shi-labs/oneformer_ade20k_swin_large` | `--inpaint_bg` | 0.9 GB | `cache/hf/hub/` |

## HF 缓存类模型

这几项用 HF 的缓存目录结构，**不要**手工放置，用命令行下载最稳：

```bat
call env\set_env.bat
huggingface-cli download LeoXie/WorldGen
huggingface-cli download haodongli/DA-2
huggingface-cli download mit-han-lab/svdq-int4-flux.1-dev
```

`env\set_env.bat` 会把 `HF_HOME` 指向本项目的 `cache/hf`，这样模型不会散落到用户目录。国内网络先加：

```bat
huggingface-cli download --endpoint https://hf-mirror.com LeoXie/WorldGen
```

## FLUX.1-dev 组件（最容易出错的部分）

这是个 gated 模型，需要先在模型页接受许可，再 `huggingface-cli login`。

**只下组件，不下整个仓库**——`transformer/` 和 `flux1-dev.safetensors` 各约 23.8 GB，而 4bit 主干已经替代了它们：

```bat
huggingface-cli download black-forest-labs/FLUX.1-dev ^
  --include "model_index.json" "scheduler/*" "text_encoder/*" ^
            "text_encoder_2/*" "tokenizer/*" "tokenizer_2/*" "vae/*" ^
  --local-dir models\FLUX.1-dev
```

目标结构：

```
models/FLUX.1-dev/
├── model_index.json
├── scheduler/scheduler_config.json
├── text_encoder/{config.json, model.safetensors}                    CLIP-L   ~246 MB
├── text_encoder_2/{config.json,
│                   model-00001-of-00002.safetensors,
│                   model-00002-of-00002.safetensors,
│                   model.safetensors.index.json}                    T5-XXL   ~9.52 GB
├── tokenizer/{vocab.json, merges.txt, special_tokens_map.json, tokenizer_config.json}
├── tokenizer_2/{spiece.model, tokenizer.json, special_tokens_map.json, tokenizer_config.json}
└── vae/{config.json, diffusion_pytorch_model.safetensors}           ~168 MB
```

> **T5 分片坑**：`text_encoder_2` 的两个分片和 `model.safetensors.index.json` 必须三者齐全，缺一个会在加载时报 missing shard。用上面的 `--include` 一次性下载就不会缺。

### 为什么模型能放在 `models/` 而不是 HF 缓存结构

上游 `pano_gen.py` 把 repo id 写死成字符串 `"black-forest-labs/FLUX.1-dev"`。`scripts/patch_worldgen.py` 把这 4 处改成 `os.environ.get("WORLDGEN_FLUX_DEV_DIR", "<原值>")`——只改字符串的来源，不改任何算法或加载逻辑。`config.py` 已经把这个变量指向 `models/FLUX.1-dev`。

补丁由 `install.bat` 自动应用，可随时回滚：

```bat
venv\Scripts\python.exe scripts\patch_worldgen.py --revert
```

## 图生场景（i2s）

除上面全部内容外还需要：

```bat
call env\set_env.bat
huggingface-cli download mit-han-lab/svdq-int4-flux.1-fill-dev
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev ^
  --include "model_index.json" "scheduler/*" "transformer/config.json" ^
  --local-dir models\FLUX.1-Fill-dev
```

`text_encoder/`、`text_encoder_2/`、`tokenizer*/`、`vae/` 可以直接从 `models\FLUX.1-dev` 复制，省下约 9.8 GB：

```bat
xcopy "models\FLUX.1-dev\text_encoder"   "models\FLUX.1-Fill-dev\text_encoder\" /E /I /Y
xcopy "models\FLUX.1-dev\text_encoder_2" "models\FLUX.1-Fill-dev\text_encoder_2\" /E /I /Y
xcopy "models\FLUX.1-dev\tokenizer"      "models\FLUX.1-Fill-dev\tokenizer\" /E /I /Y
xcopy "models\FLUX.1-dev\tokenizer_2"    "models\FLUX.1-Fill-dev\tokenizer_2\" /E /I /Y
xcopy "models\FLUX.1-dev\vae"            "models\FLUX.1-Fill-dev\vae\" /E /I /Y
```

## 超分权重

放在 `models\sr\`：

- `4x-UltraSharp.pth` — [下载](https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth)
- `RealESRGAN_x4plus.pth` — [下载](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

缺了不影响生成，只是界面里的「画质增强」不可用。

## 许可提醒

FLUX.1-dev 与 FLUX.1-Fill-dev 是**非商用许可**。个人学习、课程作业、作品集这类非商用场景可用；商用需要另行取得授权。其余模型请各自查阅仓库条款。
