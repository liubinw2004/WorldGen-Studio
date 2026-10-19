# 偏差与决策记录（DEVIATIONS）

本文件记录复现 `ZiYang-xie/WorldGen` 过程中，与原始设计/预估不一致之处，以及对应的已做决策与依据。

> 说明：文中出现的 `D:\mm` 是作者当时的工作目录。项目已改为从 `config.py` 所在位置自动推导根目录，任何路径都可以，阅读时把 `D:\mm` 理解成「项目根目录」即可。

---

## 关键技术决策（速览，T05 要求逐项覆盖）

1. **无 conda / 无 uv**：仅用系统 Python 3.12 创建 `D:\mm\venv`，全程 `pip`（走腾讯/阿里云镜像）。见 D-02。
2. **Python 3.12**（3.12.7）：torch/torchvision/nunchaku 均取 cp312 win_amd64 轮子。
3. **pytorch3d 用本地 shim**（`vendor/pytorch3d_shim`，纯 torch 实现 `matrix_to_quaternion`/`quaternion_to_matrix`，wxyz 约定，数值误差 < 1e-5），**绝不在 Windows 编译 pytorch3d**。见 D-03。
4. **基座模型经 ModelScope 获取**：gated 的 `black-forest-labs/FLUX.1-dev` 组件由 ModelScope `AI-ModelScope/FLUX.1-dev` 组装（排除 23.8GB 原生 transformer）。见 D-10、D-11。
5. **FLUX.1-dev 非商用许可**：本复现为**学生教育/个人非商用**，未用于商业用途。同 D-10。

> 其余条目 D-01 ~ D-19 为各阶段踩坑与修复的详细记录。

---

## 完整偏差表（D-01 ~ D-19）

| 编号 | 阶段 | 类型 | 现象 / 问题 | 决策 / 处理 |
|---|---|---|---|---|
| **D-01** | T02 | 依赖 | `--recursive` 会拉取 `pytorch3d`/`ml-sharp` 等重子模块 | 选择性 `submodule update --init viser DA-2`，主动 deinit 其余 |
| **D-02** | T02 | 依赖 | `pyproject.toml` 硬钉 `linux_x86_64` 的 nunchaku 轮子 | `pip install -e WorldGen --no-deps` + 独立 `requirements.win.txt` |
| **D-03** | T02 | 依赖 | Windows 无法编译 pytorch3d | `vendor/pytorch3d_shim` 纯 torch 替代（仅 2 函数） |
| **D-04** | T03 | C 盘 | 仅设 `TMP`/`TEMP` 时 pip 仍把临时文件写 C 盘 | `ENV_VARS` 显式加 `TMPDIR=D:\mm\tmp` |
| **D-05** | T02 | 源 | aliyun `pytorch-wheels` 实为 HTML 落地页且限速 | 改用官方 `download.pytorch.org/whl/cu128` |
| **D-06** | T03 | 网络 | hf-mirror 经本地代理 502 | 脚本内 pop `HTTP(S)_PROXY` + `NO_PROXY` 白名单 |
| **D-07** | T03 | 缓存 | Windows 下 HF 生成 0 字节 snapshot 占位 | `repair_repo_cache()` 依 LFS oid 回填（9 条） |
| **D-08** | T03 | 缓存 | `refs/main` 带 CRLF 致离线解析失败 | `newline=""` 写入单行 sha（无换行） |
| **D-09** | T03 | 预估 | `svdq-int4` 实测 6.3GB（预估 4GB） | 以实测为准，tier1 修正为 ≈17.2GB |
| **D-10** | T03 | 许可 | FLUX.1-dev gated，且为非商用许可 | ModelScope 分发；**学生教育/个人非商用** |
| **D-11** | T03 | 方案 | 是否需要改 WorldGen 源码 | 方案 A（缓存镜像）生效，**未改源码** |
| **D-12** | T02 | 依赖 | `da2` 实际需 `utils3d` + `matplotlib` | 补充安装（utils3d 以 git 3913c65 --no-deps） |
| **D-13** | T04 | 上游 bug | viser 客户端运行时自动构建 Windows 报 `WinError 193` | 用 `npx.cmd` 手动 `yarn install`+`vite build` |
| **D-14** | T04 | 依赖 | `viser` 是 git URL 依赖，GitHub 受限 | 从 `submodules/viser` editable 安装（0.2.23） |
| **D-15** | T04 | 运行 | gated FLUX 需离线解析 | `run_demo.{bat,sh}` 置 `HF_HUB_OFFLINE=1` |
| **D-16** | T04 | 依赖 | 声明 `xformers`，但源码无 import | 不安装；diffusers 回退 SDPA |
| **D-17** | T04 | 约束 | 浏览器截图需 C 盘装 Chromium | 遵守仅 D 盘，用 `splat.ply`+`HTTP 200` 佐证 |
| **D-18** | T05 | 幂等 | `run_bootstrap` 的 done 判定受 Python CRLF 输出影响而失效 | `reload_done_keys` 经 `tr -d '\r'` 去 CR，幂等恢复 |
| **D-19** | T05 | 存储 | 曾疑 `models/FLUX.1-dev` 与 cache 快照各占 9.26GB（冗余） | 实为**硬链接共享同一 inode**（link count=3），**无冗余**；`du -sh /d/mm`=30GB 佐证 |
| **D-20** | T07 | 上游 bug | i2s `map_image_to_pano→batch_nearest_dot` 稠密相似度矩阵 `(8192×2,091,012)=63.8GiB` 必 OOM | 运行时 shim 分块（chunk=64，数值等价），瞬时 510MiB；**未改源码** |
| **D-21** | T06 | 基座/许可 | FLUX.1-Fill-dev 亦 gated；组件经 ModelScope 获取 | ModelScope `AI-ModelScope/FLUX.1-Fill-dev`；**学生教育/个人非商用**（同 D-10） |
| **D-22** | T06 | 优化 | T5-XXL(9.5GB)/CLIP-L 与 dev 逐字节相同 | sha256 校验后**硬链接复用**，省 9.10GB；Fill VAE(335MB) 与 dev 不同故单独下载 |
| **D-23** | T07 | 性能/显存 | i2s 下 T5 编码卡死（transformer 6.6GB+depth 1.3GB 占满，T5 被换页到 CPU） | shim 在 encode 期把 transformer/depth 移 CPU，腾 9.9GB 给 T5；50 步 ~36s |
| **D-24** | T07 | 显存/稳定性 | QA 复验 3 次挂 2 次：`pre-encode` 之后静默永久卡死（栈在 `vae.encode`/`randn_tensor.to(device)`，GPU 余量 ~125MiB、util 100%、无异常无 OOM） | **①** i2s 在 12GB 上必须 `-r 1024`（终态全景恒为 2048×1024；`-r 1600` 必 OOM）；**②** `encode_prompt` 返回后必须**主动把 `text_encoder`/`text_encoder_2` 撤离 GPU**——`enable_model_cpu_offload` 的 `model_cpu_offload_seq` 是**惰性驱逐**，我提前下掉 transformer 反而让 9.5GB T5 无人驱逐 → WDDM 换页静默卡死；**③** 入口加看门狗（无进展 300s → exit 124 + 自诊断），杜绝无限等待 |

---

## C 盘口径说明（准确表述，经 QA 独立复验）

> **WorldGen 复现管线自身对 C 盘净写入为 0** —— 所有缓存/临时/解释器/模型均落 D 盘，已由 `verify_c_drive.ps1`（`logs/c_drive_report.json`）与 QA 独立复验**双重确认**。整机层面存在 **≤0.5GB 的系统与宿主工具 churn**，**与本项目无关**。

**实测分项归因**（`C:\Users\Han\AppData\Local\Temp`，预算 0.5GB）：
| 目标 | baseline bytes | now bytes | delta |
|---|---|---|---|
| `C:\Users\Han\.cache` | 0 | 0 | **0** |
| `C:\Users\Han\AppData\Local\Temp` | 13,726,440,354 | 14,185,896,564 | +459,456,210 B = **0.428 GB** |
| `C:\Users\Han\AppData\Local\pip` | 2,746,948,304 | 2,746,948,304 | **0** |
| `C:\Users\Han\AppData\Roaming\uv` | 0 | 0 | **0** |

- C Temp 唯一显著增长：`workbuddy-update-x64\WorkBuddy-Setup-5.8.3.41334564.exe` = **441,094,104 B**（宿主应用自身自动更新包）；
  其余为宿主/Agent 运行时微小输出（`codebuddy-Han/.../agent-*.output`、sogou 日志等，均 ≤13MB）。
- **WorldGen 管线写入 C Temp = 0**（`TMPDIR=D:\mm\tmp` 生效，pip/HF/npm/yarn 缓存全在 D）。
- 注：`verify_c_drive.ps1` 的 0.5GB 判定阈值**未做任何放宽**——变的是文档表述（诚实分项归因优于一个好看的总数），验证逻辑不变。

**独立复验**：`D:\mm\qa\QA-independent-verification-report.md`（QA 独立解析产物/C 盘/端到端重跑/离线声明，6/6 PASS，判定 NoOne 需路由）。

---

## T01 / T02

### D-01 子模块选择性初始化
- 原始仓库 `--recursive` 会拉取 `pytorch3d`、`ml-sharp` 等重子模块。
- **决策**：不递归 clone，改为选择性 `git submodule update --init submodules/viser submodules/DA-2`，并主动 deinit `pytorch3d`/`ml-sharp`。
- **依据**：本部署仅在 Windows 使用两个纯数学函数，绝不在 Windows 编译 pytorch3d。
- **备注（git status 说明）**：因此 `git status` 中 `submodules/ml-sharp` 显示为已删除、`submodules/pytorch3d` 缺失，**属有意为之（按设计选择性跳过）**，非误删；`submodules/viser`、`submodules/DA-2` 保留。后续看到此状态无需处理。

### D-02 依赖安装策略
- **决策**：`pip install -e D:\mm\WorldGen --no-deps` + 独立 `requirements.win.txt`（win_amd64 轮子，nunchaku/torch 以 `file:///D:/mm/wheels` 本地安装）。
- **依据**：规避源码安装 pytorch3d 与 linux-only 依赖；断言零 `linux_x86_64` 包。

### D-03 pytorch3d 本地 shim
- **决策**：`vendor/pytorch3d_shim` 提供纯 torch 的 `matrix_to_quaternion` / `quaternion_to_matrix`（wxyz 约定），经 `.pth` 注册到 sys.path。
- **依据**：WorldGen 仅用这两函数；shim 与 scipy 数值误差 3.3e-16 / 6.7e-16（<1e-5，通过）。

---

## T03

### D-04 临时目录必须落 D 盘（TMPDIR 覆盖 TMP/TEMP）
- **现象**：仅设 `TMP`/`TEMP` 时，pip 仍把 ~3GB 解包临时文件写到 C 盘。
- **决策**：在 `config.py` 的 `ENV_VARS` 中显式加入 `TMPDIR=D:\mm\tmp`，实测 `tempfile.gettempdir()=D:\mm\tmp`。
- **依据**：Iron Constraint —— 仅写 D 盘。
- **2026-10-17 加固（真隐患）**：本 venv 的 CPython 3.12.7 在 **`import tempfile` 时就**把 `tempfile.tempdir` 解析成
  `C:\Users\Han\AppData\Local\Temp`（即便当时 `TMPDIR` 尚未注入）。因此「先 import 重型库、后注入环境变量」的写法会让
  torch/transformers/PIL 把临时文件写 C 盘。已在 `config.py` 顶层加：`import config` 即 `os.environ.update(ENV_VARS)`
  并把 `tempfile.tempdir = None` 强制重解析。实测裸进程 `import config; tempfile.gettempdir() -> D:\mm\tmp`；
  已 `source env/env.sh` 的入口结果不变（`D:\mm\tmp`），属**无行为变化**的加固。
- **覆盖行为（QA 指出，设计取舍，需知晓）**：`import config` 会**无条件覆盖**调用方已设的
  `TMPDIR`/`TMP`/`TEMP`/`HF_HOME`/`HF_HUB_CACHE`/`PIP_CACHE_DIR`/`TORCH_HOME` 等 **25 个**变量（即便调用方预设了
  别的值，例如 `TMPDIR=D:/some/other/tmp` → 被改写为 `D:\mm\tmp`）。**本项目有意为之**：C 盘锁定必须单一真源，
  不允许下游悄悄改回 C 盘。**如把 `config.py` 作为库嵌入其它项目，请勿在其后自行改这些变量**（改了也会在下次
  `import config` 时被覆盖；需要自定义请改 `ENV_VARS` 本身）。

### D-05 PyTorch 轮子源切换
- aliyun `pytorch-wheels` 实为 HTML 落地页（非 PEP503）且限速约 0.43MB/s。
- **决策**：改用 `download.pytorch.org/whl/cu128`（~7.8MB/s，支持 Range 续传），curl 落地到 `D:\mm\wheels` 再本地 pip 安装。
- **结果**：torch 2.7.1+cu128 / torchvision 0.22.1+cu128 安装成功，CUDA 12.8 可用。

### D-06 hf-mirror 经本地代理 502
- **现象**：沙箱代理 `HTTP(S)_PROXY=http://127.0.0.1:10131` 导致 hf-mirror 502。
- **决策**：`fetch_models.py` 内 **pop** `HTTP(S)_PROXY` 走直连；`config.py` 增加 `NO_PROXY`/`no_proxy` 白名单。

### D-07 HF 缓存 0 字节 snapshot 占位符
- **现象**：Windows 无符号链接权限时，huggingface_hub 生成 0 字节 snapshot 占位（blobs 内有真数据）。
- **决策**：`repair_repo_cache()` 依 HF API 的 LFS oid 映射把真实 blob 回填到 snapshot（本次修复 9 条），`verify_models.py` 复核 ALL OK。

### D-08 Plan A 缓存镜像 `refs/main` 换行问题
- **现象**：`refs/main` 以 CRLF 写入时，与 snapshots 目录名不匹配，离线解析失败。
- **决策**：以 `newline=""` 写入单行 sha（无换行），L5 离线解析通过。

### D-09 svdq-int4 体积偏差
- 设计预估 `svdq-int4-flux.1-dev` ≈ 4 GB；**实测 6.30 GB**。
- **决策**：以实测为准，回填 `state.json`；tier1 合计由预估 15.3 GB 修正为 **≈17.2 GB**。

### D-10 FLUX.1-dev 使用 ModelScope 分发（许可记录）
- 模型：`black-forest-labs/FLUX.1-dev` 组件，经 ModelScope 仓库 `AI-ModelScope/FLUX.1-dev` 获取（gated，HF 直连受限）。
- **许可**：FLUX.1-dev 为非商用许可（Non-Commercial License）。本次复现为**学生教育/个人非商用**用途，符合许可范围。
- **决策**：经团队负责人拍板（选项 A）授权下载；模型不用于任何商业用途。
- **备注**：仅下载推理所需组件（text_encoder / text_encoder_2 / tokenizer / tokenizer_2 / vae / scheduler / model_index），**排除** `transformer/**` 与原生 `flux1-dev.safetensors`（23.8GB，由 svdq-int4 4-bit 权重替代）。

### D-11 方案 A 优先，方案 B 未启用
- **决策**：先落地方案 A（HF 缓存镜像 + `HF_HUB_OFFLINE=1`）；L5 离线解析通过后，**未对 WorldGen 源码打补丁**。
- `patch_worldgen.py` 作为 standby 保留（将 4 处硬编码 FLUX repo id 改为读 `WORLDGEN_FLUX_DEV_DIR`），仅在方案 A 失效时启用。

### D-12 utils3d / matplotlib 补充依赖
- 设计称 DA-2 相关依赖未用；实测 `da2` 需 `utils3d`（git 3913c65，`--no-deps` 安装）与 `matplotlib`。
- **决策**：补充安装，避免 import 失败。

---

## T04

### D-13 viser fork 的运行时客户端自动构建在 Windows 上不可用（上游 bug）
- **现象**：`ViserServer()` 初始化时调用 `_client_autobuild.ensure_client_is_built()`；当 `client/build/index.html` 缺失时，它以
  `subprocess.run([<nodeenv>/Scripts/npx, ...])` 执行构建。Windows 下 `npx` 是 POSIX shell 脚本（真正的启动器是 `npx.cmd`），
  故 `CreateProcess` 抛 `OSError: [WinError 193] %1 不是有效的 Win32 应用程序`。
- **决策**：**不修改 viser 源码**，改为在运行前**手动构建一次**客户端静态资源：
  `npx.cmd --yes yarn install` + `npx.cmd vite build --base ./ --outDir <client>/build`（Node 20.4.0 由 nodeenv 提供）。
  构建后 `build/index.html` 存在且比 `src/` 新，运行时即跳过（损坏的）自动构建。
- **产物**：`scripts/build_viser_client.{sh,bat}`；`logs/viser_client_build.console.log`。
- **备注**：构建使用 **fork 自带**的 TS 源码（忠实原仓库），非上游 PyPI 预构建客户端。

### D-14 viser fork 从 submodule 安装（非 git URL）
- WorldGen `pyproject.toml` 声明 `viser @ git+https://github.com/ZiYang-xie/viser.git`；因 GitHub 直连受限且该仓库已作为
  submodule 初始化到 `submodules/viser`。
- **决策**：`pip install -e D:\mm\WorldGen\submodules\viser`（连同其跨平台依赖），版本 `0.2.23`。
- **产物**：`scripts/install_viser.bat`。

### D-15 全部模型离线解析（Plan A 在 T04 生效）
- `run_demo.{bat,sh}` 在运行 demo 前置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，使 4 个仓库
  （`haodongli/DA-2`、`mit-han-lab/svdq-int4-flux.1-dev`、`LeoXie/WorldGen`、`black-forest-labs/FLUX.1-dev`）
  全部从 D 盘本地缓存解析，运行期**不触网、不需 HF 鉴权**。
- 运行日志确认：`Couldn't connect to the Hub: ... offline mode is enabled. Will try to load from local cache.`

### D-16 xformers 未安装（实测非运行时必需）
- WorldGen 声明 `xformers>=0.0.30`，但源码中**无任何 import**（仅存在于 egg-info 元数据）。
- **决策**：不安装；diffusers 回退到 PyTorch SDPA，生成正常。

### D-17 可视化渲染未截图（受 C 盘约束）
- viser 唯一渲染路径是浏览器端 `client.get_render()`；headless 截图需 `agent-browser`，其会向 **C 盘**安装 Chromium(~500MB)。
- **决策**：为遵守"仅写 D 盘"，不安装浏览器截图；以 `splat.ply`（合法 1.28M 点 3DGS）+ viser `HTTP 200` 作为完成证据。

---

## T05

### D-18 run_bootstrap 幂等判定受 CRLF 影响（已修复）
- **现象**：`is_done()` 采用 `\n` 分隔的键集合做纯 bash 匹配；但 Python 在 Windows 上 `print()` 会把 `\n` 翻译为 `\r\n`，
  导致键集合实际为 `\nT01\r\nT02...\r\n`，`*$'\n'"T01"$'\n'*` 永不匹配 → 所有步骤被误判为"未完成"而重跑（幂等失效）。
- **决策**：`reload_done_keys()` 以 `| tr -d '\r'` 去除回车后再做匹配；实测连续 3 次运行 **16 SKIP / 0 RUN，约 5.5s**。
- **依据**：与 D-08（refs/main CRLF）同源——Windows 行尾问题是本项目的高频陷阱，统一显式处理。

### D-19 FLUX.1-dev 无副本冗余（方案 A 镜像已用硬链接，非复制）
- **澄清**：`D:\mm\models\FLUX.1-dev`（组装产物）与 `D:\mm\cache\hf\hub\models--black-forest-labs--FLUX.1-dev\snapshots\<sha>\`（方案 A 镜像）
  看似"两份 9.26GB"，实为**硬链接**共享同一物理 inode。
- **实测证据**：`stat` 显示同一文件在两路径下 **inode 相同 (562949954302593)、link count = 3**（`models/` 实体 + cache `snapshots/` + cache `blobs/`）；
  `du -sh /d/mm` = **30GB**（若为两份真实副本应为 ~39GB）。
- **决策**：`assemble_flux.py` 的 `build_hf_cache_mirror()` 本就用 **Windows 硬链接**（日志 `hardlinked=18, copied=0`），
  **无需优化、无 9.26GB 可回收**；两路径均可用且不额外占盘。保留双路径仅为语义清晰（staging 源 + HF 缓存布局）。

---

## T06 / T07（i2s 图生 3D 场景）

### D-20 上游 i2s 全景映射的稠密相似度矩阵必 OOM（运行时 shim 修复，未改源码）
- **现象（原始 demo.py）**：`torch.OutOfMemoryError: Tried to allocate 47.83 GiB`，栈在
  `general_utils.py:107 batch_nearest_dot → torch.mm(q, src_dirs.t())`，被 `map_image_to_pano` 调用。
- **根因**：i2s 的 `pred_depth` 用 `pano_unit_rays` 生成**整球等距圆柱** rays（把输入照片当作已覆盖整球），
  `resize_img_and_rays` 据此估到 FOV≈2π，源网格≈2048×1024=**2,091,012** 条；`batch_nearest_dot` 默认
  `nn_batch=8192` → 单块 `(8192 × 2,091,012)` fp32 ≈ **63.8 GiB**。**与 `-r`、`--low_vram`、输入图无关**，
  任何单卡/多数 CPU-RAM 都不可能。
- **决策**：`vendor/worldgen_i2s_shim`（类比既有 `vendor/pytorch3d_shim`）在**运行时**把 `batch_nearest_dot`
  重绑定为**分块流式实现（chunk=64）**——对同一完整相似度矩阵求 argmax，**数值等价**，瞬时 510.5 MiB。
  **未修改任何 WorldGen 源文件**。启动器 `scripts/run_i2s_patched.py` 先 apply patch 再 `runpy` 原 `demo.py`。
- **证据**：`logs/i2s_map_diag.console.log`（隔离诊断：0.37s 完成，peak 2607MiB）；`logs/demo_i2s.console.log`（原始 OOM）。

### D-21 FLUX.1-Fill-dev 组件经 ModelScope 分发（许可记录）
- i2s 基座 `black-forest-labs/FLUX.1-Fill-dev` 亦为 gated，hf-mirror 403。经 ModelScope `AI-ModelScope/FLUX.1-Fill-dev`
  组装推理所需组件（排除 `transformer/**`、`flux1-fill-dev.safetensors` 23.8GB、`ae.safetensors`）。
- **许可**：FLUX.1 同为**非商用许可**；本复现为**学生教育/个人非商用**，符合范围（同 D-10）。

### D-22 T5-XXL / CLIP-L 硬链接复用（省 9.10GB）
- **事实**：Fill 仓的 `text_encoder/model.safetensors`(246,144,352) 与 `text_encoder_2/model-0000{1,2}-of-00002.safetensors`
  (4,994,582,224 / 4,530,066,360) 与 dev **逐字节相同**。
- **决策**：`assemble_flux.py --variant fill` 先取 ModelScope listing 的每文件 `Sha256`，对本地 dev 副本做**全量 sha256 校验**，
  一致才 `os.link` 硬链接复用；不一致则回退下载。实测 3 文件均 `sha256 verified`，新增下载仅 **0.32GB**（Fill 独有：VAE 335MB 等）。
- **注意**：Fill 的 `vae/diffusion_pytorch_model.safetensors` 为 **335,306,212**（dev 为 167,666,902）**不同，必须单独下载**。
- **证据**：`logs/fetch_models_fill.console.log`；`stat` 显示 `text_encoder/model.safetensors` **6 links / inode 562949954302593**（与 dev 同 inode）。

### D-23 i2s 的 T5 编码显存争用（transformer/depth 让位，50 步 ~36s）
- **现象（仅修 batch 后）**：进程在 `_get_t5_prompt_embeds → _relative_position_bucket` 卡死 ~24 分钟（`py-spy` 定位）。
- **根因**：常驻 4-bit transformer(6.6GB)+DA-2 depth(1.3GB) 占满 12GB，`enable_model_cpu_offload` 下 9.5GB T5-XXL 无法同驻 GPU，
  被 `accelerate` 逐块换页至 CPU → 极慢。i2s 特有：t2s 在做深度前就已编码 prompt。
- **决策**：shim 追加——`pred_depth` 后把 depth 模型移 CPU（`pred_pano_depth` 前拉回）；`encode_prompt` 期间把 transformer 移 CPU。
  encode 前 **GPU free=9946 MiB ≥ 9.5GB**，T5 落 GPU，编码瞬间完成；去噪期 accelerate 钩子自动把 transformer 重新载回 GPU。
- **结果**：50 步 @ ~1.38 it/s（~36s），端到端 ~71.5s，显存峰值 11,975/12,288MiB（无 OOM）。产物 2,097,152 高斯、坐标全有限、viser HTTP 200。
- **证据**：`logs/demo_i2s_run3.log`、`logs/T07-report.md`。

### D-24 i2s 在 12GB 上的两条硬约束：`-r 1024` 与「T5 编码后必须主动撤离 GPU」

- **背景**：QA（严过关）独立复验文档入口 `run_demo_i2s.sh`，3 次尝试中 **2 次挂死**。挂死特征：
  `pre-encode` 之后日志**零进展**、无异常、无 OOM、进程存活、GPU util 100%、`memory.used` 11,957–11,962 MiB
  （剩余 ~125 MiB）。`py-spy dump` 两次快照栈**完全一致且冻结**，两处现场：
  `_encode_vae_image → vae.encode → conv_forward` 与 `prepare_mask_latents → randn_tensor → .to(device)`。
- **根因（QA 实验证实，非推测）**：`enable_model_cpu_offload()` 的
  `model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"` 采用**惰性驱逐**——
  只有**下一个**组件上台时，才会把上一个组件推下 GPU。D-23 的修法在 encode **之前**就把 transformer
  手动移出 GPU，于是 VAE 上台时它的「前任」早已不在，**9.5 GiB 的 T5-XXL 永远不会被驱逐**；
  VAE 的 conv / `randn` 分配只能挤进 ~125 MiB，Windows WDDM 以**分页**代替抛
  `torch.cuda.OutOfMemoryError` → **永久静默卡死**。一句话：D-23 只解决了「transformer vs T5」，
  漏了「T5 vs VAE」。
- **修复（仍为运行时 shim，未改源码）**：`vendor/worldgen_i2s_shim` 的 `_patch_encode_prompt` 在
  `out = orig(...)` **返回之后**遍历 `("text_encoder","text_encoder_2")` 逐个 `.to("cpu")`，再
  `torch.cuda.empty_cache()`，并打印 `post-encode: text encoders -> CPU [...], GPU free=... MiB`。
  实测 free 由 125 MiB 级 → **9,918 MiB**，VAE/latent 阶段再无挤压。
- **配套**：`scripts/run_i2s_watchdog.py`（`.sh`/`.bat` 共用）在无日志进展超过 `I2S_WATCHDOG_SECS`
  （默认 300s）时杀子进程并 **exit 124**，打印中文自诊断「疑似显存换页静默卡死…检查 `nvidia-smi` 余量
  是否 <200 MiB，并确认 post-encode 已将 T5 撤离」+ 末 20 行日志，**杜绝无限等待**。
- **`-r 1024` 约束**：无论 `-r` 取何值，i2s 的终态全景恒为 **2048×1024**（`splat.ply` 为 2,097,152 个
  高斯 = 全景像素数）。`-r 1600` 在 12GB 上直接 OOM；**12GB 卡一律 `-r 1024`**。
- **证据**：`logs/demo_i2s_stab{1,2,3}.log`（三次长 prompt 连续通过，含 `post-encode ... GPU free=`
  实测行）、`logs/T07-report.md` 的「缺陷修复记录」与「3× 稳定性矩阵」。

