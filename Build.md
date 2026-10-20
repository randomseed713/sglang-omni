# glibc 2.32 / CUDA 12.9

基于 v0.1.8（`870d77446`），使用 Python 3.10、PyTorch 2.13.0+cu129，以及公开上游 SGLang v0.5.21（本地构建版本 0.5.21+cu129）。
跳过 Rust 扩展，启用包括 L20（SM89）在内的 SM8x/SM90 内核。
以下命令在本仓库根目录执行；已有环境和源码目录时跳过创建、克隆。

## 1. 创建环境

```bash
unset PYTHONPATH PYTHONHOME
uv venv .venv-py310-cu129 --python 3.10 --managed-python
source .venv-py310-cu129/bin/activate
```

## 2. 获取源码并编译

```bash
mkdir -p third_party
git clone --depth 1 --branch v0.5.21 https://github.com/sgl-project/sglang.git third_party/sglang

uv run --python .venv-py310-cu129/bin/python --script scripts/build_sglang.py \
  --cuda-home /usr/local/cuda --jobs 16 \
  --index-url https://artifacts.antgroup-inc.cn/simple/
```

构建配置为 `packaging/profiles/cu129.toml`，产物位于 `dist/cu129/`。
本机已有依赖源码缓存，可追加 `--dependency-cache .build/cu129/dependencies`。
此配置面向 MiniCPM-o / L20：不编译或安装 DeepGEMM、FlashRec。
本机使用 GCC 10、CUDA Toolkit 12.8，PyTorch 及运行时使用 cu129。
构建配置包含 GCC 10 的 JIT 语法适配，以及 SM8x 的 FA3 调度元数据适配，保留 CUDA Graph。

## 3. 安装并验证

```bash
uv pip install --python .venv-py310-cu129/bin/python \
  --prerelease=allow \
  --default-index https://artifacts.antgroup-inc.cn/simple/ \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  --find-links dist/cu129 --constraint packaging/cu129-constraints.txt \
  --overrides packaging/cu129-overrides.txt \
  --reinstall-package sglang \
  -e '.[minicpm-o]'

uv pip check
python -c 'import torch; print(torch.__version__); print(torch.ones(1, device="cuda") + 1)'
```

当前机器通过 `/usr/local/cuda/compat` 加载兼容驱动库。
CUDA 12.9 依赖通过 overrides 显式选择，不修改项目默认的 CUDA 13 依赖。
该环境跳过 CUDA 13 的 NIXL、Mooncake、NVSHMEM 包及 DeepGEMM；Dots-TTS 的 NumPy 2.x 依赖也不安装，相关代码仍保留。
当前驱动未生成 MPS PID 文件时，使用控制 socket 核验 daemon 身份。

## 4. 下载并启动 MiniCPM-o

```bash
export HF_HOME=/home/chenzhengda.zd/workspace/models/llm/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
hf download openbmb/MiniCPM-o-4_5

CUDA_VISIBLE_DEVICES=4 sgl-omni serve \
  --model-path openbmb/MiniCPM-o-4_5 --model-name MiniCPM-o-4_5 \
  --variant text --thinker.engine.attention_backend fa3 \
  --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5 sgl-omni serve \
  --model-path openbmb/MiniCPM-o-4_5 \
  --model-name MiniCPM-o-4_5 \
  --variant text \
  --thinker.tp_size 2 \
  --thinker.gpu '[0,1]' \
  --thinker.process thinker \
  --thinker.engine.attention_backend fa3 \
  --thinker.engine.mem_fraction_static 0.75 \
  --placement.require_memory_fraction_for_colocation false \
  --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=6 sgl-omni serve \
  --model-path openbmb/MiniCPM-o-4_5 \
  --model-name MiniCPM-o-4_5 \
  --variant text --mps on \
  --thinker.process thinker \
  --thinker.engine.attention_backend fa3 \
  --thinker.engine.mem_fraction_static 0.75 \
  --placement.require_memory_fraction_for_colocation false \
  --host 0.0.0.0 --port 8001
```

GPU 编号按实际空闲情况修改。服务就绪后，在另一个终端执行：

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniCPM-o-4_5","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```
