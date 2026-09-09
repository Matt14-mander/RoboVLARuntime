# FluxVLA Profiler

## 目标

该工具使用一个真实 LIBERO observation 重复调用 FluxVLA 的 chunk 生成入口，
测量模型推理成本，不执行完整 rollout，也不报告任务成功率。

它复用 FluxVLA `LiberoEvalRunner` 的模型加载、dataset、autocast、推理参数和动作
反归一化逻辑。启动时创建指定 task/trial 的 LIBERO 环境，执行 runner 配置中的
等待步数，然后把得到的 observation 用作固定 profiling 输入。

## 运行方式

需要先激活已经安装 FluxVLA、LIBERO、PyTorch 和对应 CUDA 版本的 Python 环境。

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli profile-fluxvla `
  --fluxvla-root .upstream/FluxVLA `
  --config .upstream/FluxVLA/configs/smolvla/smolvla_libero_spatial_finetune.py `
  --checkpoint D:/checkpoints/smolvla/checkpoint.pt `
  --task-suite libero_spatial `
  --task-id 0 `
  --trial-id 0 `
  --action-period 0.1 `
  --action-dimension 7 `
  --num-inference-steps 10 `
  --inference-seed 7 `
  --warmup 10 `
  --samples 100 `
  --mode diagnostic `
  --output runs/profiles/smolvla-spatial-diagnostic
```

`--action-period` 必须填写该 checkpoint 和环境对应的动作周期，不能根据控制循环
频率猜测。输出目录必须不存在，避免覆盖已有实验。

## 两种计时模式

`diagnostic` 在预处理后和模型调用后同步 CUDA，输出：

- `preprocess`：包括 dataset transforms 和将输入搬到 GPU；
- `inference`：同步后的模型 wall time；
- `postprocess`：转 CPU、裁剪 chunk 和逐动作反归一化；
- `inference_gpu`：CUDA Event 测得的模型区间；
- `total`：一次完整后端请求。

阶段同步会改变正常流水行为，因此该模式用于定位耗时来源。

`throughput` 不在 preprocess/inference 边界强制同步，只在完整请求完成前保证 CUDA
工作结束。它只报告 `total` 和 `inference_gpu`，用于测量正常顺序请求吞吐。两个模式
应分别运行，不能把 diagnostic 的吞吐数字当成部署性能。

## 输出文件

```text
profile-output/
  config.json
  requests.jsonl
  summary.json
  environment.json
  metadata.json
```

`requests.jsonl` 同时保留 warm-up 和正式样本，每条记录包含 observation/request ID、
wall time、后端阶段时间、chunk horizon、动作维度、当前显存和单次请求峰值显存。
warm-up 记录带有 `warmup: true`，不会进入 `summary.json` 的分位数。

`summary.json` 对正式样本输出 mean、p50、p95、p99、max、顺序请求吞吐率和
生成动作步吞吐率。`environment.json`
记录 Python、PyTorch、CUDA 和 GPU 信息。`config.json` 保存 profiler 参数以及 FluxVLA
配置源码、checkpoint 路径、task/trial、推理步数和统计 key。

## 当前边界

Profiler 固定同一个有效 observation，用于隔离输入变化并测量稳态成本。它不反映不同
场景图像复杂度，也不测任务成功率。完成第一批测量后，应增加 recorded-observation
replay，使输入覆盖一个真实 episode，同时保持不同 scheduler 使用同一组输入。

当前实现使用 PyTorch 当前 CUDA stream 的 Event。若模型内部使用额外 stream，必须
先核对同步关系再解释 `inference_gpu`。多 GPU 或分布式 runner 也需要在目标环境中
单独验证。
