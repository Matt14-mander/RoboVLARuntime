# FluxVLA：SmolVLA 在 LIBERO 中的评测调用链

分析日期：2026-09-05

上游仓库：`FluxVLA/FluxVLA`

分析基准：commit `8e22b69b2ff8c8c333d4095596cde8e1e3b57ade`（2026-09-04，`[Fix] Align PI0 runtime horizons and attention shapes`）

代表配置：`configs/smolvla/smolvla_libero_spatial_finetune.py`

本笔记通过静态源码分析得到。尚未安装 FluxVLA 依赖、下载 checkpoint 或运行 LIBERO，因此张量的实际值、运行时延、动作物理语义和任务成功率仍需动态验证。

## 1. 结论先行

FluxVLA 的 `scripts/eval.py` 是评测编排入口，不执行模型计算。SmolVLA 的 LIBERO 调用链为：

```text
scripts/eval.py
  Config.fromfile(config)
  cfg.merge_from_dict(--cfg-options)
  _run_eval()
    build_runner_from_cfg(eval_cfg)
      RUNNERS registry
      LiberoEvalRunner.__init__()
        build SmolVLAFlowMatching
        load checkpoint
        load dataset_statistics.json
        build LiberoParquetEvalDataset
        build DenormalizeLiberoAction
    LiberoEvalRunner.run_setup()
      move model to CUDA / chosen dtype
    LiberoEvalRunner.run()
      reset LIBERO environment
      execute dummy actions for num_steps_wait
      dataset(obs) -> model batch
      SmolVLAFlowMatching.predict_action(**batch)
      take first eval_chunk_size actions
      for each action:
        DenormalizeLiberoAction
        env.step(action)
      write summary.json
    LiberoEvalRunner.cleanup()
```

代表配置中，模型一次生成 `chunk_size=50` 个动作，runner 只执行前 `eval_chunk_size=10` 个，然后使用新观测重新推理。当前 LIBERO runner 是同步执行：推理时仿真环境不推进；chunk 执行时模型不并行生成下一段动作。

这条评测路径适合验证模型成功率，但不能直接代表真实机器人在推理延迟期间的闭环行为。RoboVLA-Runtime 的第一阶段需要补上独立的时间、缓冲、延迟和 fallback 语义。

## 2. `eval.py` 负责什么

入口：`scripts/eval.py`。

### 2.1 命令行和配置

`parse_args()` 读取：

- `--config`：Python/MMEngine 配置文件；
- `--ckpt-path`：checkpoint；
- `--cfg-options`：运行时配置覆盖。

它使用 `parse_known_args()`，未知参数会被返回但主函数忽略。主函数先调用 `configure_inference_attention_defaults()`，然后加载配置并合并覆盖项。

`_get_eval_runner_cfg()` 同时兼容：

```python
eval = dict(type='LiberoEvalRunner', ...)
```

和：

```python
eval = dict(runner=dict(type='LiberoEvalRunner', ...), ...)
```

函数对 runner 配置执行 `deepcopy`，所以每个 task suite 都有独立的配置副本。`task_suite_name` 可以是字符串，也可以是列表；主函数为每个 suite 单独构造和销毁 runner。

### 2.2 runner 的构建与生命周期

`_run_eval()` 执行：

```text
复制 eval runner 配置
移除 Feishu report-only 字段
写入 suite-specific max_steps
附加完整 cfg 与 ckpt_path
build_runner_from_cfg
run_setup
run
查找 run_dir/summary.json
finally: cleanup
```

`build_runner_from_cfg()` 使用 `RUNNERS` registry。配置里的：

```python
type='LiberoEvalRunner'
```

会匹配带有 `@RUNNERS.register_module()` 的 `LiberoEvalRunner` 类。除 `type` 外的配置字段会作为构造参数传入。

一个容易忽略的事实是：模型构建、checkpoint 加载、dataset 和反归一化 transform 的创建发生在 `LiberoEvalRunner.__init__()`，早于 `run_setup()`。`run_setup()` 主要设置 seed、CUDA device、eval 状态和 dtype。

### 2.3 结果汇总

runner 在 `run_dir` 写出 `summary.json`。`eval.py` 收集不同 suite 的 summary，合并：

- `suite_stats`；
- `task_results`；
- `total_time`；
- `total_tasks`；
- `total_successes`；
- `total_trials`。

整体成功率为 `total_successes / total_trials`。只有 rank 0 且没有通过 `task_ids` 运行子集时，才会尝试调用 Feishu reporter。

## 3. 代表性 SmolVLA 配置

文件：`configs/smolvla/smolvla_libero_spatial_finetune.py`。

### 3.1 模型配置

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `model.type` | `SmolVLAFlowMatching` | registry 中的 VLA 类 |
| `vlm_backbone.type` | `SmolVLMBackbone` | 视觉语言 backbone |
| `llm_expert.type` | `SmolVLMExpert` | 动作 expert |
| `max_action_dim` | 32 | 模型内部 padded action 维度 |
| `ori_action_dim` | 7 | 原始任务动作维度；训练损失会裁剪到此维度 |
| `chunk_size` | 50 | 一次 flow-matching 生成的动作步数 |
| `num_steps` | 10 | Euler ODE 去噪步数 |
| `pretrained_name_or_path` | `./checkpoints/smolvla_base/model.safetensors` | base 权重位置 |

模型使用两个 512×512 视觉输入。VLM prefix 包含图像、语言和 state；action expert suffix 包含 noisy action 与时间嵌入。

### 3.2 eval 配置

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `eval.type` | `LiberoEvalRunner` | 评测 runner |
| `task_suite_name` | `libero_spatial` | LIBERO suite |
| `model_family` | `smolvla` | 日志/分支标识 |
| `eval_chunk_size` | 10 | 每次预测后最多执行的动作数 |
| `num_trials_per_task` | 50 | 每个任务的 episode 数 |
| `num_steps_wait` | 10 | episode 开始时执行 dummy action 的步数 |
| `seed` | 7 | 评测随机种子 |

`model.chunk_size=50` 与 `eval.eval_chunk_size=10` 含义不同：前者决定模型生成长度和计算，后者只决定 runner 截取并执行多少步。将 `eval_chunk_size` 从 10 改小，不会让本次模型生成更快。

## 4. runner 初始化过程

文件：`fluxvla/engines/runners/libero_eval_runner.py`。

`LiberoEvalRunner.__init__()` 的关键过程：

1. 从完整配置复制 `cfg.model`，如果存在 `cfg.inference_model` 则优先使用它。
2. `build_vla_from_cfg(model_cfg).eval()` 构建模型。
3. 加载 `safetensors` 或 PyTorch checkpoint；对于 `.pt`，存在同名 `.safetensors` 时优先使用后者。
4. 默认从 checkpoint 上两级目录读取 `dataset_statistics.json`。
5. 把统计文件注入 dataset 和 `DenormalizeLiberoAction`。
6. 默认统计 key 为 `<task_suite_name>_no_noops`。
7. 构建 `LiberoParquetEvalDataset` 和 action denormalizer。

`run_setup()`：

- 设置随机种子；
- 设置当前 CUDA device；
- 将模型置于 eval；
- 设置各 backbone freeze 标志；
- 按配置将模型移动到 device 和 mixed precision dtype。

`cleanup()` 清空模型、dataset 和 transform 引用，运行 GC，并尝试清理 CUDA cache/IPC。

## 5. observation 到模型 batch

runner 将以下字段加入环境 observation：

```python
obs['task_description'] = task_description
obs['is_new_episode'] = is_new_episode
```

然后调用：

```python
batch, replay_img = self.dataset(obs)
```

### 5.1 图像

配置读取：

- `agentview_image`；
- `robot0_eye_in_hand_image`。

`ProcessLiberoEvalInputs` 对图像做两个空间维度的翻转，转为 RGB PIL，并产生 `img_masks`。`TransformImage` 使用 top-left letterbox，输出 512×512 图像并按配置归一化。

dataset 最终将图像移动到 CUDA 并加 batch 维度。静态代码表明 `images` 为合并后的图像 tensor；其精确布局需在首次运行时打印确认。

### 5.2 语言

代表配置设置 `use_conversation=False` 和 `add_new_line=True`，因此 prompt 是原始 `task_description` 加换行，不使用类 OpenVLA 的问答模板。默认最大 token 长度为 180，短序列右侧 padding。

### 5.3 proprio state

`LiberoProprioFromInputs` 拼接：

```text
robot0_eef_pos
quaternion -> axis-angle
robot0_gripper_qpos
```

然后用 `<suite>_no_noops/proprio` 的 mean/std 归一化，并在代表配置中补零到 32 维。dataset 转为 BF16 CUDA tensor，并加 batch 维度，所以预期 `states` 形状为 `[1, 32]`。

### 5.4 episode history

dataset 接收 `is_new_episode`，在新 episode 时重置 image buffer，输出 `reset_history`。代表配置的 `img_buffer_len` 使用默认值 1，因此没有多帧历史堆叠，但 reset 语义仍存在。

## 6. SmolVLA 如何生成 action chunk

文件：`fluxvla/models/vlas/smolvla_flowmatching.py`。

`predict_action()` 接收：

```text
images
lang_tokens
states
img_masks
lang_masks
```

调用过程：

1. 根据 `states.shape[0]` 获取 batch size。
2. 若没有外部 noise，采样形状为 `[B, chunk_size, max_action_dim]` 的高斯噪声。
3. 编码 image、language、state prefix，并计算一次 prefix KV cache。
4. 执行 `num_steps=10` 次 Euler ODE 去噪。
5. 返回 `x_t`。

代表配置下，静态预期返回形状为：

```text
[B, 50, 32]
```

推理路径没有在模型末端根据 `ori_action_dim=7` 裁剪结果。runner 后续依靠 action transform 裁剪到 7 维。这看起来是 padded action 表示的既有设计，不能在没有跨配置回归证据时当作 bug。

## 7. action chunk 如何被环境消费

runner 在 `torch.autocast` 和 `torch.no_grad` 中调用：

```python
actions = self.vla.predict_action(**predict_kwargs)
```

如果结果为三维：

```python
actions = actions[0, :eval_chunk_size, :]
```

代表配置把 `[1, 50, 32]` 截为 `[10, 32]`。随后对每一个 action：

1. 使用 `<suite>_no_noops/action` 的 mean/std 反归一化；
2. 二值化 gripper action；
3. 反转 gripper action；
4. 截取前 `action_dim=7` 维；
5. 调用 `env.step(action_denormed.tolist())`。

因此 LIBERO 实际收到 7 维动作。7 个数的具体坐标约定和 delta/absolute 语义需要结合 FluxVLA 数据转换、LIBERO controller 和一次动态记录确认；仅从此调用链不下结论。

模型生成一次后，runner 顺序执行最多 10 个环境 step。期间不会重新调用模型。`preprocess_every_step` 默认是 `True`：每个环境 step 后都会计算一次下一 observation 的 dataset batch，但变量会被后续 step 覆盖，最终下一轮推理使用 chunk 末尾 observation 对应的 batch。这也会维护图像历史和 replay 数据。设为 `False` 时，只在需要新 chunk 时预处理。

当前时序可画成：

```text
obs_0
  └─ preprocess
      └─ synchronous predict: A_0[0:50]
          ├─ execute A_0[0]
          ├─ execute A_0[1]
          ├─ ...
          └─ execute A_0[9]
              └─ obs_10 -> next prediction
```

这里不存在显式 action buffer、deadline、observation timestamp、推理与执行并行或 buffer underrun fallback。

## 8. profiler 应插在哪里

至少需要四层计时：

| 边界 | 建议位置 | 测得内容 |
| --- | --- | --- |
| observation preprocessing | `self.dataset(obs)` 前后 | 图像、prompt、state 处理和 H2D |
| model generation | `self.vla.predict_action()` 前后 | 完整 50-step chunk 生成 |
| action postprocess | `self.denormalize_action(inputs)` 前后 | 单动作反归一化与 gripper 处理 |
| execution/step | `env.step()` 前后 | 仿真 step 墙钟耗时 |

CUDA 调用异步，直接用 CPU `time` 包围 `predict_action()` 可能低估 GPU 完成时间。细粒度诊断应使用 CUDA events 或明确同步；正常吞吐测试不能在每个小阶段强制同步。两种模式需要分开报告。

更关键的是记录关联：每个 action 必须能追溯到生成它的 observation、request 和 chunk。FluxVLA 当前 LIBERO loop 没有这些显式 ID，因此第一版 profiler 可以先在 RoboVLA-Runtime 建立事件 schema，再评估如何低侵入地上游化。

## 9. 第一批动态验证问题

- [ ] 实际 `images`、`lang_tokens`、`states`、model output 的 shape/dtype/device。
- [ ] `dataset_statistics.json` 的 key 与数组维度。
- [ ] LIBERO 7D action 的坐标、单位和 delta/absolute 语义。
- [ ] `preprocess_every_step=True/False` 是否影响 SmolVLA 结果或仅影响性能。
- [ ] model generation 的 warm-up、p50、p95 和 CUDA 显存。
- [ ] `eval_chunk_size` 对任务表现、调用次数和 episode 墙钟时间的影响。
- [ ] 推理期间 LIBERO 逻辑时间是否完全冻结；静态 loop 表明是，但需动态 trace 确认。

## 10. 与 RoboVLA-Runtime 的接口边界

从这条链路可以提炼三个适配点：

```text
FluxVLA dataset(obs)             -> Observation preprocessing adapter
vla.predict_action(**batch)      -> PolicyBackend.predict_chunk
denormalize + env.step(action)   -> Action postprocessor + Environment adapter
```

第一阶段不复制 SmolVLA 或 LIBERO runner。先用 mock backend 和 toy plant 实现时间正确的调度内核；第二阶段再写薄适配器，把 FluxVLA 的 chunk 生成调用接进相同事件记录协议。

