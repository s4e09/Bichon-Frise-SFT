# 比熊
用 QQ 群聊记录构建「风格模仿」微调数据集：让模型学会用指定人物的一贯口吻回复群聊消息。

## 用法

```bash
node scripts/build_qq_style_dataset.js [源文件] [输出目录]
```

默认参数：

- 源文件：`E:/TencentFile/QQrecv/dialogs.jsonl`（QQ 群聊记录，每行一条 `{ context, reply, group_qq }`）
- 输出目录：`E:/TencentFile/QQrecv/qq_style_2024plus`

## 过滤规则

- 只保留 2024-01-01（Asia/Shanghai）之后的记录
- 目标回复须包含文本、不超过 200 字、不含链接和本地路径
- 上下文中的媒体消息折叠为 `[图片]`、`[语音]` 等标签

## 输出

| 文件 | 说明 |
| --- | --- |
| `style_train.jsonl` / `style_val.jsonl` | 训练/验证集（system/user/assistant 格式，按时间 8:1:1 切分） |
| `style_test.jsonl` | 测试集（不含答案） |
| `style_test_answers.jsonl` | 测试集参考答案 |
| `style_train_alpaca.jsonl` / `style_val_alpaca.jsonl` | Alpaca 格式（instruction/input/output） |
| `persona_profile.json` | 人物画像：回复长度、标点、emoji、n-gram、重复回复统计 |
| `stats.json` | 过滤与切分统计 |

## 配置

脚本内常量：`TARGET_QQ`（目标人物 QQ 号）、`CUTOFF`（时间截断）、`MAX_CHARS`（回复长度上限）、`SYSTEM_PROMPT`（训练用的 system prompt）。

## 训练

训练脚本使用 `ms-swift`，默认读取清洗后的 Alpaca 格式数据：

```bash
bash scripts/train_lora.sh configs/style_lora_2024plus.env
```

如果开发机沿用 `go.sh` 工作流，也可以直接运行：

```bash
bash go.sh
```

默认训练配置：

- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 数据目录：`$HOME/storage/SunJan10/qq_style_2024plus`
- 训练集：`style_train_alpaca.jsonl`
- 验证集：`style_val_alpaca.jsonl`
- LoRA：`rank=8`、`alpha=16`、目标模块 `q_proj,k_proj,v_proj,o_proj`
- 训练：`epoch=3`、`lr=2e-4`、`max_length=1024`

训练前需要先把 `E:/TencentFile/QQrecv/qq_style_2024plus` 中的数据搬到开发机的 `DATA_DIR`。如需改路径、epoch 或 batch size，编辑 `configs/style_lora_2024plus.env`。

### 本机 Windows 训练

本机训练使用 `transformers + peft`，默认读取本机清洗产物：

```powershell
.\scripts\train_local.ps1 -DryRun
.\scripts\train_local.ps1
```

默认配置在 `configs/local_peft_2024plus.json`：

- 数据目录：`E:/TencentFile/QQrecv/qq_style_2024plus`
- 输出目录：`D:/bixiong_lora/output/style_lora_2024plus`
- 缓存目录：`D:/bixiong_lora/cache`
- LoRA：`rank=8`、`alpha=16`
- 显存友好设置：`batch_size=1`、`gradient_accumulation_steps=16`、`max_length=768`、开启 gradient checkpointing

本机建议使用独立 Conda 环境，避免污染已有环境：

```powershell
.\scripts\setup_local_env.ps1
```

## 导出

训练完成后可合并 LoRA：

```bash
bash scripts/export_lora.sh configs/style_lora_2024plus.env
```

也可以显式指定 checkpoint：

```bash
bash scripts/export_lora.sh configs/style_lora_2024plus.env output/style_lora_2024plus/checkpoint-100
```
