# bixiong

用 QQ 群聊记录构建「风格模仿」微调数据集：让模型学会用指定人物的一贯口吻回复群聊消息。

## 用法

```bash
node scripts/build_qq_style_dataset.js [源文件] [输出目录]
```

默认参数：

- 源文件：`E:/TencentFile/QQrecv/dialogs.jsonl`（QQ 群聊记录，每行一条 `{ context, reply, group_qq }`）
- 输出目录：`E:/TencentFile/QQrecv/qq_style_2025plus`

## 过滤规则

- 只保留 2025-01-01（Asia/Shanghai）之后的记录
- 目标回复须为纯文本、不超过 200 字、不含链接和本地路径
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
