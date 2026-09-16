# Bichon-Frise-SFT 本机训练总结

时间：2026-09-17 01:50 左右完成

## 训练环境

- 机器：本机 Windows
- GPU：NVIDIA GeForce RTX 4070 12GB
- Conda 环境：`bixiong-lora`
- 基座模型：`Qwen/Qwen3-4B-Instruct-2507`
- 训练方式：PEFT LoRA
- 数据集：`E:/TencentFile/QQrecv/qq_style_2024plus`
- 输出目录：`D:/bixiong_lora/output/style_lora_2024plus`
- 缓存目录：`D:/bixiong_lora/cache`
- 未使用 `G:` 移动硬盘

## 数据与参数

- 训练集：2077 条
- 验证集：260 条
- 测试集：260 条
- epoch：3
- max_length：768
- batch size：1
- gradient accumulation：16
- LoRA rank：8
- LoRA alpha：16
- 可训练参数：5,898,240
- 总参数：4,028,366,336
- 可训练比例：0.1464%

## 训练结果

- 总步数：390
- 总训练耗时：约 3760 秒
- train_loss：3.436
- eval_loss：
  - step 100：3.8725
  - step 200：3.8085
  - step 300：3.8260
  - step 390：3.8160

## 产物

- 最终 adapter：`D:/bixiong_lora/output/style_lora_2024plus/final_adapter`
- 最终 checkpoint：`D:/bixiong_lora/output/style_lora_2024plus/checkpoint-390`
- 采样输出：`D:/bixiong_lora/output/style_lora_2024plus/sample_outputs.json`

## 实际输出样例

采样参数：`temperature=0.9`，`top_p=0.9`，`max_new_tokens=48`，`repetition_penalty=1.03`。

### 样例 1

上下文：

```text
特务: 什么部门
特务: 给你联系联系人脉 [QQ表情]
他: ictbg数据通信
他: 强大的js
他: 怎么华子还要填写实习申请表
```

参考答案：`爽的呀`

模型输出：`是不是要签三方了`

### 样例 2

上下文：

```text
他: 送人了
```

参考答案：`那我只能相信老黄了`

模型输出：

```text
颜牛
🐮
呢
```

### 样例 3

上下文：

```text
叛徒: [图片]
叛徒: 排了很长的队
叛徒: [图片]
叛徒: 20
他: [图片]
```

参考答案：`nv养老嘛，颜牛别卷同事了`

模型输出：

```text
🥛
呢
```

### 样例 4

上下文：

```text
叛徒: 草，感觉午饭时间尊重一下华子已经是我们组的日常了（）
```

参考答案：`😲`

模型输出：`😧`

### 样例 5

上下文：

```text
特务: @叛徒
特务: 我把你要离职的事情跟王一桐说了
叛徒: @特务
 她早就知道了 [引用]
特务: @老好人
 那下周来杭州玩吧
```

参考答案：`其他邮回去`

模型输出：

```text
@老好人
 来的时候再告诉他们
```

## 观察

- 模型已经学到短回复、emoji、换行和“呢”等高频口癖。
- 输出内容不一定命中真实答案，但风格有明显贴近迹象。
- eval_loss 在 step 200 最低，最终 step 390 仅略高；后续可评估 `checkpoint-200`、`checkpoint-390` 和 `final_adapter` 的主观盲测效果。
- 第一轮没有继续迭代，按要求训练完成后停止。
