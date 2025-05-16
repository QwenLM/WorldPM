# WorldPM 🌍
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2505.10527-b31b1b.svg)](https://arxiv.org/abs/2505.10527)
[![GitHub](https://img.shields.io/badge/GitHub-WorldPM-4b32c3?logo=github)](https://github.com/QwenLM/WorldPM)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-yellow)](https://huggingface.co/Qwen/WorldPM-72B)
[![ModelScope](https://img.shields.io/badge/🤖%20ModelScope-purple)](https://modelscope.cn/models/Qwen/WorldPM-72B)

[English](./README.md) | [中文](./README_CN.md)
## 📚 简介
📄 [WorldPM](https://arxiv.org/abs/2505.10527)（世界偏好建模）证明了偏好建模遵循与语言建模类似的**扩展规律**。通过对1500万条偏好数据进行大规模训练，我们发现偏好模型能够学习统一的偏好表示。

![main-loss](http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/WorldPM/main-loss.png)

<details>
<summary>🔍 主要发现</summary>

* **在对抗性评估中，测试损失呈现幂律下降**，表明模型在识别有意图错误和表面完善但不相关或不完整的回复方面的能力得到提升。
* **客观指标显示出涌现现象**，更大的模型在更多基准测试中表现出测试损失的幂律下降。WorldPM代表了一个具有挑战性的任务，需要更大的模型来获取客观知识的偏好，这表明它具有巨大的进步潜力。
* **主观评估没有显示明显的扩展趋势。** 我们从风格偏好的角度分析了潜在原因。虽然WorldPM在扩展过程中变得更加风格中性，但一些主观评估表现出风格偏好，导致评估性能降低。

</details>


<details>
<summary>🤔 深入理解：偏好建模中的扩展性</summary>

## 为什么主观领域不具有扩展性

在我们的偏好建模扩展实验中，我们观察到客观领域有明显的扩展趋势，但主观领域没有。我们将其归因于主观评估的多维特性——评估结果本质上是多个维度的平均值。这导致某些维度呈现正向扩展，而其他维度呈现负向扩展，最终表现为整体缺乏扩展性。值得注意的是，正如论文所述，对于某些表面层面的维度（如风格），WorldPM克服了这些偏见，导致评估分数显著降低。

## 为什么偏好建模是可扩展的

<details>
<summary>💡 关键见解</summary>

偏好建模的可扩展性可能看起来违反直觉，主要有两个顾虑：

1. **任务视角**：偏好建模似乎过于简单，只有二元信号（表示哪个回应更受偏好），导致监督信号稀疏。

2. **数据视角**：人类论坛数据看起来嘈杂且似乎难以扩展。

### 应对这些顾虑

**关于稀疏监督：**
考虑为什么下一个词预测能成功建模语言——为了准确预测下一个词（例如，90%的概率），语言模型必须理解全面的语言规则。同样，为了成功预测90%的偏好数据集标签，模型必须学习足够通用的人类偏好表示。

**关于嘈杂数据：**
噪声指的是标签或监督信号中的表面随机性。然而，由于论坛数据代表真实的人类标注，它本质上包含自己的合理性。即使个体人类智能无法辨别其中的模式，强大的语言模型也能发现潜在结构。

### 主要结论
神经网络的可扩展性既不依赖于密集的监督信号，也不依赖于精确的监督信号。只要监督信号合理且具有挑战性，扩展就是可能的——尽管密集和精确的信号会加速收敛过程。

</details>
</details>

## 🎯 模型使用

### 基础模型和微调变体

WorldPM在通过大规模训练进行统一偏好表示学习方面取得了突破。虽然我们的实验表明其在各种偏好场景中都具有强大的泛化能力，但我们建议针对具体任务进行微调以获得最佳性能。

#### 基础模型
- 🌟 [WorldPM-72B](https://modelscope.cn/models/Qwen/WorldPM-72B)

#### 微调版本
每个模型都在不同规模的人类偏好数据集上进行微调：

| 模型 | 数据集 | 训练规模 |
|-------|---------|-------|
| [WorldPM-72B-HelpSteer2](https://modelscope.cn/models/Qwen/WorldPM-72B-HelpSteer2) | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 7K |
| [WorldPM-72B-UltraFeedback](https://modelscope.cn/models/Qwen/WorldPM-72B-UltraFeedback) | [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | 100K |
| [WorldPM-72B-RLHFLow](https://modelscope.cn/models/Qwen/WorldPM-72B-RLHFLow) | [RLHFLow](https://huggingface.co/datasets/RLHFlow/pair_data_v2_80K_wsafety) | 800K |

### 💡 建议
基础WorldPM-72B模型是进行自定义微调的理想起点。我们的实验证实，从WorldPM开始比从头训练能获得更好的性能。

## 🛠️ 环境要求

[![transformers](https://img.shields.io/badge/transformers-≥4.40.0-green.svg)](https://github.com/huggingface/transformers)

<details>
<summary>⚠️ 重要依赖</summary>

- 必需：`transformers>=4.40.0`
- 推荐：最新版本的`transformers`

> **警告**：版本要求至关重要，因为Qwen2.5集成从`transformers 4.37.0`开始

有关GPU要求和性能指标，请查看[Qwen2基准测试结果](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)。

</details>

## 🚀 快速开始

### 📋 关键说明
- 训练期间system prompt保持为空（无预设偏好）
- 保留`system`标记
- 奖励计算使用`<|endoftext|>`标记的隐藏状态

### 💻 使用示例

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

model_name = 'Qwen/WorldPM-72B'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name,trust_remote_code=True )
model = AutoModel.from_pretrained(
    model_name, 
    config = config, 
    device_map = "auto", 
    trust_remote_code=True,
).eval()


def get_score(model, tokenizer, conversation):
    con_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(con_str, return_tensors="pt", add_special_tokens=False)
    outputs = model(input_ids=input_ids.to(model.device))
    score = outputs[0].cpu().item()
    return score


messages = [
    [
        {'role': 'system', 'content': ""},
        {'role': 'user', 'content': "Tips for better sleep?"},
        {'role': 'assistant', 'content': "Three key tips for better sleep:\n1. Set a consistent bedtime and wake-up schedule\n2. Avoid screens 1 hour before bed\n3. Keep your bedroom cool and dark\n\nStart with these habits and you should notice improvement within a week."}
    ],
    [
        {'role': 'system', 'content': ""},
        {'role': 'user', 'content': "Tips for better sleep?"},
        {'role': 'assistant', 'content': "just try sleeping earlier or take some pills if u cant sleep lol. maybe watch tv until u get tired idk"}
    ]
]

score = [get_score(model, tokenizer=tokenizer, conversation=message) for message in messages]

print(score) #[0.6298332214355469, -1.6425328254699707] for WorldPM-72B
```

## 📝 引用

```bibtex
@article{WorldPM,
  title={WorldPM:Scaling Human Preference Modeling}, 
  author={Binghai Wang, Runji Lin, Keming Lu, Le Yu, Zhenru Zhang, Fei Huang, Chujie Zheng, Kai Dang, Yang Fan, Xingzhang Ren, An Yang, Dayiheng Liu, Tao Gui, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Bowen Yu, Jingren Zhou, and Junyang Lin},
  journal={arXiv preprint arXiv:2505.10527},
  year={2025}
}
```

## 🤝 社区与支持

我们欢迎来自社区的讨论和反馈！以下是联系方式：

- 📝 在GitHub上提出问题报告或功能请求
- 💡 在GitHub讨论中分享想法和问题
- ✉️ 直接联系作者：[邮箱](mailto:refrain.wbh@gmail.com)

欢迎通过以上任何渠道与我们交流。我们重视您的意见，期待听到您的反馈！
