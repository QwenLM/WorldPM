# WorldPM 🌍

[![License](https://img.shields.io/badge/License-Qwen-green.svg)](https://huggingface.co/Qwen/WorldPM-72B/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-WorldPM-yellow)](https://huggingface.co/Qwen/WorldPM-72B)
[![arXiv](https://img.shields.io/badge/arXiv-2505.10527-b31b1b.svg)](https://arxiv.org/abs/2505.10527)
[![GitHub](https://img.shields.io/badge/GitHub-WorldPM-4b32c3?logo=github)](https://github.com/QwenLM/Qwen-World)


## 📚 Introduction
📄 [WorldPM](https://arxiv.org/abs/2505.10527) (World Preference Modeling) demonstrates that preference modeling follows similar **scaling laws** as language modeling. Through large-scale training on 15M preference data, we reveal that preference models can learn unified preference representations.

![main-loss](http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/WorldPM/main-loss.png)


<details>
<summary>🔍 Key Findings</summary>


* **In adversarial evaluation, test losses demonstrate a power law decrease**, indicating the model's enhanced ability to identify responses with intentional errors and those that are well-written but irrelevant or incomplete.
* **The objective metrics reveal an emergent phenomenon**, where larger models demonstrate a power law decrease in test losses across more benchmarks. WorldPM represents a challenging task that requires larger models to elicit preferences for objective knowledge, pointing to its substantial potential for further advancement.
* **Subjective evaluations show no apparent scaling trends.** We analyze potential reasons from the perspective of style preferences. While WorldPM becomes more style-neutral as it scales up, some subjective evaluations exhibit style preferences, resulting in lower evaluation performance.

</details>

<details>
<summary>🤔 Deep Dive: Understanding Scaling in Preference Modeling</summary>

## Why Subjective Domains Don't Scale

In our scaling experiments for preference modeling, we observed clear scaling trends in objective domains but not in subjective ones. We attribute this to the multi-dimensional nature of subjective evaluations - the assessment results are essentially averages across many dimensions. This leads to positive scaling in some dimensions and negative scaling in others, resulting in an apparent lack of overall scaling. Notably, as explained in our paper, for certain surface-level dimensions like style, WorldPM overcomes these biases, leading to significantly lower evaluation scores.

## Why Preference Modeling is Scalable

<details>
<summary>💡 Key Insights</summary>

The scalability of preference modeling might seem counterintuitive, with two main concerns:

1. **Task Perspective**: Preference modeling appears too simple with only binary signals (indicating which response is preferred), resulting in sparse supervision.

2. **Data Perspective**: Human forum data appears noisy and seemingly difficult to scale.

### Addressing the Concerns

**On Sparse Supervision:**
Consider why next token prediction successfully models language - to accurately predict the next word (e.g., with 90% probability), language models must understand comprehensive language rules. Similarly, to successfully predict 90% of preference dataset labels, models must learn sufficiently universal human preference representations.

**On Noisy Data:**
Noise refers to the apparent randomness in labels or supervision signals. However, since forum data represents genuine human annotations, it inherently contains its own rationality. Even if individual human intelligence cannot discern the patterns, powerful language models can discover underlying structures.

### Key Conclusion
Neural network scalability might depend neither on dense supervision signals nor on precise supervision signals. As long as the supervision signals are reasonable and challenging, scaling is possible - although dense and precise signals would accelerate convergence.

</details>
</details>

## 🎯 Model Usage

### Base Model and Fine-tuned Variants

WorldPM represents a breakthrough in unified preference representation learning through large-scale training. While our experiments demonstrate strong generalization capabilities across various preference scenarios, we recommend task-specific fine-tuning for optimal performance.

#### Base Model
- 🌟 [WorldPM-72B](https://huggingface.co/Qwen/WorldPM-72B)

#### Fine-tuned Versions
Each model is fine-tuned on human preference datasets of varying sizes:

| Model | Dataset | Training Scale |
|-------|---------|-------|
| [WorldPM-72B-HelpSteer2](https://huggingface.co/Qwen/WorldPM-72B-HelpSteer2) | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 7K |
| [WorldPM-72B-UltraFeedback](https://huggingface.co/Qwen/WorldPM-72B-UltraFeedback) | [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | 100K |
| [WorldPM-72B-RLHFLow](https://huggingface.co/Qwen/WorldPM-72B-RLHFLow) | [RLHFLow](https://huggingface.co/datasets/RLHFlow/pair_data_v2_80K_wsafety) | 800K |


### 💡 Recommendation
The base WorldPM-72B model serves as an excellent starting point for custom fine-tuning. Our experiments confirm that starting from WorldPM leads to better performance compared to training from scratch.



## 🛠️ Requirements

[![transformers](https://img.shields.io/badge/transformers-≥4.40.0-green.svg)](https://github.com/huggingface/transformers)

<details>
<summary>⚠️ Important Dependencies</summary>

- Required: `transformers>=4.40.0`
- Recommended: Latest version of `transformers`

> **Warning**: Version requirement is crucial as Qwen2.5 integration started from `transformers 4.37.0`

For GPU requirements and performance metrics, check the [Qwen2 benchmark results](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

</details>

## 🚀 Quick Start

### 📋 Key Notes
- System prompt remains empty during training (no preset preferences)
- System marker is preserved
- Reward computation uses the hidden state of `<|endoftext|>` token

### 💻 Usage Example with Hugging Face

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

</details>


## 📝 Citation

```bibtex
@article{WorldPM,
  title={WorldPM:Scaling Human Preference Modeling}, 
  author={Binghai Wang, Runji Lin, Keming Lu, Le Yu, Zhenru Zhang, Fei Huang, Chujie Zheng, Kai Dang, Yang Fan, Xingzhang Ren, An Yang, Dayiheng Liu, Tao Gui, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Bowen Yu, Jingren Zhou, and Junyang Lin},
  journal={arXiv preprint arXiv:2505.10527},
  year={2025}
}
```

## 🤝 Community & Support

We welcome discussions and feedback from the community! Here's how you can reach out:

- 📝 Open an issue on GitHub for bug reports or feature requests
- 💡 Share your ideas and questions in GitHub Discussions
- ✉️ Contact the authors directly at [here](mailto:refrain.wbh@gmail.com)

Feel free to engage with us through any of these channels. We value your input and look forward to hearing from you!
