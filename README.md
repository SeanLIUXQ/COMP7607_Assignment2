# COMP7607 Assignment 2: Analysis of Prompting Strategies for Code Generation

[English](https://github.com/SeanLIUXQ/COMP7607_Assignment2?tab=readme-ov-file#english) | [中文](https://github.com/SeanLIUXQ/COMP7607_Assignment2?tab=readme-ov-file#%E4%B8%AD%E6%96%87https://www.google.com/search?q=%23中文)

------

## 中文

### 📋 项目概述

本项目是 **COMP7607 Natural Language Processing (Fall 2025)** 课程 Assignment 2 的实现代码与分析报告 。

本项目旨在深入探究 **提示工程（Prompt Engineering）** 对大型语言模型（LLM）在代码生成任务上性能的影响。实验基于 **HumanEval** 基准测试集，使用了 **Qwen3-8B** 模型，分析了提示词质量、复杂度、示例数量（Few-shot）以及多样性对代码生成准确率（Pass@1）的影响。

此外，本项目还对比了两种推理策略：

1. **Baseline Method**：直接基于提示词生成代码。
2. **Combine Method**：引入了自我修正（Self-Refine）和基于单元测试反馈的修复循环（CodeT-style repair）。

### 🎯 实验维度 (Dimensions)

根据作业要求，本项目实现了针对以下四个维度的对比实验：

1. **Prompt Quality (提示词质量)**
   - `clean`: 标准、正确的描述与示例。
   - `wrong_demo`: 包含故意错误的示例代码。
   - `irrelevant_demo`: 包含正确但与当前任务无关的示例。
   - `bad_instruction`: 包含误导性的自然语言指令。
2. **Prompt Complexity (提示词复杂度)**
   - `simple`: 极度简化的任务描述。
   - `original`: 原始 HumanEval 描述。
   - `detailed`: 包含额外约束和边界条件的详细描述。
3. **Number of Demonstrations (示例数量)**
   - $k \in \{0, 1, 2, 4\}$：比较 Zero-shot 与 Few-shot 的效果。
4. **Prompt Diversity (提示词多样性)**
   - `low`: 使用固定模板。
   - `high`: 使用多种不同句式和结构的模板。

### 📁 项目结构

```
COMP7607_Assignment2/
├── baseline_eval_results/      # Baseline 方法的评测结果 (.jsonl)
├── combine_eval_results/       # Combine (Self-refine) 方法的评测结果
├── baseline_eval_summaries/    # 结果摘要统计
├── combine_eval_summaries/     # 结果摘要统计
├── main_generate_baseline.py   # Baseline 方法的主生成脚本
├── method_combine.py           # Combine 方法的主生成脚本 (Self-refine + Repair)
├── evaluate_functional_correctness.py # 功能正确性评估脚本
├── execution.py                # 代码执行沙箱/工具
├── HumanEval.jsonl             # 数据集
├── requirements.txt            # Python 依赖
└── README.md                   # 说明文档
```

### 🚀 快速开始

#### 1. 环境准备

确保您的 Python 版本为 3.8+，并安装依赖：

Bash

```
pip install -r requirements.txt
```

#### 2. API 配置

本项目支持 OpenAI 兼容格式的 API（如阿里云 Bailian/DashScope）。请在环境变量中设置您的 API Key：

Bash

```
# Linux / macOS
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export MODEL_NAME="qwen3-8b" # 或其他您使用的模型

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 💻 使用方法 (Usage)

脚本 `main_generate_baseline.py` 支持通过命令行参数控制实验变量。

#### 通用参数

- `--exp_family`: 实验维度 (`quality`, `complexity`, `num_demos`, `diversity`, `none`)。
- `--condition`: 具体条件 (如 `clean`, `simple` 等)。
- `--num_demos`: 示例数量 (仅在 `num_demos` 实验下生效)。
- `--diversity_mode`: 多样性模式 (`low`, `high`)。
- `--max_samples`: 测试样本数量 (默认为 80)。

#### 运行示例

**1. 运行 Baseline 默认设置 (Original, k=0)**

Bash

```
python main_generate_baseline.py --exp_family none
```

**2. 实验：提示词质量 (Prompt Quality)**

Bash

```
# 测试包含错误示例的情况
python main_generate_baseline.py --exp_family quality --condition wrong_demo
```

**3. 实验：提示词复杂度 (Prompt Complexity)**

Bash

```
# 测试简化描述的情况
python main_generate_baseline.py --exp_family complexity --condition simple
```

**4. 实验：示例数量 (Number of Demonstrations)**

Bash

```
# 4-shot learning
python main_generate_baseline.py --exp_family num_demos --num_demos 4
```

**5. 运行 Combine (Self-Refine) 方法**

*(假设 method_combine.py 接受类似的参数结构)*

Bash

```
python method_combine.py --exp_family quality --condition clean
```

#### 评估结果

生成完成后，使用评估脚本计算 Pass@1 准确率：

Bash

```
python evaluate_functional_correctness.py --sample_file baseline_eval_results/baseline_A2_default.jsonl
```

### 📊 关键结论 (Key Findings)

基于实验报告的分析，主要发现如下：

1. **复杂度至关重要**：提示词的详细程度对性能影响最大。过度简化的描述 (`simple`) 会导致准确率大幅下降（从 ~87% 降至 ~55%）。
2. **自我修正的有效性**：Combine 方法（Self-Correction）在大多数情况下都能提升 Baseline 的性能（平均提升约 5%），特别是在初始提示词存在噪音（如错误示例）时表现出更强的鲁棒性。
3. **示例数量的边际递减**：增加示例数量（Few-shot）在 $k=1$ 时达到峰值，继续增加示例 ($k=2, 4$) 并没有带来显著的线性提升，甚至可能引入干扰。
4. **多样性影响较小**：改变提示词的句式和结构（Diversity）对最终代码生成的准确率影响微乎其微。

### 👤 作者

- **Name**: Sean LIU
- **Course**: COMP7607 @ HKU
- **Report**: Analysis of Prompting Strategies for Coding.docx

------

## English

### 📋 Project Overview

This repository contains the implementation for **COMP7607 Assignment 2**, focusing on the **Analysis of Prompting Strategies for Coding**.



We explore how different prompt factors affect the reasoning and code generation capabilities of LLMs (specifically **Qwen3-8B**) using the **HumanEval** benchmark. Furthermore, we compare a standard **Baseline** method against a **Combine** method that utilizes self-refinement and test-based repair.

### 🎯 Experimental Dimensions

As per the assignment requirements, we analyze four key dimensions:

1. **Prompt Quality**: `clean`, `wrong_demo`, `irrelevant_demo`, `bad_instruction`.
2. **Prompt Complexity**: `simple`, `original`, `detailed`.
3. **Number of Demonstrations**: $k \in \{0, 1, 2, 4\}$.
4. **Prompt Diversity**: `low` vs. `high`.

### 🚀 Getting Started

1. **Install Dependencies**:

   Bash

   ```
   pip install -r requirements.txt
   ```

2. **Set API Key**:

   Bash

   ```
   export OPENAI_API_KEY="your-api-key"
   export MODEL_NAME="qwen3-8b"
   ```

### 💻 Usage

Run the baseline generation script with specific experiment parameters:

Bash

```
# 1. Baseline (Default)
python main_generate_baseline.py --exp_family none

# 2. Experiment: Quality (e.g., Wrong Demo)
python main_generate_baseline.py --exp_family quality --condition wrong_demo

# 3. Experiment: Complexity (e.g., Simple)
python main_generate_baseline.py --exp_family complexity --condition simple

# 4. Experiment: Num Demos (e.g., k=2)
python main_generate_baseline.py --exp_family num_demos --num_demos 2
```

### 📊 Results Summary

- **Specification Quality dominates**: Simplistic prompts drastically reduce performance.
- **Self-Correction works**: The Combine method consistently improves over the baseline, especially recovering from noisy prompts.
- **Few-shot saturation**: Performance peaks around $k=1$; adding more shots provides diminishing returns.
- **Diversity is secondary**: Paraphrasing prompts has minimal impact compared to content quality.

For full details, please refer to the report: `Analysis of Prompting Strategies for Coding.docx`.
