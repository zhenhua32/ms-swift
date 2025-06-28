# MS-Swift 框架思维导图

## 🍲 MS-Swift (Scalable lightWeight Infrastructure for Fine-Tuning)

### 🎯 核心定位
- **全栈式大模型工具箱**
  - 从训练到部署的一站式解决方案
  - 支持500+大模型 + 200+多模态模型
  - 轻量化高效微调技术集成

### 🏗️ 架构层次

#### 1️⃣ 用户接口层
- **命令行工具 (CLI)**
  - `swift sft` - 监督微调
  - `swift infer` - 推理
  - `swift deploy` - 部署
  - `swift eval` - 评测
- **Web界面 (UI)**
  - 基于Gradio的零门槛界面
  - 可视化训练配置
  - 实时监控面板
- **Python API**
  - 编程式调用接口
  - Jupyter Notebook支持

#### 2️⃣ 核心训练层
- **传统训练 (Trainers)**
  - 预训练 (Pre-training)
  - 监督微调 (SFT)
  - 序列分类训练
- **人类对齐训练 (RLHF)**
  - **GRPO** - Group Relative Policy Optimization
  - DPO - Direct Preference Optimization
  - PPO - Proximal Policy Optimization
  - RM - Reward Model Training
  - KTO, CPO, SimPO, ORPO
- **轻量化技术 (Tuners)**
  - LoRA/QLoRA - 低秩适应
  - DoRA - 权重分解的LoRA
  - AdaLoRA - 自适应LoRA
  - LongLoRA - 长序列LoRA
  - GaLore, Q-GaLore - 梯度低秩投影
  - LISA, UnSloth, Liger

#### 3️⃣ 模型支持层
- **大模型 (LLM)**
  - Qwen系列 (Qwen3, Qwen2.5等)
  - InternLM, GLM4, Mistral
  - DeepSeek, Yi, Baichuan, Gemma
- **多模态模型 (MLLM)**
  - Qwen2.5-VL, Qwen2-Audio
  - Llava, InternVL, MiniCPM-V
  - GLM4v, Xcomposer, Yi-VL
- **特殊模型**
  - Embedding模型
  - Reranker模型
  - 序列分类模型

#### 4️⃣ 推理加速层
- **推理引擎**
  - PyTorch (原生)
  - **vLLM** - 高效推理引擎
  - SGLang - 结构化生成语言
  - LMDeploy - 轻量部署
- **量化技术**
  - AWQ - 权重量化
  - GPTQ - 后训练量化
  - BNB - 8bit/4bit量化
- **并行技术**
  - 数据并行 (DDP)
  - 模型并行 (Tensor Parallel)
  - 流水线并行 (Pipeline Parallel)
  - DeepSpeed ZeRO

#### 5️⃣ 数据与评测层
- **数据集**
  - 150+内置数据集
  - 自定义数据集支持
  - 多模态数据处理
- **评测系统**
  - EvalScope后端
  - 100+评测数据集
  - 自动化评测流程
- **插件系统**
  - 奖励模型插件
  - 多轮调度器
  - 自定义损失函数

### 🔥 GRPO训练器深度解析

#### 核心特性
- **组相对策略优化**
  - 基于组的优势计算
  - 相对奖励比较
  - 动态裁剪机制
- **异步推理生成**
  - 并行生成与训练
  - 队列缓存机制
  - 内存优化策略
- **多奖励函数支持**
  - 奖励模型集成
  - 自定义奖励函数
  - 权重化组合

#### 技术创新
- **vLLM集成**
  - 服务器模式 (Server Mode)
  - 共址模式 (Colocate Mode)
  - 张量并行支持
- **动态采样 (DAPO)**
  - 零方差组重采样
  - 提高训练效率
  - 避免退化样本
- **多轮对话支持**
  - Agent tool calling
  - 对话历史管理
  - 自定义调度策略

### 🌟 生态集成

#### 硬件支持
- **GPU**: A10/A100/H100, RTX系列, T4/V100
- **国产芯片**: Ascend NPU
- **其他**: CPU, MPS (Apple Silicon)

#### 平台集成
- **ModelScope**: 魔搭社区模型中心
- **HuggingFace**: 兼容HF生态
- **Docker**: 官方镜像支持
- **Notebook**: 免费GPU资源

#### 开发者工具
- **文档完整**: 中英双语文档
- **示例丰富**: examples目录
- **社区活跃**: Discord + 微信群
- **持续更新**: 快速迭代发布

### 🚀 使用流程

#### 快速开始
```bash
# 1. 安装
pip install ms-swift -U

# 2. 训练
swift sft --model Qwen2.5-7B-Instruct --dataset ms-bench

# 3. 推理
swift infer --model_id_or_path output/checkpoint-xxx

# 4. 部署
swift deploy --model_id_or_path output/checkpoint-xxx
```

#### Web界面
```bash
swift web-ui  # 启动Web界面
```

#### Python编程
```python
# 模型准备
model, tokenizer = get_model_tokenizer(model_path)
model = Swift.prepare_model(model, lora_config)

# 训练
trainer = Seq2SeqTrainer(model=model, args=args, ...)
trainer.train()
```

### 📊 技术优势

#### 性能优化
- **内存高效**: LoRA等轻量化技术
- **计算加速**: vLLM/SGLang引擎
- **并行训练**: 多种并行策略
- **量化部署**: 模型压缩技术

#### 易用性
- **零门槛**: Web界面操作
- **一键部署**: 简化部署流程
- **丰富示例**: 快速上手
- **完整文档**: 详细指导

#### 扩展性
- **插件系统**: 自定义扩展
- **多后端**: 推理引擎选择
- **云原生**: 容器化部署
- **社区驱动**: 开源生态

### 🎯 应用场景

#### 学术研究
- 大模型微调实验
- RLHF算法研究
- 多模态模型开发

#### 工业应用
- 企业级模型定制
- 智能客服系统
- 内容生成平台

#### 教育培训
- 大模型教学
- 实践项目
- 技能培训

---

## 总结

MS-Swift是一个功能完整、技术先进的大模型训练部署框架，特别在RLHF训练方面有独特优势。其GRPO训练器代表了当前RLHF训练的前沿技术，结合了异步推理、多奖励函数、动态采样等创新特性，为大模型的人类对齐训练提供了高效解决方案。
