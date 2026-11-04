# 使用说明

## 当前模型

```python
import torch
from ecgweave.diagnosis.stats_model import build_classifier

model = build_classifier()  # 随机初始化
batch = {
    "signals": torch.randn(1, 12, 5000),
    "reference": torch.full((1, 12, 250, 4), 0.25),
    "reference_support": torch.ones(1, 12, 250, dtype=torch.bool),
}
logits = model(batch)  # [1, 71]
```

`stats_prior20` 使用统计查询读取器和方差残差，不含额外导联注意力。分类器共6,639,486参数；每类128维，展平表征为9088维。`diagnosis/model.py`保留基础结构，当前模型请从 `stats_model.build_classifier` 创建。

诊断训练为单阶段20轮记录呈现、10,900次更新，目标为完整71标签BCE与固定训练先验辅助损失；实现见 `diagnosis/prior_objective.py`。最终权重按验证集宏AUROC选于第9轮、第4,905步。配置见 `configs/diagnosis.json`。

## 波形与软参考

`pipeline.waveform_batch` 先将500Hz波形依次作五阶0.67Hz高通和40Hz低通零相位滤波，再输入冻结定界器。分类器始终接收原始mV。四态顺序为BG、P、QRS、T；逐点softmax后对连续20点平均，以float16量化，再恢复float32并归一化。无效片段和常数导联的支持为False。

已有参考时，输入NPZ同时包含 `reference: float32[B,12,250,4]` 与 `reference_support: bool[B,12,250]`，可省略 `--delineator`。原波形可附 `sample_support: bool[B,12,5000]` 标记真实采样点。

CLI默认FP32；CUDA支持 `--precision bf16`，对应论文的BF16前向、FP32统计读取及logits。不同精度、批量和后端可能产生数值差异。`task_names71.json`固定类别顺序，其中包含诊断、形态及节律陈述，分数不直接作为临床结论。

## 其他阶段

- 预训练：`pretraining.model.ValidECGMAE`，同步肢体导联遮蔽和有效片段重建。
- 定界：`delineation.model.build_segmenter`，`gls_local`结构；模式混合与伪标签损失在同目录。
- 配置：`configs/`；运行检查：`python -m ecgweave.smoke --device cpu`，GPU改为 `cuda:0`。

## 复现范围

测试使用合成数据检查计算、梯度与接口，不代表性能复现。当前发布不含完整数据处理、患者划分、训练调度或临床分析流程。PTB-XL Fold10有历史开发暴露，不能作为全新盲测。权重身份见 [CHECKPOINTS.json](CHECKPOINTS.json)，权重文件尚未公开。
