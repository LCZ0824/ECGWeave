# 输入合同与复现范围

## 预训练

`ecgweave.pretraining.model.ValidECGMAE`保留最终F0的参数键，`core.ModelConfig`定义192维、12层、3头编码器与128维、2层逐导联解码器。有效支持以完整20点片段为单位；填充不参与可见输入、目标统计或重建损失。

```python
import torch
from ecgweave.pretraining.core import LossConfig, ModelConfig
from ecgweave.pretraining.model import ValidECGMAE, make_valid_mask

torch.set_num_threads(2)
model = ValidECGMAE(ModelConfig())
signals = torch.randn(1, 12, 5000)  # 合成mV输入，仅示范接口
valid = torch.ones(1, 250, dtype=torch.bool)
mask = make_valid_mask(valid, 0.75, "limbsync_mixed")
prediction = model(signals, mask, valid)
loss = model.loss(prediction, signals, mask, LossConfig("local_softnorm"), valid)
loss.backward()
```

原配置为九来源1,236,202条准入记录、batch32、100,000次更新。PTB-XL只准入其训练患者，其他来源的全局患者互斥性未建立。原患者身份与多来源波形组装缓存不公开，不能仅凭这段合成示例复现原F0。

## 少标注定界

`delineation.model.build_segmenter()`直接创建最终`gls_local`结构。无需原S*文件即可创建结构；复现训练则仍须按原配方先得到共同S*，固定教师后训练学生。编码器读取原始mV；局部分支使用逐导联标准化输入与两阶差分。

`delineation.pattern_mix.mix_batches`保留原双向供体选择、支持和拼接位置。`delineation.loss.pseudo_ce`按0.95阈值筛选伪标签，正式配置使用`denominator="valid"`；双视图损失取均值。模式混合的U→L窗口平均置信度阈值为严格大于0.8，与逐点伪标签阈值分别定义。

LUDB为120/40/40患者训练池、验证、内部测试；其中7人提供梯度标注，未标注流包括这7人在内的120人。另40人的验证标签用于选点。100轮、每轮60次更新，验证mIoU最高且并列最早；三个接续种子保持各自身份。

## 多标签识别

```python
import torch
from ecgweave.diagnosis.model import JointMultiresClassifier, ModelConfig

model = JointMultiresClassifier(ModelConfig(), arm="wide_joint")
batch = {
    "signals": torch.randn(1, 12, 5000),
    "reference": torch.full((1, 12, 250, 4), 0.25),
    "reference_support": torch.ones(1, 12, 250, dtype=torch.bool),
}
logits = model(batch)  # [1,71]；随机初始化，仅检查接口
logits.square().mean().backward()
```

最终分类器6,889,202参数，使用71类固定顺序。`objective.py`保留按类别条件采样、平方hinge排序、按TRAIN阳性率校正的完整71标签BCE及精确呈现预算。第一阶段为12轮BCE、6,540步；第二阶段为20遍记录呈现、5,444步。全部参数共同更新，固定定界器不接受梯度。学习率、分层衰减及其余设置见`configs/diagnosis.json`。

原PTB-XL为17,418/2,183/2,198条训练/验证/内部测试记录，对应15,023/1,942/1,904位患者。选点只使用完整Fold9宏AUROC，取最高且并列最早；最终单种子点为第9,262步。内部Fold10存在历史开发暴露，不是全新盲测或外部验证。这里不包含完整训练调度器、对照复现、临床统计分析脚本或患者缓存。

## 原始波形到参考

`pipeline.waveform_batch`由固定定界器输出逐点四态概率，完整20点片段取均值并归一；填充片段及整段常数导联无有效参考支持。原波形仍按原样进入分类器。可选`sample_support`为`bool[B,12,5000]`，仅表明真实采样支持。

该入口使用FP32概率；历史批量导出曾含半精度存储与不同批量的数值路径。本次验证包括同输入同参考的分类器数值一致性、定界器数值一致性及合成前端合同，不将新入口视为已重复验证全部临床预测或原缓存逐字节身份。要精确回放原分类器输出，需使用对应的固定参考缓存、检查点和原计算精度。

## 权重与验证

`docs/CHECKPOINTS.json`登记F0、S*、最终定界器及最终诊断器的SHA256。训练权重不随本次公开发布，访问安排联系通讯作者；没有公开下载的权重时，安装包只提供核心算法、结构及合成运行。

`pytest`测试只使用合成数据，核对有效支持、遮挡、双向混合、联合注意力、梯度及严格state_dict往返。`python -m ecgweave.smoke`检查正式尺寸下的结构、CPU/CUDA前反向与资源占用；结果仅是运行与接口验证，不是模型性能评估。
