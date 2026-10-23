# ECGWeave

**Physiological references guide multiscale learning for electrocardiogram interpretation**

ECGWeave连接十二导联心电波形预训练、少标注波段定界与记录级多标签识别。定界器提供固定的BG/P/QRS/T软参考；诊断模型通过粗细通路交换和类别特异查询，联合读取波形特征与原始电压幅值。最终诊断结构的工程标识为`wide_joint`。

本仓库发布三阶段核心模型、遮蔽与损失、模式引导混合、固定配置、71类顺序、推理接口和合成测试。多来源数据获取、患者划分文件与完整训练主管依赖原受控研究环境，未随本次核心代码发布。训练权重尚未公开；仅有源码时可以检查结构和梯度，不能获得论文中的训练后预测。

## 安装与检查

Python 3.10及以上，先安装适合本机的PyTorch，再执行：

```bash
git clone https://github.com/LCZ0824/ECGWeave.git
cd ECGWeave
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ecgweave.smoke --device cpu
```

核验环境为Python 3.10、PyTorch 2.8.0+cu128。GPU合成检查可执行`python -m ecgweave.smoke --device cuda:0`。合成检查不下载数据或权重，也不启动模型训练。

## 代码与配置

|入口|内容|
|---|---|
|[预训练](src/ecgweave/pretraining/)|同步肢体导联遮蔽、共享编码器、逐导联解码、有效支持与local_softnorm重建|
|[定界](src/ecgweave/delineation/)|原始网格定界头、差分局部分支、双向模式混合与伪标签损失|
|[诊断](src/ecgweave/diagnosis/)|粗细波形编码、有界参考关系、双向交换、联合查询读出与条件排序目标|
|[三阶段配置](configs/)|原研究超参数；诊断配置已移除本机GPU身份|
|[输入与复现说明](docs/REPRODUCTION.md)|数据合同、分阶段调用、选点及当前发布范围|
|[检查点身份](docs/CHECKPOINTS.json)|本地研究权重的SHA256；文件未随仓库发布|
|[源码来源](docs/SOURCE_PROVENANCE.json)|冻结实现来源、原SHA256、公开适配与发布文件身份|

## 使用可信本地权重推理

波形必须为`float32[B,12,5000]`，500 Hz、10秒、原始mV，导联依次为I、II、III、aVR、aVL、aVF、V1–V6。不在入口隐式重采样、换算单位或逐记录归一化。

输入NPZ可以只包含`signals`，并由固定定界器生成软参考：

```bash
ecgweave-predict --input input.npz --checkpoint wide_joint.pt \
  --delineator delineator.pt --output prediction.npz --device cpu
```

也可提供对齐的`reference: float32[B,12,250,4]`及`reference_support: bool[B,12,250]`：

```bash
ecgweave-predict --input aligned_input.npz --checkpoint wide_joint.pt \
  --output prediction.npz --device cpu
```

输出为71类`logits`、sigmoid研究分数及固定`task_names`，不设置临床判定阈值。71项包括诊断、形态与节律陈述，并非71种独立疾病。检查点只应来自可信来源。

## 数据、权重与许可

本仓库不包含患者记录、波形、逐人标签、划分身份、患者预测或临床缓存。公开与受限数据库须向原提供方按各自条件获取。医院数据未获公开共享授权，不予公开；访问申请请联系通讯作者Kang Chen（ck11208@rjh.com.cn），具体可访问范围依医院审批及适用的数据使用条件确定。

本次发布代码和配置，训练权重未公开。SHA登记用于核对已有可信权重，不构成下载地址或发布承诺。

代码使用[Apache-2.0](LICENSE)，来源与归属见[NOTICE](NOTICE)及[第三方说明](THIRD_PARTY_LICENSES.md)。代码许可证不授予任何数据集或未发布权重的访问权限。
