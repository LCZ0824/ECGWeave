# ECGWeave

十二导联心电的预训练、少标注波段定界与多标签诊断代码。当前版本对应论文的 `stats_prior20` 模型：71类输出，9088维表征。

## 安装

Python ≥ 3.10，先安装适合本机的 PyTorch：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ecgweave.smoke --device cpu
```

## 推理

需要自行提供训练权重。输入 NPZ 中的 `signals` 为 `float32[B,12,5000]`：500 Hz、10秒、原始 mV；导联顺序为 I、II、III、aVR、aVL、aVF、V1–V6。

```bash
ecgweave-predict --input input.npz --checkpoint stats_prior20.pt \
  --delineator delineator.pt --output prediction.npz --device cpu
```

输出包含71类 logits、sigmoid 分数及类别名称。也可输入已对齐的软参考，见[使用说明](docs/REPRODUCTION.md)。模型、损失和配置分别位于 `src/ecgweave/`、`configs/`。

## 数据与权重

仓库提供核心实现和合成测试，不含完整训练调度、患者数据、划分文件或训练权重；目前不能仅凭本仓库复现论文的全部结果。[检查点信息](docs/CHECKPOINTS.json)用于核对已有权重。数据库需按原提供方的条件获取；医院数据访问请联系通讯作者 Kang Chen（ck11208@rjh.com.cn），由医院审批。

代码采用 [Apache-2.0](LICENSE)。第三方来源见 [NOTICE](NOTICE) 和[许可说明](THIRD_PARTY_LICENSES.md)。
