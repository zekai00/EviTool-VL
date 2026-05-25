## 默认工作偏好
- 用中文回答。
- 修改代码前先说明计划。
- 写代码时请给出明确详细的注释。
- 修改代码后尽量运行相关测试。
- 不要擅自引入新依赖。
- 对不确定的地方要明确说明。


# Repository Guidelines

## 项目结构与模块组织

EviTool-VL 是一个用于证据闭合视觉工具推理的 Python 研究工作区。核心模块位于 `tools/`，提供 JSON action/observation 工具，例如 `ocr`、`detect`、`crop`、`measure`、`trace` 和 `click`。评测入口位于 `eval/`，数据集和 SFT 转换脚本位于 `datasets/`，实验辅助脚本位于 `scripts/`，训练配置位于 `configs/`。结果摘要应放在 `reports/`；大型生成产物应放在 `outputs/`、`checkpoints/` 或 `/root/models`，不要提交到 Git。

## 构建、测试与开发命令

安装依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-ocr.txt
```

检查环境：

```bash
python3 scripts/check_env.py
```

运行工具冒烟测试：

```bash
python3 scripts/smoke_tools.py --image data/eval_mini/images/chartqa/0000.png --out-dir outputs/tools_smoke
```

运行直接回答评测或工具调用评测：

```bash
python3 eval/eval_baseline.py --model /root/models/Qwen2.5-VL-3B-Instruct --output outputs/direct.jsonl
python3 eval/eval_tool_baseline.py --model /root/models/Qwen2.5-VL-3B-Instruct --output outputs/tool.jsonl
```

提交前对改动过的 Python 文件做语法检查：

```bash
python3 -m py_compile tools/*.py eval/*.py datasets/*.py scripts/*.py
```

## 编码风格与命名约定

使用 Python 3 和 4 空格缩进；在合适的位置添加类型标注；函数应保持短小，并使用清晰的 JSON schema。工具 action 使用 snake_case 字段名，像素框统一采用 `[x1, y1, x2, y2]` 格式。注释应简短，只解释不明显的逻辑。数据集构建、评测和工具执行应优先保持确定性。

## 测试指南

当前还没有正式的 pytest 测试套件。请使用冒烟测试、`py_compile` 和小规模评测子集作为回归检查。修改工具时，需要同时验证模块 runner 和下游 evaluator。新增脚本应按用途命名，例如 `check_train_eval_split.py` 或 `prepare_sft_v2_data.py`。

## 提交与 Pull Request 指南

近期提交使用简短的祈使句摘要，例如 `Add SFT pipeline and split eval reporting` 和 `Improve UI detection and evidence labeling`。每个提交应保持范围清晰，避免把生成产物和代码修改混在一起。PR 应描述实验或工具变更，列出已运行的命令，链接报告或 summary JSON，并说明所需的本地资源或模型路径。

## 安全与配置提示

不要提交 API key、模型权重、数据集缓存或私有输出。供应商访问凭据应使用 `DASHSCOPE_API_KEY` 等环境变量配置；报告时只说明凭据是否已配置，绝不要输出具体值。
