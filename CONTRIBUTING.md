# 贡献指南

感谢你对 LLM4VLM 项目的关注！本文档提供贡献指南。

## 如何贡献

### 报告问题

发现 Bug 或有功能建议？请创建 Issue：
- 描述问题详情
- 提供复现步骤
- 附加错误日志（如有）

### 提交代码

1. **Fork 项目**

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **修改代码**
   - 遵循现有代码风格
   - 添加必要的注释
   - 确保代码可运行

4. **测试**
   ```bash
   # 运行基本测试
   python code/prepare_data.py
   python code/train_vln_r2r_enhanced.py --epochs 1
   ```

5. **提交**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature
   ```

6. **创建 Pull Request**

## 代码风格

- 遵循 PEP 8 规范
- 使用 4 空格缩进
- 函数添加文档字符串
- 变量和函数使用小写

### 示例

```python
def process_data(data_path: str, output_path: str) -> dict:
    """
    处理输入数据并生成输出

    Args:
        data_path: 输入数据路径
        output_path: 输出数据路径

    Returns:
        处理结果字典
    """
    # 实现代码
    pass
```

## 开发环境设置

```bash
# 创建虚拟环境
python -m venv vln-env
source vln-env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发工具（可选）
pip install black flake8 pytest
```

## 代码检查

```bash
# 代码格式化
black code/

# 代码检查
flake8 code/
```

## 文档

- README.md: 项目主文档
- DATASET.md: 数据集说明
- MODELCARD.md: 模型说明
- CONTRIBUTING.md: 贡献指南

## 社区准则

- 尊重他人，友善交流
- 用英语或中文沟通
- 对事不对人

## 许可证

贡献代码即同意采用 MIT 许可证。
