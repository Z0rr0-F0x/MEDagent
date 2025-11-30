# MEDagent - 乳腺癌AI诊疗系统

一个基于人工智能的乳腺癌智能早筛、诊疗及预后评估系统，结合了EfficientNet影像诊断、CrewAI多智能体协作和RAG知识库检索技术。

## 功能特性

### 🤖 AI影像诊断
- 基于EfficientNet模型的乳腺癌影像智能诊断
- 99.35%高精度预测概率
- 自动化影像分析和风险评估

### 👨‍⚕️ 多智能体协作诊疗
- **问卷审核专家**: 审核患者基本信息的合理性和一致性
- **乳腺癌早筛专家**: 基于风险评估提供个性化筛查建议
- **乳腺癌诊断专家**: 融合AI影像结果与临床症状进行诊断
- **乳腺癌预后评估专家**: 制定个性化治疗方案和预后评估
- **流程审核专家**: 确保诊疗流程的逻辑一致性
- **首席乳腺癌专家**: 整合所有意见提供最终综合建议

### 📚 知识库与文献检索
- RAG(检索增强生成)本地知识库支持
- PubMed最新文献自动检索
- 支持自定义PDF医学文献上传
- 基于最新医学指南的智能推荐

### 🎯 核心功能
- 乳腺癌风险评估和早期筛查
- AI辅助影像诊断
- 个性化治疗方案制定
- 预后评估和康复建议
- 多维度医学知识整合

## 技术架构

### AI模型与框架
- **CrewAI**: 多智能体协作框架
- **LangChain**: LLM应用开发框架
- **Ollama**: 本地大语言模型部署
- **Chroma**: 向量数据库用于RAG检索
- **EfficientNet**: 医学影像诊断模型

### 数据处理
- PyPDF文档加载和处理
- 文本分割和向量化
- 生物医学文献检索(NCBI PubMed)

## 安装配置

### 环境要求
- Python 3.8+
- Ollama本地服务
- Conda环境管理

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd MEDagent
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置Ollama**
```bash
# 安装Ollama (参考官方文档)
# 拉取所需模型
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
```

4. **启动Ollama服务**
```bash
ollama serve
```

5. **配置EfficientNet环境**
```bash
# 创建conda环境
conda create -n efficientnet python=3.8
conda activate efficientnet
# 安装EfficientNet相关依赖 (具体参考EfficientNet目录说明)
```

## 使用说明

### 启动应用
```bash
streamlit run MEDagent_new.py
```

### 使用流程

1. **AI影像诊断**
   - 点击"开始AI诊断"按钮
   - 系统自动运行EfficientNet模型分析
   - 查看AI诊断结果和预测概率

2. **填写患者信息**
   - 基本 demographics (年龄、性别)
   - 家族病史和个人病史
   - 生育史和激素使用情况
   - 生活方式信息

3. **上传医学文献**(可选)
   - 支持PDF格式的医学指南
   - 用于增强RAG知识库

4. **开始AI会诊**
   - 点击"开始AI会诊"
   - 系统启动多智能体协作分析
   - 获取综合性的诊疗建议

### 输出结果
- **问卷审核**: 信息合理性验证
- **风险评估**: 个性化乳腺癌风险评估
- **诊断意见**: AI辅助诊断结果
- **治疗方案**: 个性化治疗建议
- **预后评估**: 康复预期和生活质量建议
- **专家审核**: 全流程质量控制

## 项目结构

```
MEDagent/
├── MEDagent_new.py          # 主应用文件
├── requirements.txt          # Python依赖
├── README.md                # 项目说明
├── RAGpdf/                  # 默认RAG知识库PDF目录
├── EfficientNet/            # EfficientNet模型目录
│   └── diagnostic_inference.py
└── chroma_db/              # 向量数据库(自动生成)
```

## 配置说明

### Ollama配置
- 默认地址: `http://localhost:11434`
- 使用的模型: `ollama/llama3.1:8b`
- 嵌入模型: `nomic-embed-text:latest`

### PubMed配置
- 需要配置有效的邮箱地址以符合NCBI使用政策
- 建议替换代码中的邮箱地址

### EfficientNet路径配置
- 默认脚本路径: `/commondocument/group6/MEDagent_demo/EfficientNet/diagnostic_inference.py`
- 可根据实际部署环境调整路径

## 注意事项

⚠️ **重要声明**
- 本系统仅供医学研究和教育使用
- 不能替代专业医疗诊断
- 所有诊断建议需经执业医师确认

🔧 **技术要求**
- 确保Ollama服务正常运行
- 需要足够的GPU资源运行EfficientNet模型
- 建议使用稳定的网络连接进行PubMed检索

📊 **数据隐私**
- 患者数据仅在本地处理
- 不会上传至外部服务器
- 建议在生产环境中添加数据加密

## 开发团队

本项目基于CrewAI多智能体框架开发，整合了最新的AI技术在乳腺癌诊疗领域的应用。

## 许可证

请查看项目根目录的LICENSE文件了解具体许可证信息。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**免责声明**: 本系统仅作为辅助工具，不能替代专业医疗诊断和治疗决策。