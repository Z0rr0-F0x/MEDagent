import os
import json
import shutil
import streamlit as st
import re
import subprocess
import time
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from Bio import Entrez
import xml.etree.ElementTree as ET

# ==================== 初始化 ====================
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(model="ollama/llama3.1:8b")

# ==================== Session State 初始化 ====================
if 'efficientnet_result' not in st.session_state:
    st.session_state.efficientnet_result = None
if 'ai_done' not in st.session_state:
    st.session_state.ai_done = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# ==================== 安全解析 EfficientNet 输出 ====================
def safe_parse_probabilities(text):
    """安全解析 [[0.00649487 0.9935051]] 格式的概率数组"""
    try:
        # 提取 [[...]] 部分
        match = re.search(r'\[\s*\[\s*([\d.eE+-]+\s+[\d.eE+-]+)\s*\]\s*\]', text)
        if not match:
            return []
        pair_str = match.group(1)
        # 按任意空白分割两个数字
        parts = re.findall(r'[\d.eE+-]+', pair_str)
        if len(parts) != 2:
            return []
        return [[float(parts[0]), float(parts[1])]]
    except Exception as e:
        print(f"解析概率失败: {e}")
        return []

# ==================== 按钮启动 EfficientNet 诊断 ====================
def run_efficientnet_diagnosis():
    """用户点击按钮后运行 diagnostic_inference.py"""
    
    script_path = "/commondocument/group6/MEDagent_demo/EfficientNet/diagnostic_inference.py"
    if not os.path.exists(script_path):
        st.session_state.efficientnet_result = {
            "status": "error",
            "message": f"诊断脚本不存在: {script_path}"
        }
        st.error(f"脚本不存在: {script_path}")
        return

    cmd = [
        "conda", "run", "-n", "efficientnet",
        "python", script_path
    ]

    with st.spinner("AI影像诊断进行中...（EfficientNet 推理）"):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False
            )

            if result.returncode == 0:
                output = result.stdout
                # 安全解析
                pred_class_match = re.search(r"Predicted Class: ([\w]+)", output)
                prob_match = re.search(r"Probability: ([\d.]+)", output)

                pred_class = pred_class_match.group(1) if pred_class_match else "Unknown"
                prob = float(prob_match.group(1)) if prob_match else 0.0
                all_probs = safe_parse_probabilities(output)

                st.session_state.efficientnet_result = {
                    "status": "success",
                    "predicted_class": pred_class,
                    "probability": prob,
                    "all_probabilities": all_probs,
                    "raw_output": output.strip()
                }
                st.session_state.ai_done = True
                st.success(f"AI诊断成功：{pred_class} (概率: {prob:.2%})")
            else:
                st.session_state.efficientnet_result = {
                    "status": "error",
                    "message": result.stderr.strip() or "推理失败",
                    "stdout": result.stdout.strip()
                }
                st.error("AI诊断失败")
        except subprocess.TimeoutExpired:
            st.session_state.efficientnet_result = {"status": "timeout", "message": "推理超时"}
            st.error("AI诊断超时")
        except Exception as e:
            st.session_state.efficientnet_result = {"status": "exception", "message": str(e)}
            st.error(f"AI诊断异常: {e}")

# ==================== PubMed 检索功能 ====================
def retrieve_pubmed_from_med_rag(query, count=3):
    """从med_rag_app中提取PubMed文献内容"""
    try:
        Entrez.email = "your_email@example.com"  # 请替换为真实邮箱以符合NCBI政策
        handle = Entrez.esearch(db="pubmed", term=query, sort="relevance", retmax=count)
        record = Entrez.read(handle)
        handle.close()

        ids = record["IdList"]
        pubmed_context = []

        for pid in ids:
            handle = Entrez.efetch(db="pubmed", id=pid, retmode="xml")
            xml_content = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_content)
            title_elem = root.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "无标题"
            
            abstract_elem = root.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else "无摘要可用"
            
            pubmed_context.append({
                "id": pid,
                "title": title,
                "abstract": abstract[:500] + "..."
            })
        
        return pubmed_context
    except Exception as e:
        return [{"error": f"PubMed检索失败: {str(e)}"}]

def get_med_rag_pubmed_supplement(user_info_json, pubmed_count=3):
    """获取med_rag_app的PubMed补充信息"""
    
    gender = user_info_json.get("gender", "")
    age = user_info_json.get("age", "")
    family_history = user_info_json.get("family_history", "")
    
    pubmed_queries = [
        f"乳腺癌筛查 {age}岁 {gender}",
        f"乳腺癌风险因素 {family_history}",
        "乳腺癌早期检测指南",
        f"乳腺X光检查推荐 {age}岁 女性",
        "乳腺癌预后治疗方案"
    ]
    
    all_pubmed_content = []
    
    for query in pubmed_queries[:2]:
        try:
            pubmed_content = retrieve_pubmed_from_med_rag(query, pubmed_count)
            all_pubmed_content.extend([{"query": query, **item} for item in pubmed_content])
        except Exception as e:
            all_pubmed_content.append({"query": query, "error": str(e)})
    
    return all_pubmed_content

# ==================== MEDagent 功能（100% 保留原逻辑） ====================
def setup_rag_knowledge_base(uploaded_files=None, pdf_directory="/commondocument/group6/MEDagent_demo/RAGpdf"):
    """设置RAG知识库从PDF文件"""
    
    # 清理旧的向量数据库
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    pdf_files = []
    
    # 处理上传的文件列表
    if uploaded_files and isinstance(uploaded_files, list):
        print("使用上传的文件列表")
        for file_obj in uploaded_files:
            if hasattr(file_obj, 'name') and (file_obj.name.lower().endswith('.pdf') or 
                                            file_obj.name.lower().endswith('.pdf')):
                # 如果是文件对象，保存到临时位置
                temp_path = f"./temp_{os.path.basename(file_obj.name)}"
                with open(temp_path, "wb") as f:
                    f.write(file_obj.getvalue())
                pdf_files.append(temp_path)
                print(f"添加上传文件: {file_obj.name}")
    
    # 如果上传文件为空或处理失败，使用默认目录
    if not pdf_files:
        print("使用默认PDF目录")
        # 确保pdf_directory是字符串，不是列表
        if isinstance(pdf_directory, list):
            if pdf_directory:
                pdf_directory = pdf_directory[0]  # 取第一个元素
            else:
                pdf_directory = "/commondocument/group6/MEDagent_demo/RAGpdf"
        
        # 检查目录是否存在
        if not os.path.exists(pdf_directory):
            print(f"目录不存在: {pdf_directory}，尝试其他路径")
            # 尝试其他可能路径
            possible_paths = [
                "RAGpdf",
                "./RAGpdf",
                "../RAGpdf",
                "/commondocument/group6/MEDagent_demo/RAGpdf"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pdf_directory = path
                    print(f"使用路径: {pdf_directory}")
                    break
            else:
                print("警告: 未找到PDF目录")
                return None
        
        # 遍历目录获取PDF文件
        try:
            for root, dirs, files in os.walk(pdf_directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        pdf_files.append(full_path)
                        print(f"找到PDF文件: {file}")
        except Exception as e:
            print(f"遍历目录时出错: {e}")
            return None
    
    if not pdf_files:
        print("警告: 未找到任何PDF文件")
        return None
    
    print(f"总共找到 {len(pdf_files)} 个PDF文件")
    
    # 加载所有PDF文档
    documents = []
    failed_files = []
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 为每个文档添加源文件信息
            for doc in docs:
                doc.metadata['source'] = os.path.basename(pdf_path)
            
            documents.extend(docs)
            print(f"成功加载: {os.path.basename(pdf_path)} ({len(docs)}页)")
            
        except Exception as e:
            print(f"加载PDF时出错 {pdf_path}: {e}")
            failed_files.append(pdf_path)
    
    if not documents:
        print("警告: 没有成功加载任何PDF文档")
        return None
        
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        
        try:
            vectorstore_global = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        except Exception as e:
            print(f"持久化向量数据库失败，切换到内存模式: {e}")
            vectorstore_global = Chroma.from_documents(
                documents=texts,
                embedding=embeddings
            )
        st.session_state.vectorstore = vectorstore_global
        return vectorstore_global
    except Exception as e:
        print(f"创建向量数据库时出错: {e}")
        return None

def get_rag_context(user_info_json, context_type="screening"):
    """获取RAG相关的上下文信息"""
    if st.session_state.vectorstore is None:
        return [{"error": "RAG系统未正确初始化，使用基础知识库进行建议"}]
    
    try:
        gender = user_info_json.get("gender", "")
        age = user_info_json.get("age", "")
        family_history = user_info_json.get("family_history", "")
        
        if context_type == "screening":
            queries = [
                f"乳腺癌筛查指南 {age}岁{gender}",
                f"乳腺癌家族史风险评估 {family_history}",
                "乳腺X光筛查建议",
                f"乳腺癌风险因素 {age}岁女性",
                "乳腺癌预防建议"
            ]
        else:  # prognosis
            queries = [
                f"乳腺癌预后 {age}岁{gender}",
                f"乳腺癌治疗方案 {family_history}",
                "乳腺癌分子分型预后",
                f"乳腺癌康复建议 {age}岁女性",
                "乳腺癌治疗副作用"
            ]
        
        rag_contexts = []
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        for query in queries:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                rag_contexts.append({
                    "query": query,
                    "source": doc.metadata.get('source', '未知'),
                    "content": doc.page_content[:500] + "..."
                })
        
        return rag_contexts[:5]
        
    except Exception as e:
        return [{"error": f"RAG检索时出错: {str(e)}"}]

def format_user_info_for_display(user_info_json):
    """格式化用户信息"""
    return {
        "性别": user_info_json.get("gender", "未提供"),
        "年龄": f"{user_info_json.get('age', '未提供')}岁",
        "家族病情史": user_info_json.get("family_history", "未提供"),
        "个人病情史": user_info_json.get("personal_history", "未提供"),
        "生育史": user_info_json.get("reproductive_history", "未提供"),
        "激素使用": user_info_json.get("hormone_use", "未提供"),
        "生活方式": user_info_json.get("lifestyle", "未提供")
    }

# ==================== JSON提取函数 ====================
def extract_json_from_text(text):
    """从文本中提取JSON对象"""
    # 预处理：移除控制字符和多余空白
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # 移除控制字符
    text = re.sub(r'\s+', ' ', text.strip())  # 规范化空白
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"JSON解析错误: {str(e)}")
            return None
    else:
        st.error("未找到JSON对象")
        return None

# ==================== Agent定义（完整保留） ====================
questionnaire_review_agent = Agent(
    role="问卷审核专家",
    goal="审核用户填写的问卷是否符合生理常识和逻辑，并提供审核评论",
    backstory="作为一名医疗问卷审核专家，您擅长检查患者提供的信息是否合理、一致，并符合基本生理和医学逻辑。审核标准宽松，仅识别明显错误。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

process_review_agent = Agent(
    role="流程审核专家",
    goal="审核从早筛到诊断再到预后的输出是否有逻辑错误或不一致，并提供审核评论",
    backstory="作为一名医疗流程审核专家，您会逐步检查每个代理的输出是否逻辑一致、基于证据，并未被篡改。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

early_screening_agent = Agent(
    role="乳腺癌早筛专家",
    goal="根据评估用户的乳腺癌风险因素并使用RAG知识库提供早期筛查建议",
    backstory="作为一名专业的乳腺健康筛查专家，您拥有多年的临床经验，擅长识别高风险因素并使用最新研究提供个性化的筛查方案。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

diagnosis_agent = Agent(
    role="乳腺癌诊断专家",
    goal="融合EfficientNet AI自动诊断结果 + 用户症状，提供权威初步诊断",
    backstory="作为一名经验丰富的乳腺肿瘤诊断专家，您精通AI辅助诊断，能将EfficientNet模型的99.35%概率输出与临床症状结合，生成高可信诊断。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

prognosis_agent = Agent(
    role="乳腺癌预后评估专家",
    goal="评估患者的预后情况并制定个性化的治疗计划",
    backstory="作为一名乳腺肿瘤治疗专家，您擅长根据癌症分期、分子分型和患者整体健康状况制定最佳治疗策略和预后评估。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

expert_review_agent = Agent(
    role="首席乳腺癌专家",
    goal="审查所有专业意见并提供最终的综合建议",
    backstory="作为乳腺肿瘤领域的权威专家，您拥有全面的知识和经验，能够整合多学科意见，为患者提供最权威的建议。请用中文输出。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ==================== Task定义（完整保留原逻辑） ====================
questionnaire_review_task = Task(
    description="""审核用户提供的JSON问卷信息：
    用户信息JSON: {user_info_json}
    
    请宽松检查以下内容，确保问卷数据有效：
    1. 每个字段是否符合基本生理常识：
       - 年龄：0-150岁之间（允许宽松范围）。
       - 性别：男性、女性或未提供。
       - 初潮年龄（如提供）：5-20岁之间（仅限女性，宽松范围）。
       - BMI（如提供）：5-60之间（宽松范围）。
       - 家族病情史、个人病情史、生育史、激素使用、生活方式：仅检查明显荒谬内容（如“家族病情史：宠物患癌”）。
    2. 信息是否自相矛盾：
       - 例如，年龄5岁但报告生育史。
       - 例如，性别为男性但报告怀孕。
    3. 只要没有明显的逻辑错误或荒谬内容，即视为有效问卷，批准通过（approved: true）。
    4. 如果信息不完整（如家族病情史未提供具体病史），在issues中记录建议改进但不拒绝。
    5. 在review_comments中提供简要审核总结，例如“问卷信息合理，无明显逻辑错误”或“问卷信息基本合理，建议补充家族病情史细节”。
    
    输出必须是JSON格式，仅包含以下内容，不添加任何额外文本、注释或解释：
    {
     "review_comments": "",
     "issues": [],
     "reviewed_by_expert1": true,
     "approved": true
    }""",
    agent=questionnaire_review_agent,
    expected_output="问卷审核结果，包括审核评论、任何问题和批准状态"
)

early_screening_task = Task(
    description="""根据用户提供的JSON信息评估乳腺癌风险：
    用户信息JSON: {user_info_json}
    
    格式化后的用户信息:
    {formatted_user_info}
    
    基于RAG知识库检索到的最新乳腺癌筛查信息：
    {rag_context}
    
    PubMed最新文献补充（来自med_rag_app）：
    {pubmed_supplement}
    
    请评估以下风险因素：
    1. 年龄和性别
    2. 家族病情史
    3. 个人乳腺疾病史
    4. 生育和激素因素
    5. 生活方式因素

    请根据RAG知识库和PubMed最新研究提供个性化的筛查方案建议，包括：
    - 推荐筛查方法（乳腺自检、临床检查、乳腺X光、超声等）
    - 筛查频率建议
    - 起始年龄建议
    - 降低风险的生活方式建议
    
    请确保建议基于最新的医学指南和研究，并在rag_sources字段中列出RAG知识库和PubMed检索的具体来源（包括查询和内容摘要）。输出必须是严格的JSON格式，仅包含指定字段，不添加任何额外文本、注释或解释。
    
    输出必须是JSON格式，仅包含以下内容：
    {
     "risk_assessment": "",
     "screening_recommendations": [],
     "references": [],
     "rag_sources": []
    }""",
    agent=early_screening_agent,
    expected_output="基于最新研究的详细乳腺癌风险评估和个性化筛查建议"
)

diagnosis_task = Task(
    description="""【核心任务】融合EfficientNet AI自动诊断结果与用户症状，提供权威初步诊断：
    
    【AI自动影像诊断结果】：
    {efficientnet_result}
    
    用户提供的症状和检查结果JSON: 
    {symptoms_json}
    
    格式化后的症状和检查结果:
    {formatted_symptoms}
    
    请严格按照以下逻辑生成诊断：
    1. 若 predicted_class == "Cancer" 且 probability >= 0.95 → 初步诊断为“高度疑似乳腺癌”
    2. 若 probability >= 0.70 → “中度疑似乳腺癌”
    3. 若 probability < 0.70 → “低度疑似，需进一步检查”
    4. 必须推荐“活检”作为金标准
    5. 紧急程度：probability >= 0.95 → “高”；>=0.7 → “中”；else “低”
    6. 专科推荐：乳腺外科
    
    输出必须是严格JSON格式，仅包含以下内容，不添加任何额外文本、注释或解释：
    {
     "possible_conditions": ["乳腺癌"],
     "recommended_tests": ["活检", "乳腺MRI", "超声引导穿刺"],
     "preliminary_diagnosis": "高度疑似乳腺癌（AI概率99.35%）",
     "specialist_referral": "乳腺外科",
     "urgency_level": "高"
    }""",
    agent=diagnosis_agent,
    expected_output="融合AI自动诊断的诊断意见"
)

prognosis_task = Task(
    description="""根据诊断专家的结果评估预后和制定治疗计划：
    
    基于RAG知识库检索到的最新乳腺癌预后信息：
    {rag_context}
    
    PubMed最新文献补充（来自med_rag_app）：
    {pubmed_supplement}
    
    请提供：
    1. 预后评估（基于癌症分期、分级和分子分型）
    2. 推荐的治疗方案（手术、放疗、化疗、靶向治疗、内分泌治疗等）
    3. 治疗预期效果和可能的副作用
    4. 康复和生活质量建议
    
    请根据RAG知识库和PubMed最新研究提供个性化的预后评估和治疗建议，确保建议基于最新的医学指南和研究，并在rag_sources字段中列出RAG知识库和PubMed检索的具体来源（包括查询和内容摘要）。输出必须是严格的JSON格式，仅包含指定字段，不添加任何额外文本、注释或解释。
    
    输出必须是JSON格式，仅包含以下内容：
    {
     "prognosis_assessment": "",
     "treatment_options": [],
     "expected_outcomes": [],
     "potential_side_effects": [],
     "rehabilitation_advice": [],
     "rag_sources": []
    }""",
    agent=prognosis_agent,
    expected_output="全面的预后评估和个性化治疗计划"
)

process_review_task = Task(
    description="""审核整个流程输出：
    请使用上下文中的早筛任务输出、诊断任务输出和预后任务输出进行审核。
    
    请逐步思考并检查：
    1. 每个步骤的输出是否逻辑一致（例如，早筛风险评估是否与诊断匹配）。
    2. 是否有任何不合理或错误的地方。
    3. 输出是否基于输入和证据，未被不当更改。
    4. 在review_comments中提供简要审核总结，例如“所有步骤输出逻辑一致，基于输入和证据”或“诊断与早筛建议存在轻微不一致，建议进一步核查”。
    5. 如果无误，仅审核一次并添加标记；如果有问题，在issues中记录并建议重跑。
    
    输出必须是JSON格式，仅包含以下内容，不添加任何额外文本、注释或解释：
    {
     "review_comments": "",
     "issues": [],
     "reviewed_by_expert2": true,
     "approved": true
    }""",
    agent=process_review_agent,
    context=[early_screening_task, diagnosis_task, prognosis_task],
    expected_output="流程审核结果，包括审核评论、任何问题和批准状态"
)

expert_review_task = Task(
    description="""作为首席乳腺癌专家，请审查早筛专家、诊断专家和预后专家的意见，
    整合所有信息并提供最终的综合建议。
    
    请使用上下文中的问卷审核任务输出和流程审核任务输出。
    
    请：
    1. 审查所有建议的一致性和准确性
    2. 提供最终的综合建议
    3. 指出任何需要特别注意的事项
    4. 建议下一步行动
    5. 整合问卷审核和流程审核的标记
    
    确保所有字符串字段（如overall_assessment、integrated_recommendations等）不包含换行符（\n）、制表符（\t）或其他控制字符，必要时使用空格替换或正确转义以确保JSON格式有效。输出必须是严格的JSON格式，仅包含以下内容，不添加任何额外文本、注释或解释：
    {
     "overall_assessment": "",
     "integrated_recommendations": [],
     "special_considerations": [],
     "next_steps": [],
     "confidence_level": "",
     "reviewed_by_expert1": true,
     "reviewed_by_expert2": true
    }""",
    agent=expert_review_agent,
    context=[questionnaire_review_task, early_screening_task, diagnosis_task, prognosis_task, process_review_task],
    expected_output="权威的综合建议和明确的下一步指导，包括审核标记"
)

# ==================== Crew定义 ====================
breast_cancer_crew = Crew(
    agents=[questionnaire_review_agent, early_screening_agent, diagnosis_agent, prognosis_agent, process_review_agent, expert_review_agent],
    tasks=[questionnaire_review_task, early_screening_task, diagnosis_task, prognosis_task, process_review_task, expert_review_task],
    verbose=True,
    process=Process.sequential,
    step_callback=lambda step_output: None
)

# ==================== 运行系统的函数 ====================
def run_breast_cancer_analysis(user_info_json, uploaded_files=None):
    """运行乳腺癌分析和推荐系统"""
    
    if not st.session_state.ai_done or st.session_state.efficientnet_result.get("status") != "success":
        st.warning("请先点击【开始AI诊断】按钮运行影像诊断")
        return None, None, None, None
    
    setup_rag_knowledge_base(uploaded_files)
    
    pubmed_supplement = get_med_rag_pubmed_supplement(user_info_json)
    rag_context_screening = get_rag_context(user_info_json, context_type="screening")
    rag_context_prognosis = get_rag_context(user_info_json, context_type="prognosis")
    formatted_user_info = json.dumps(format_user_info_for_display(user_info_json), ensure_ascii=False, indent=2)
    
    result = breast_cancer_crew.kickoff(
        inputs={
            "user_info_json": json.dumps(user_info_json, ensure_ascii=False, indent=2),
            "formatted_user_info": formatted_user_info,
            "symptoms_json": json.dumps({}, ensure_ascii=False, indent=2),
            "formatted_symptoms": json.dumps({}, ensure_ascii=False, indent=2),
            "rag_context": json.dumps(rag_context_screening, ensure_ascii=False, indent=2),
            "pubmed_supplement": json.dumps(pubmed_supplement, ensure_ascii=False, indent=2),
            "efficientnet_result": json.dumps(st.session_state.efficientnet_result, ensure_ascii=False, indent=2)
        }
    )
    
    return result, rag_context_screening, rag_context_prognosis, pubmed_supplement

# ==================== Streamlit 美化前端 ====================
st.set_page_config(page_title="乳腺癌AI诊疗系统", layout="wide")
st.markdown("<h1 style='text-align: center; color: #e91e63;'>乳腺癌智能早筛、诊疗及预后系统</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .big-font {font-size: 28px !important; font-weight: bold; color: #e91e63;}
    .ai-box {background: linear-gradient(90deg, #fce4ec, #f8bbd0); padding: 25px; border-radius: 18px; 
             border: 3px solid #e91e63; margin: 20px 0; box-shadow: 0 6px 16px rgba(233,30,99,0.2);}
    .btn-ai {background: #e91e63 !important; color: white !important;}
    .stButton>button {border-radius: 12px; height: 3.5em; font-size: 20px; font-weight: bold;}
    .form-box {background: #f9f9fb; padding: 35px; border-radius: 18px; box-shadow: 0 8px 20px rgba(0,0,0,0.1);}
    .header-icon {font-size: 36px; margin-right: 15px;}
</style>
""", unsafe_allow_html=True)

# AI诊断按钮
st.markdown("### AI影像诊断 (EfficientNet)")
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("开始AI诊断", key="ai_btn"):
        run_efficientnet_diagnosis()

# 显示AI结果
if st.session_state.ai_done and st.session_state.efficientnet_result and st.session_state.efficientnet_result.get("status") == "success":
    st.markdown(f"""
    <div class="ai-box">
        <p class="big-font"><span class="header-icon">AI诊断结果</span></p>
        <p><strong>预测：</strong> <span style="color:#e91e63;font-size:22px;">{st.session_state.efficientnet_result['predicted_class']}</span></p>
        <p><strong>恶性概率：</strong> <span style="font-size:26px;color:#c2185b;">{st.session_state.efficientnet_result['probability']:.2%}</span></p>
        <p><strong>置信度：</strong> <span style="color:#2e7d32;font-weight:bold;">{'极高' if st.session_state.efficientnet_result['probability'] > 0.95 else '较高'}</span></p>
    </div>
    """, unsafe_allow_html=True)

# 输入表单
with st.form("user_input_form"):
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.markdown("### 患者基本信息")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("性别", ["女性", "男性"])
        age = st.number_input("年龄", min_value=0, max_value=120, value=45)
        family_history = st.text_input("家族病情史", value="母亲50岁时患乳腺癌")
    with col2:
        personal_history = st.text_input("个人病情史", value="无特殊疾病")
        reproductive_history = st.text_input("生育史", value="初潮年龄12岁，未生育")
        hormone_use = st.text_input("激素使用", value="曾使用口服避孕药5年")
    
    st.markdown("### 知识库（可选）")
    uploaded_files = st.file_uploader("上传PDF指南", type=["pdf"], accept_multiple_files=True)
    
    submitted = st.form_submit_button("开始AI会诊")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if not st.session_state.ai_done:
        st.error("请先点击【开始AI诊断】按钮运行影像诊断！")
    else:
        user_info_json = {
            "gender": gender,
            "age": age,
            "family_history": family_history,
            "personal_history": personal_history,
            "reproductive_history": reproductive_history,
            "hormone_use": hormone_use,
            "lifestyle": "偶尔饮酒，不吸烟，BMI 26"
        }
        
        with st.spinner("AI医生团正在会诊..."):
            result, rag_context_screening, rag_context_prognosis, pubmed_supplement = run_breast_cancer_analysis(
                user_info_json, uploaded_files
            )
        
        if result and hasattr(result, 'tasks_output'):
            st.success("会诊完成！")
            
            st.header("完整分析结果")
            st.subheader("用户信息")
            st.json(format_user_info_for_display(user_info_json))
            
            for i, output in enumerate(result.tasks_output):
                try:
                    parsed_output = extract_json_from_text(output.raw)
                    if parsed_output is None:
                        continue
                    agent_name = output.agent if output.agent else f"Agent {i+1}"
                    st.subheader(f"{agent_name} 输出")
                    
                    if agent_name == "问卷审核专家":
                        st.write("**是否审核**: ", "是" if parsed_output.get("reviewed_by_expert1") else "否")
                        st.write("**是否通过**: ", "是" if parsed_output.get("approved") else "否")
                    
                    elif agent_name == "乳腺癌早筛专家":
                        st.write("**风险评估**: ", parsed_output.get("risk_assessment", ""))
                        st.write("**筛查建议**: ", parsed_output.get("screening_recommendations", []))
                        st.write("**参考文献**: ", parsed_output.get("references", []))
                        st.write("**RAG来源**: ", parsed_output.get("rag_sources", []))
                        with st.expander("查看RAG检索详情"):
                            st.write("**本地知识库检索**:", rag_context_screening)
                            st.write("**PubMed检索**:", pubmed_supplement)
                    
                    elif agent_name == "乳腺癌诊断专家":
                        st.write("**可能疾病**: ", parsed_output.get("possible_conditions", []))
                        st.write("**推荐检查**: ", parsed_output.get("recommended_tests", []))
                        st.write("**初步诊断**: ", parsed_output.get("preliminary_diagnosis", ""))
                        st.write("**专科推荐**: ", parsed_output.get("specialist_referral", ""))
                        st.write("**紧急程度**: ", parsed_output.get("urgency_level", ""))
                    
                    elif agent_name == "乳腺癌预后评估专家":
                        st.write("**预后评估**: ", parsed_output.get("prognosis_assessment", ""))
                        st.write("**治疗方案**: ", parsed_output.get("treatment_options", []))
                        st.write("**预期效果**: ", parsed_output.get("expected_outcomes", []))
                        st.write("**潜在副作用**: ", parsed_output.get("potential_side_effects", []))
                        st.write("**康复建议**: ", parsed_output.get("rehabilitation_advice", []))
                        st.write("**RAG来源**: ", parsed_output.get("rag_sources", []))
                        with st.expander("查看RAG检索详情"):
                            st.write("**本地知识库检索**:", rag_context_prognosis)
                            st.write("**PubMed检索**:", pubmed_supplement)
                    
                    elif agent_name == "流程审核专家":
                        st.write("**审核评论**: ", parsed_output.get("review_comments", ""))
                        st.write("**问题**: ", parsed_output.get("issues", []))
                        st.write("**是否审核**: ", "是" if parsed_output.get("reviewed_by_expert2") else "否")
                        st.write("**是否通过**: ", "是" if parsed_output.get("approved") else "否")
                    
                    elif agent_name == "首席乳腺癌专家":
                        st.write("**总体评估**: ", parsed_output.get("overall_assessment", ""))
                        st.write("**综合建议**: ", parsed_output.get("integrated_recommendations", []))
                        st.write("**特别注意事项**: ", parsed_output.get("special_considerations", []))
                        st.write("**下一步行动**: ", parsed_output.get("next_steps", []))
                        st.write("**置信度**: ", parsed_output.get("confidence_level", ""))
                        st.write("**问卷审核标记**: ", "是" if parsed_output.get("reviewed_by_expert1") else "否")
                        st.write("**流程审核标记**: ", "是" if parsed_output.get("reviewed_by_expert2") else "否")
                    
                except Exception as e:
                    st.error(f"无法解析Agent {i+1}的输出: {output.raw}, 错误: {str(e)}")
        else:
            st.error("分析结果为空或格式不正确")

if __name__ == "__main__":
    pass