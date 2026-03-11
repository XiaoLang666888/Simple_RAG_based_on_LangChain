# 任务1：数据准备全流程
# 优先设置 HuggingFace 镜像以避免 SSL 证书问题
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"  # 绕过证书校验（内网/学校网络常用临时方案）

# 导入所需依赖（在设置镜像后再导入）
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import chardet
import re

# ===================== 步骤1：数据提取 =====================
def load_law_csv(file_name):
    """
    加载法律CSV文件
    :param file_name: CSV文件名
    :return: 原始Document列表
    """
    # 识别文件编码
    try:
        with open(file_name, "rb") as f:
            raw_data = f.read()
            encoding_result = chardet.detect(raw_data)
            file_encoding = encoding_result["encoding"]
            print(f"识别文件编码：{file_encoding}（可信度：{encoding_result['confidence']:.2f}）")
    except Exception as e:
        print(f"编码识别失败，默认使用utf-8-sig：{e}")
        file_encoding = "utf-8-sig"

    # 加载CSV文件
    loader = CSVLoader(
        file_path=file_name,
        encoding=file_encoding,
        source_column="data",  # 直接使用CSV中的data列作为文本内容
        csv_args={
            "fieldnames": ["data"],  # 指定列名
            "delimiter": ",",   # CSV分隔符
            "skipinitialspace": True  # 忽略分隔符后的空格
        }
    )

    docs = loader.load()
    print(f"原始数据加载成功！共 {len(docs)} 条记录")
    return docs
    

# ===================== 步骤2：数据清洗（移除无效信息） =====================

def clean_law_documents(documents):
    cleaned_docs = []
    seen_texts = set()  # 用集合记录已出现的清洗后文本，去重
    for idx, doc in enumerate(documents):
        text = doc.page_content
        # 1. 先执行所有清洗操作，统一文本格式
        # 移除多余换行符、空格
        text = text.replace("\r", "").replace("  ", " ").strip()
        # 移除特殊字符（保留中文、英文、数字、常见标点）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】《》,.!?;:\(\)\[\]]", "", text)
        
        # 2. 再判断是否为空、过短、重复
        if not text:
            print(f"跳过第{idx+1}条空数据")
            continue
        if text in seen_texts:
            print(f"跳过第{idx+1}条重复数据")
            continue
        if len(text) < 10:
            print(f"跳过第{idx+1}条过短数据（长度：{len(text)}）")
            continue
        
        # 3. 记录已出现的文本，生成清洗后的Document
        seen_texts.add(text)
        cleaned_doc = doc.model_copy()
        cleaned_doc.page_content = text
        cleaned_docs.append(cleaned_doc)
    
    print(f"数据清洗完成！剩余 {len(cleaned_docs)} 条有效记录，过滤了 {len(documents) - len(cleaned_docs)} 条无效记录")
    return cleaned_docs

# ===================== 步骤3：文本分割（适配向量检索） =====================
def split_law_documents(cleaned_docs):  
    text_splitter = CharacterTextSplitter(
        separator="；",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(cleaned_docs)

    # 第一步：给每个分割后的文档添加唯一编号（从0开始递增）
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = idx  # 添加chunk_id元数据
    
    # 第二步：分割后去重
    unique_split_docs = []
    seen_content = set()  # 记录已出现的文本内容
    for doc in split_docs:
        content = doc.page_content.strip()
        # 判断内容非空、不重复、且长度≥最小阈值
        if content not in seen_content and len(content) > 0: 
            seen_content.add(content)
            unique_split_docs.append(doc)                 
    
    # 第三步：更新编号（去重+过滤后重新排序编号，更规整）
    for new_idx, doc in enumerate(unique_split_docs):
        doc.metadata["chunk_id"] = new_idx
    
    # 打印分割和去重后的统计信息
    print(f"文本分割完成！原始生成 {len(split_docs)} 个文本片段")
    print(f"分割后去重完成！剩余 {len(unique_split_docs)} 个有效唯一文本片段")
    print(f"示例片段：{unique_split_docs[1].page_content[:150]}...")
    return unique_split_docs

# ===================== 步骤4：文本向量化（生成向量表示） =====================
def init_embedding_model():
    """
    初始化嵌入模型（BAAI/bge-base-zh-v1.5）
    :return: 初始化后的embedding模型
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",  # 实验模型，支持中文
        model_kwargs={"device": "cpu"},  
        encode_kwargs={"normalize_embeddings": True}  # 归一化向量，提升检索效果
    )
    print("嵌入模型初始化成功！")
    return embedding_model

# ===================== 步骤5：向量入库（存入FAISS并保存本地） =====================
def build_faiss_vector_db(split_docs, embedding_model):
    """
    构建FAISS向量数据库
    :param split_docs: 分割后的文本片段列表
    :param embedding_model: 嵌入模型
    :return: FAISS向量数据库对象
    """
    # 从文本片段构建向量库
    db = FAISS.from_documents(split_docs, embedding_model)
    # 保存向量库到本地（后续可直接加载，无需重复向量化）
    db.save_local("law_rag_faiss_db")
    print(f"FAISS向量数据库构建完成！")
    print(f"向量库已保存到本地目录：law_rag_faiss_db")
    return db

# ===================== 主程序：执行任务1全流程 =====================
if __name__ == "__main__":
    # 1. 配置文件名
    CSV_FILE_NAME = "law_data_3k.csv"
    VECTOR_DB_PATH = "law_rag_faiss_db"  # 向量库保存路径

    # 2. 按流程执行
    print("===== 开始执行任务1：数据准备 =====")
    raw_docs = load_law_csv(CSV_FILE_NAME)
    cleaned_docs = clean_law_documents(raw_docs)
    split_docs = split_law_documents(cleaned_docs)
    embedding_model = init_embedding_model()
    # 检查向量库是否已存在，存在则直接加载
    import os
    if os.path.exists(VECTOR_DB_PATH) and len(os.listdir(VECTOR_DB_PATH)) > 0:
        print(f"检测到已存在向量库，直接加载：{VECTOR_DB_PATH}")
        faiss_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True  # 允许加载本地文件
        )
    else:
        # 不存在则构建并保存
        faiss_db = build_faiss_vector_db(split_docs, embedding_model)

    # 3. 验证向量库检索功能
    print("\n===== 验证向量库检索功能 =====")
    # test_query = "借款人去世，继承人是否应履行偿还义务？"
    # test_query = "如何通过法律手段应对民间借贷纠纷？"
    test_query = "你现在是精通中国法律的法官，请对以下案件做出分析：2012年5月1日，原告xxx在被告xxxx购买“玉兔牌”香肠15包，其中价值558.6元的14包香肠已过保质期。xxx结账后到服务台索赔，协商未果诉至法院，要求xxxx店支付14包香肠售价十倍的赔偿金5586元。买了超过保质期的香肠可以找商家赔偿吗？"
    similar_docs_with_score = faiss_db.similarity_search_with_score(test_query, k=5)  # 带相似度得分
    print(f"查询问题：{test_query}")

    if not similar_docs_with_score:
        print("未检索到任何相关文本！")
    else:
        for i, (doc, score) in enumerate(similar_docs_with_score, 1):
            chunk_id = doc.metadata.get("chunk_id", "未知编号")  # 获取编号
            print(f"\n第{i}条相关文本（编号：{chunk_id}，相似度：{score:.4f}）：")
            print(doc.page_content)
    print("\n===== 任务执行完毕 =====")