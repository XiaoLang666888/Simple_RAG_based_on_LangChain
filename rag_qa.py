# rag_qa.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# 1. 导入依赖
from data_prepare import init_embedding_model  # 复用任务1的代码
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Tongyi  # 通义千问
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 2. 加载生成的向量库
def load_vector_db():
    """加载向量库，构建检索器（任务2核心）"""
    embedding_model = init_embedding_model()
    vector_db_path = "law_rag_faiss_db"
    
    # 加载本地向量库
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # 构建检索器
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # 召回Top5相关条文
    )
    print("检索器构建完成！")
    return retriever

# 3. 构造Prompt模板
def build_prompt():
    template = """你是专业的法律知识问答助手。你需要使用以下检索到的上下文片段来回答问题，禁止根据常识和已知信息回答问题。如果你不知道答案，直接回答“未找到相关答案”。
                Question: {question}
                Context: {context}
                Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

# 4. 初始化LLM（通义千问-turbo）
def init_llm():
    # 配置通义千问API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到环境变量 DASHSCOPE_API_KEY。请在运行前设置，如：$env:DASHSCOPE_API_KEY='你的Key'"
        )
    
    llm = Tongyi(model="qwen-turbo")  # 轻量版适合快速测试
    print("LLM初始化完成！")
    return llm

# 5. 构建完整RAG问答链
def build_rag_chain():
    retriever = load_vector_db()
    prompt = build_prompt()
    llm = init_llm()
    
    # 构建问答链：检索→Prompt注入→LLM生成→输出解析
    rag_chain = (
        {
            "context": retriever ,
            "question": RunnablePassthrough()  # 传递用户问题
        }
        | prompt  # 注入检索结果到Prompt
        | llm  # LLM生成回答
        | StrOutputParser()  # 解析输出结果
    )
    print("RAG问答链构建完成！")
    return rag_chain

# 6. 测试实验文档要求的问题
def test_rag_chain():
    rag_chain = build_rag_chain()
    test_questions = [
        "借款人去世，继承人是否应履行偿还义务？",
        "如何通过法律手段应对民间借贷纠纷？",
        "你现在是精通中国法律的法官，请对以下案件做出分析：2012年5月1日，原告xxx在被告xxxx购买“玉兔牌”香肠15包，其中价值558.6元的14包香肠已过保质期。xxx结账后到服务台索赔，协商未果诉至法院，要求xxxx店支付14包香肠售价十倍的赔偿金5586元。买了超过保质期的香肠可以找商家赔偿吗？"
    ]
    
    # 执行测试并输出结果
    for i, question in enumerate(test_questions, 1):
        print(f"\n===== 测试问题{i} =====")
        print(f"Question：{question}")
        print("Answer：", end="")
        answer = rag_chain.invoke(question)
        print(answer)

if __name__ == "__main__":
    test_rag_chain()