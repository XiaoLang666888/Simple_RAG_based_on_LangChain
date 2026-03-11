# 导入必要的库
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
import os

def init_qwen_llm():
    """
    初始化通义千问大模型（对话式，贴合问答场景）
    """
    # 验证环境变量是否配置成功
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先配置")
    
    # 初始化对话式大模型
    chat_llm = ChatTongyi(
        model="qwen-turbo",  # 可选 qwen-plus / qwen-max 提升回答精度
        temperature=0.3,     # 法律问题设为低随机性，保证回答严谨
        max_tokens=1024     # 足够容纳详细法律回答
    )
    return chat_llm

def get_answer_from_llm(question, chat_llm):
    """
    纯大模型回答单个问题（无 RAG）
    :param question: 单个用户问题
    :param chat_llm: 初始化好的对话式大模型
    :return: 大模型回答结果
    """
    try:
        # 构造人类消息
        messages = [HumanMessage(content=question)]
        # 调用大模型
        response = chat_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"调用大模型失败：{str(e)}"

if __name__ == "__main__":
    # 1. 初始化大模型
    print("正在初始化通义千问大模型...")
    try:
        qwen_chat_llm = init_qwen_llm()
        print("大模型初始化成功！\n")
    except ValueError as e:
        print(f"初始化失败：{e}")
        exit(1)
    
    # 2. 定义待查询的问题列表
    test_questions = [
        "借款人去世，继承人是否应履行偿还义务？",
        "如何通过法律手段应对民间借贷纠纷？",
        "你现在是精通中国法律的法官，请对以下案件做出分析：2012年5月1日，原告xxx在被告xxxx购买“玉兔牌”香肠15包，其中价值558.6元的14包香肠已过保质期。xxx结账后到服务台索赔，协商未果诉至法院，要求xxxx店支付14包香肠售价十倍的赔偿金5586元。买了超过保质期的香肠可以找商家赔偿吗？"
    ]
    
    # 3. 批量遍历问题，逐一获取回答
    for idx, question in enumerate(test_questions, 1):
        print(f"===== 正在处理第 {idx} 个问题 =====")
        print(f"问题：{question}\n")
        print("大模型正在思考回答...")
        llm_answer = get_answer_from_llm(question, qwen_chat_llm)
        
        # 4. 输出单个问题的回答结果
        print(f"\n第 {idx} 个问题的回答结果：")
        print(llm_answer)
        print("-" * 80 + "\n")  # 分隔线，方便区分不同问题的结果
    
    print("所有问题查询完成！")