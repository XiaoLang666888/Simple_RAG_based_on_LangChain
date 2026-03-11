# Simple_RAG_based_on_LangChain
USTC 25年秋 Web信息处理课程实验

代码分为三部分：

data_prepare.py : 用于将 law_data_3k.csv 中的数据完成提取、文本分割、向量化、数据入库等环节。

only_model.py : 用于配置千问模型 api，选定 RAG 中的 LLM。

rag_qa.py : 基于 langchain框架完成由数据检索->prompt构建->结果生成的过程衔接。

