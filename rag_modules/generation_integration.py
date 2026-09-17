import os

from openai import OpenAI
from typing import List
from langchain_core.documents import Document

class GenerationIntegrationModule:
    def __init__(
        self,
        model_name:str = "deepseek-flash",
        temperature:float = 0.1,
        max_tokens:int=2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def generate_adaptive_answer(
        self,
        question:str,
        documents:List[Document]
    )->str:
        context_parts = []

        for index,doc in enumerate(documents,1):
            content = doc.page_content.strip()

            if content:
                level = doc.metadata.get("retrieval_level","")
                header = f"[证据{index}]"

                if level:
                    header += f"[{level.upper()}]"

                context_parts.append(f"{header}\n{content}")


        context = "\n\n".join(context_parts)

        prompt = f"""
            作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

            检索到的相关信息：
            {context}

            用户问题：{question}

            请提供准确、实用的回答。根据问题的性质：
            - 如果是询问多个菜品，请提供清晰的列表
            - 如果是询问具体制作方法，请提供详细步骤
            - 如果是一般性咨询，请提供综合性回答

            回答依据要求：
            - 事实性结论必须有上述检索信息支持，不补充未提供的用量、
            时间、制作步骤、食材作用或营养结论。
            - 若检索信息只有图路径，只解释路径中明确呈现的关联。
            - 不将 REQUIRES 自动解释为“主要食材”或“不可替代的配料”。
            - 路径中的无方向连接不代表关系箭头方向，不自行补充箭头。
            - 信息不足时直接说明，不用常识补齐。
            - 对关键事实，在相应句子或段落末尾标注来源，例如：[证据 1]。
            - 引用编号必须来自上面提供的证据，不得自行编造编号。
            - 同一结论由多份证据共同支持时，可以标注多个编号。
            - 若证据存在冲突，分别说明各自的内容并引用对应编号。
            - 信息不足的部分明确说明，不用无关证据支持结论。

            回答：
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()