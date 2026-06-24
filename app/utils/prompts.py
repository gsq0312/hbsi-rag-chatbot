"""Prompt templates for RAG Service"""
from typing import List, Dict
from langchain_core.documents import Document as LangchainDocument

ADMISSION_SYSTEM_PROMPT = """你是一位专业的大学招生咨询助手，负责回答学生和家长关于学校招生、专业设置、申请流程、学费资助等方面的问题。

你的职责：
1. 基于提供的知识库内容回答问题，确保信息准确可靠
2. 回答要清晰、友好、专业，适合学生和家长阅读
3. 如果知识库中没有相关信息，诚实地说明，不要编造信息
4. 可以适当引导用户查阅官方网站或联系招生办获取最新信息
5. 回答要使用中文

注意事项：
- 始终保持友好和专业的态度
- 对于具体数字（如分数线、学费金额等），建议用户以官方最新公布为准
- 如果问题超出招生咨询范围，可以礼貌地说明你的职责范围
- 回答要简洁明了，重点突出
"""

QA_PROMPT_TEMPLATE = """基于以下参考信息，回答用户的问题。

参考信息：
{context}

用户问题：{question}

请提供准确、友好的回答。如果参考信息中没有相关内容，请明确说明，并建议用户查阅官方网站或联系招生办公室。
"""


class PromptBuilder:
    @staticmethod
    def build_system_prompt() -> str:
        return ADMISSION_SYSTEM_PROMPT

    @staticmethod
    def build_qa_prompt(question: str, context_documents: List[LangchainDocument], max_context_length: int = 3000) -> str:
        context_parts = []
        current_length = 0
        for i, doc in enumerate(context_documents, 1):
            source = doc.metadata.get('source', '未知来源')
            content = doc.page_content
            context_part = f"[来源{i}: {source}]\n{content}\n"
            context_parts.append(context_part)
            current_length += len(context_part)
            if current_length >= max_context_length:
                break
        context = "\n".join(context_parts)
        return QA_PROMPT_TEMPLATE.format(context=context, question=question)

    @staticmethod
    def build_chat_messages(question: str, context_documents: List[LangchainDocument], conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": PromptBuilder.build_system_prompt()}]
        if conversation_history:
            for msg in conversation_history[:-1]:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({"role": msg['role'], "content": msg['content']})
        user_prompt = PromptBuilder.build_qa_prompt(question, context_documents)
        messages.append({"role": "user", "content": user_prompt})
        return messages


class ConversationHistory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        if len(self._sessions[session_id]) > self.max_history * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history * 2:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, []).copy()

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def clear_all(self) -> None:
        self._sessions.clear()

    def get_session_count(self) -> int:
        return len(self._sessions)


_conversation_history: ConversationHistory = None


def get_conversation_history() -> ConversationHistory:
    global _conversation_history
    if _conversation_history is None:
        _conversation_history = ConversationHistory()
    return _conversation_history
