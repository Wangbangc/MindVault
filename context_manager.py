"""
上下文窗口管理器

提供滑动窗口截断对话历史，控制长时间对话中的 token 成本。
"""

from typing import TypedDict


class Turn(TypedDict):
    role: str  # "user" | "assistant" | "tool"
    content: str


class ContextManager:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: list[Turn] = []

    def add_turn(self, role: str, content: str) -> None:
        """添加一个对话轮次。"""
        self.history.append(Turn(role=role, content=content))

    def get_context(self) -> list[Turn]:
        """返回裁剪后的对话历史（滑动窗口）。"""
        if len(self.history) <= self.max_turns:
            return self.history
        return self.history[-self.max_turns:]

    def count_turns(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        """清空所有历史（新对话时使用）。"""
        self.history.clear()
