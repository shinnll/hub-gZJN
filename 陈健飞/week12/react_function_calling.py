"""
Function Calling API 版 ReAct Agent（支持多轮问答）

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

◆ 本周改造（多轮问答）：
  与 react_manual.py 同样思路——把 messages 跨轮持久化，实现记忆层 / 短期记忆。
    - run() 增加可选参数 messages，传入则在该历史基础上继续
    - 每轮结束把 Final Answer 写回 messages，供后续追问引用
    - 新增 ChatSession 会话类，在多次 ask() 之间持久维护同一份 messages

使用方式：
  # 单轮（与原版行为一致）
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"

  # 多轮交互（本周新增）
  python chat.py --mode fc
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 如果用户的问题是基于之前对话的追问（例如"那五粮液呢？"），请直接利用对话历史中已有的信息回答，无需重复调用工具
"""


def run(question: Optional[str] = None,
        max_steps: int = 10,
        messages: Optional[list] = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果。

    多轮支持：messages 为 None 时新建 [system, user(question)]（单轮）；
    传入会话历史列表时在其基础上继续（记忆层 / 短期记忆）。
    函数就地追加 messages，调用方持有同一引用即可获得跨轮记忆。

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比。
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    if messages is None:
        messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    if question is not None:
        messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            answer = msg.content or "（模型返回空内容）"
            # 把最终答案写回历史，供后续追问引用（记忆层）
            messages.append({"role": "assistant", "content": answer})
            yield {
                "step":    step,
                "type":    "final",
                "thought": "",
                "answer":  answer,
            }
            return

        # 模型请求调用工具
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


class ChatSession:
    """
    多轮对话会话（记忆层载体），与 react_manual.ChatSession 接口一致。

    在多次 ask() 之间持久维护同一份 messages，使 Agent 能"记住"之前的问答。
    对应课件：短期记忆（In-Context Memory）。
    """

    def __init__(self, max_steps: int = 10, max_history: int = 24,
                 system_prompt: str = FC_SYSTEM_PROMPT):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_steps = max_steps
        self.max_history = max_history

    def ask(self, question: str, max_steps: Optional[int] = None) -> Generator[dict, None, None]:
        yield from run(
            question,
            max_steps=max_steps if max_steps is not None else self.max_steps,
            messages=self.messages,
        )
        self._trim()

    def reset(self) -> None:
        self.messages = [self.messages[0]]

    def history(self) -> list:
        return list(self.messages)

    def _trim(self):
        if len(self.messages) - 1 <= self.max_history:
            return
        overflow = (len(self.messages) - 1) - self.max_history
        removed = 0
        i = 1
        while removed < overflow and i < len(self.messages):
            del self.messages[i]
            removed += 1
        logger.warning("会话历史已裁剪 %d 条，避免上下文溢出", removed)


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
