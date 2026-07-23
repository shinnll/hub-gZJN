"""
手写 Prompt 解析版 ReAct Agent（支持多轮问答）

教学重点：
  1. ReAct 核心循环：Thought → Action → Observation，逐步推理
  2. System Prompt 约束输出格式，Python 正则解析每一步
  3. 对话历史拼接方式：每轮结果追加到 prompt，形成上下文记忆
  4. 停止条件：模型输出 Final Answer 或达到最大步数

◆ 本周改造（多轮问答）：
  原版 run() 每次都重建 messages = [system, user(question)]，跑完即弃，
  是典型的「单轮、无状态」Agent（课件第 4 页：每轮重新开始、无法积累上下文）。
  改造后：
    - run() 增加可选参数 messages，传入则在该历史基础上继续（记忆层 / 短期记忆）
    - 每轮结束把 Final Answer 写回 messages，使后续问题能"看到"之前的问答
    - 新增 ChatSession 会话类，在多次 ask() 之间持久维护同一份 messages
  这正好对应课件「Agent 核心架构 · 记忆层」：短期记忆 = In-Context Memory。

使用方式：
  # 单轮（与原版行为一致，兼容 evaluate.py / serve.py）
  python react_manual.py --question "茅台和五粮液2023年毛利率差多少？"

  # 多轮交互（本周新增）
  python chat.py --mode manual
"""

import os
import re
import json
import time
import logging
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
2. company_lookup(name) - 将公司名称转换为股票代码
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
4. financial_indicator(symbol) - 获取实时财务指标（PE/PB/ROE等）
5. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
- 如果用户的问题是基于之前对话的追问（例如"那五粮液呢？"），请直接利用对话历史中已有的信息回答，无需重复调用工具
"""

# ── 格式解析 ──────────────────────────────────────────────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type":    "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer":  final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m  = _ACTION_RE.search(text)
    input_m   = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type":         "action",
        "thought":      thought_m.group(1).strip() if thought_m else "",
        "action":       action_m.group(1).strip(),
        "action_input": action_input,
    }


# ── ReAct 核心循环 ─────────────────────────────────────────────────────────────

def run(question: Optional[str] = None,
        max_steps: int = 10,
        messages: Optional[list] = None) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环，yield 每一步的结构化结果。

    多轮支持：
      - messages 为 None 时，按原版行为新建 [system, user(question)]（单轮）
      - messages 传入一个会话历史列表时，会在该历史基础上追加本轮 user 提问并继续推理
        （即"记忆层"：短期记忆 = In-Context Memory，跨轮保留上下文）
      函数不会复制 messages，而是直接就地追加，因此调用方持有同一份引用即可获得跨轮记忆。

    每个 yield 的 dict 格式：
      {"step": int, "thought": str, "action": str, "action_input": dict, "observation": str}
    最后一个 yield：
      {"step": int, "thought": str, "type": "final", "answer": str}
    """
    from tools import TOOLS_MAP

    # —— 记忆层接入点：决定本轮的 messages 起点 ——
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if question is not None:
        messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],  # 让模型停在调用工具前
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            # 把最终答案写回历史，供后续追问引用（记忆层）
            messages.append({"role": "assistant", "content": parsed["answer"]})
            yield {
                "step":    step,
                "type":    "final",
                "thought": parsed["thought"],
                "answer":  parsed["answer"],
            }
            return

        if parsed["type"] == "unparseable":
            messages.append({"role": "assistant", "content": llm_output})
            yield {
                "step":        step,
                "type":        "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        # 执行工具
        tool_name  = parsed["action"]
        tool_args  = parsed["action_input"]
        tool_fn    = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        step_result = {
            "step":         step,
            "type":         "action",
            "thought":      parsed["thought"],
            "action":       tool_name,
            "action_input": tool_args,
            "observation":  str(observation),
        }
        yield step_result

        # 将本步结果追加到对话历史
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role":    "user",
            "content": f"Observation: {observation}\n",
        })

    # 超出最大步数，强制终止
    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


class ChatSession:
    """
    多轮对话会话（记忆层载体）。

    在多次 ask() 之间持久维护同一份 messages，使 Agent 能"记住"之前的问答。
    对应课件：短期记忆（In-Context Memory）—— 当前对话的消息历史拼入 Prompt 上下文。

    设计要点：
      - 每次 ask() 都在同一份 messages 上追加，天然形成跨轮上下文
      - max_history：当消息条数超过该上限时，丢弃最早的若干轮（保留 system）。
        这是课件提到的"上下文过长时如何截断或压缩"的轻量实现；
        生产环境可改为用 LLM 对旧轮做摘要（summarization）而非直接丢弃。
    """

    def __init__(self, max_steps: int = 10, max_history: int = 24,
                 system_prompt: str = SYSTEM_PROMPT):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_steps = max_steps
        self.max_history = max_history  # 仅统计 system 之外的消息条数上限

    def ask(self, question: str, max_steps: Optional[int] = None) -> Generator[dict, None, None]:
        """提出一个问题，返回逐步推理的生成器；结束时自动把答案写入会话历史。"""
        yield from run(
            question,
            max_steps=max_steps if max_steps is not None else self.max_steps,
            messages=self.messages,
        )
        self._trim()

    def reset(self) -> None:
        """清空对话历史（保留 system 提示词）。"""
        self.messages = [self.messages[0]]

    def history(self) -> list:
        """返回当前完整的对话历史（供调试 / Web 端展示）。"""
        return list(self.messages)

    def _trim(self) -> None:
        """超过 max_history 时，丢弃最早的若干条（保留 system）。"""
        if len(self.messages) - 1 <= self.max_history:
            return
        # messages[0] 是 system，永不删除
        overflow = (len(self.messages) - 1) - self.max_history
        # 成对丢弃最早的 user/assistant 轮，保持角色交替完整
        removed = 0
        i = 1
        while removed < overflow and i < len(self.messages):
            del self.messages[i]
            removed += 1
        logger.warning("会话历史已裁剪 %d 条，避免上下文溢出", removed)


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",   # cyan
    "action":   "\033[33m",   # yellow
    "obs":      "\033[32m",   # green
    "final":    "\033[35m",   # magenta
    "error":    "\033[31m",   # red
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    """单轮运行并打印（原版行为，兼容 agent.py 入口）。"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: 手写Prompt解析")
    print('='*60)

    start = time.time()
    step_count = 0

    for step_data in run(question, max_steps=max_steps):
        step_count += 1
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
