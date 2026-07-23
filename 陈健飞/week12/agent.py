"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  # 单轮问答
  python agent.py
  python agent.py --mode manual --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc --question "五粮液近一年股价涨跌幅？"

  # 多轮交互（循环调用 ChatSession.ask，无需 chat.py）
  python agent.py --chat
  python agent.py --chat --mode fc

环境变量：
  DASHSCOPE_API_KEY  必填（manual 版 / serve 用）
  DEEPSEEK_API_KEY   必填（fc 版用，当前默认走 DeepSeek）
  AGENT_MODEL        默认 qwen-max（manual）/ deepseek-v4-flash（fc）
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"


# ── 多轮交互循环（核心：在同一个 ChatSession 上反复调用 ask）─────────────────
COLORS = {
    "thought": "\033[36m", "action": "\033[33m", "obs": "\033[32m",
    "final":   "\033[35m", "error": "\033[31m", "reset": "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _print_step(step_data: dict, mode: str) -> None:
    """打印单步推理结果（与 run_and_print 风格一致）。"""
    stype = step_data["type"]
    if stype == "action":
        print(f"\n[Step {step_data['step']}]")
        if mode == "manual":
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        else:
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
        print(_c("action", f"🔧 Action:  {step_data['action']}"))
        print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
        print(_c("obs",    f"👁  Obs:     {step_data['observation'][:300]}"))
    elif stype == "final":
        print(f"\n{'─'*60}")
        if step_data.get("thought"):
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
    elif stype in ("error", "max_steps"):
        print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def run_chat(mode: str = "manual", max_steps: int = 10) -> None:
    """
    多轮交互入口。

    关键就是下面这个循环：
      - session 在循环外只创建一次，它的 messages 在多次 ask() 之间持久保留（记忆层）
      - 每轮读一句用户输入，调用一次 session.ask(question)，逐步打印推理过程
      - /reset 清空历史、/history 看历史、/exit 退出
    """
    if mode == "manual":
        from react_manual import ChatSession
    else:
        from react_function_calling import ChatSession

    session = ChatSession(max_steps=max_steps)
    print("=" * 60)
    print(f"  ReAct 金融分析 Agent · 多轮对话模式（{mode}）")
    print("  输入问题开始对话；/reset 清空历史，/history 看历史，/exit 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            print("再见！")
            break
        if question == "/reset":
            session.reset()
            print("🧹 已清空对话历史，开始新会话。")
            continue
        if question == "/history":
            print(f"\n{'─'*60}\n当前对话历史（共 {len(session.history())} 条）:")
            for m in session.history():
                content = m["content"]
                preview = content if len(content) <= 200 else content[:200] + " …"
                print(f"[{m['role']}] {preview}")
            print("─" * 60)
            continue

        print(f"\n模型: {os.getenv('AGENT_MODEL', 'qwen-max')}  实现: {mode}")
        for step_data in session.ask(question):
            _print_step(step_data, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--chat", action="store_true",
        help="进入多轮交互模式（循环调用 ChatSession.ask，无需 chat.py）",
    )
    args = parser.parse_args()

    if args.chat:
        run_chat(args.mode, args.max_steps)
    else:
        if args.mode == "manual":
            from react_manual import run_and_print
        else:
            from react_function_calling import run_and_print
        run_and_print(args.question, args.max_steps)
