"""
demo.py —— 发布任务的 Sub Agent 演示

运行：
  cd task_publisher
  python demo.py            # 交互提问：你提一个整体问题，agent 自己拆成 ≤3 步去派发
  python demo.py --web      # web 模式（需 DEEPSEEK_API_KEY + TAVILY_API_KEY，LLM 真拆解）
  python demo.py --serial   # 串行基线，看 A/B 加速比
  python demo.py --no-input # 用内置示例题自动跑（便于演示/自测）

教学重点：你只提一个整体问题 → orchestrator（sub agent）自己分解成 ≤3 子任务
          → 并行派给叶子 worker → 聚合报告。分解是 agent 做的，不是你预先拆好。
"""
import argparse
import json
import logging

logging.basicConfig(level=logging.WARNING)

from orchestrator import run_orchestrator

# 注意：这是一个「整体问题」，没有用任何分隔符预拆——agent 会自己分解它。
SAMPLE = "系统调研一下中国新能源汽车市场的整体表现与未来走向"


def _acquire_task(args) -> str:
    """拿顶层任务：优先 --task；否则交互输入；EOF 或空输入回退示例。"""
    if args.task:
        return args.task
    try:
        hint = ("请输入一个整体任务（agent 会自动帮你拆解并派发，无需自己分步）："
                if not args.serial else "串行基线模式 — 请输入任务：")
        task = input(hint + "\n> ").strip()
    except EOFError:
        task = SAMPLE
    if not task:
        task = SAMPLE
    return task


def main():
    ap = argparse.ArgumentParser(description="发布任务的 Orchestrator Sub Agent 演示")
    ap.add_argument("--web", action="store_true", help="用真实联网 worker（需 key）")
    ap.add_argument("--serial", action="store_true", help="串行基线（A/B 对比）")
    ap.add_argument("--task", default=None, help="自定义顶层任务；不填则运行时交互输入")
    ap.add_argument("--no-input", action="store_true",
                    help="非交互：不填 --task 时用内置示例任务（便于自动演示/自测）")
    args = ap.parse_args()

    task = SAMPLE if args.no_input else _acquire_task(args)

    worker_type = "web" if args.web else "sim"
    print("=== 发布任务的 Orchestrator Sub Agent ===")
    print(f"worker_type={worker_type}  serial={args.serial}")
    print(f"任务：{task}\n")

    out = run_orchestrator(task, worker_type=worker_type, serial=args.serial)

    print("【agent 分解出的子任务（≤3）】")
    for st in out["subtasks"]:
        print(f"  - [{st['id']}] {st['title']}")
    print(f"\n【并行统计】 {json.dumps(out['parallel_stats'], ensure_ascii=False)}")

    print("\n【聚合报告】")
    print(out["final_answer"])


if __name__ == "__main__":
    main()
