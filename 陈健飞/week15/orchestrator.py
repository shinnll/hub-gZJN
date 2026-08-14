"""
orchestrator.py —— 作业核心：发布任务的 Sub Agent（Orchestrator / 分配者）

教学定位（对应 week15 Graph Engineering 主线的「分配机制」）：
  - 这是老师说的「一个 sub agent」：系统的分配中心 / orchestrator 根节点
  - 职责：接收顶层任务 → 拆成 ≤3 子任务 → 并行派发给叶子 worker → 聚合结果
  - 它不亲自干活（那是 worker 的事），只做「拆得对、派得准、聚得清」
  - 两条硬约束从机制层卡死（非仅靠 prompt）：
      1) 最多 3 个子 agent：split 后强制 subtasks[:3]，ThreadPoolExecutor(max_workers=3)
      2) 子 agent 不许再派发：worker 工具集不含 dispatch_subagents（见 workers.py）

复用真实 Demo 的 ReAct / 派发机制，但——
  - 分配者用「LLM 单次结构化拆分 + 确定性并行」（沿用 Demo 踩坑经验，比纯 ReAct 稳）
  - 叶子 worker 默认 sim（模拟，零依赖），可切 web（联网）
  - 无 DEEPSEEK_API_KEY 时，split 自动降级为 agent 启发式框架拆分（agent 自己套框架，
    不靠人在提问里用分隔符预拆），整套仍可独立跑通

【依赖说明】本文件顶部只 import 标准库与 workers（零依赖）；对 llm_client 的 import
延迟到 split_task 内部，因此没有 openai / 没有 key 时 sim 模式依然可跑。
"""
import time
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

from workers import get_worker  # workers.py 零依赖，可安全顶层 import

MAX_SUBAGENTS = 3  # 老师要求：最多 3 个子 agent


# ──────────────────────────────────────────────────────────────────────────
# 1) 拆分（TaskAllocator 核心）
# ──────────────────────────────────────────────────────────────────────────
def _heuristic_split(task: str, max_subagents: int = MAX_SUBAGENTS) -> list[dict]:
    """无 LLM 时的 agent 启发式拆解：对一个「整体任务」套用通用调研框架，生成 ≤3 子任务。

    关键点：这不是让人在提问时用 ；/|/并且 等分隔符预拆，而是 agent 自己按框架把
    一个问题拆成几个可并行调研的角度。人只负责提一个整体问题。
    """
    base = re.sub(r"\s+$", "", task.strip().rstrip("。. "))
    framework = [
        ("现状与背景", f"梳理「{base}」的当前现状、规模与发展背景"),
        ("参与者与格局", f"分析「{base}」的主要参与者、竞争格局与关键力量"),
        ("风险、趋势与建议", f"评估「{base}」的主要风险、未来趋势，并给出可执行建议"),
    ]
    return [{"id": f"s{i + 1}", "title": t, "objective": o}
            for i, (t, o) in enumerate(framework[:max_subagents])]


_SPLIT_SYSTEM = """你是一个任务拆解器。把一个模糊的顶层任务拆成最多 {max_n} 个相互独立、可并行执行的子任务。
只输出 JSON，不要解释。格式：
{{"subtasks": [{{"title": "子任务标题", "objective": "该子任务要达成的具体目标"}}, ...]}}
若任务本身已足够单一，只输出 1 个子任务。子任务数量不超过 {max_n}。"""

_SPLIT_USER = "顶层任务：{task}"


def _extract_json(text: str):
    """从 LLM 输出里尽量抠出 JSON。"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def split_task(task: str, max_subagents: int = MAX_SUBAGENTS,
               use_llm: Optional[bool] = None) -> list[dict]:
    """把顶层任务拆成子任务列表（≤max_subagents）。分解由 agent 完成：
    有 key 用 LLM 真正理解并拆解；无 key 降级为 agent 启发式框架拆解。"""
    if use_llm is None:
        try:
            from llm_client import get_client
            get_client()
            use_llm = True
        except Exception:
            use_llm = False

    if use_llm:
        try:
            from llm_client import llm_chat
            out = llm_chat(_SPLIT_SYSTEM.format(max_n=max_subagents),
                           _SPLIT_USER.format(task=task), temperature=0.0, max_tokens=512)
            data = _extract_json(out)
            subs = data.get("subtasks", []) if isinstance(data, dict) else []
            subs = [s for s in subs if isinstance(s, dict) and s.get("title")]
            if subs:
                subs = subs[:max_subagents]
                return [{
                    "id": f"s{i + 1}",
                    "title": s.get("title", ""),
                    "objective": s.get("objective") or s.get("title", ""),
                } for i, s in enumerate(subs)]
        except Exception as e:
            logger.warning(f"LLM 拆分失败，降级为 agent 启发式框架拆分：{e}")

    return _heuristic_split(task, max_subagents)


# ──────────────────────────────────────────────────────────────────────────
# 2) 派发（确定性并行）
# ──────────────────────────────────────────────────────────────────────────
def dispatch_workers(subtasks: list[dict], worker_type: str = "sim",
                     serial: bool = False, max_workers: int = MAX_SUBAGENTS) -> dict:
    """并行（或串行基线）执行各子任务，返回 {results, wall_clock, serial_sum, speedup, n_workers}。"""
    n = len(subtasks)
    n_workers = min(max_workers, n)
    worker = get_worker(worker_type)

    results = {}
    t0 = time.time()
    if serial:
        # 串行基线：用于 A/B 量化并行收益（Amdahl 定律观察）
        for st in subtasks:
            results[st["id"]] = worker(st)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(worker, st): st["id"] for st in subtasks}
            for fut in futures:
                rid = futures[fut]
                results[rid] = fut.result()
    wall_clock = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for r in results.values()), 2)
    speedup = round(serial_sum / wall_clock, 2) if wall_clock > 0 else 1.0
    return {
        "results": results,
        "wall_clock": wall_clock,
        "serial_sum": serial_sum,
        "speedup": speedup,
        "n_workers": n,
    }


# ──────────────────────────────────────────────────────────────────────────
# 3) 聚合（Aggregator）
# ──────────────────────────────────────────────────────────────────────────
def aggregate(task: str, subtasks: list[dict], dispatch_out: dict) -> str:
    """把各 worker 结果按子任务维度聚合成结构化报告。"""
    lines = [
        f"# 任务分配报告：{task}",
        f"派发子任务数：{dispatch_out['n_workers']}（上限 {MAX_SUBAGENTS}）",
        "",
    ]
    for st in subtasks:
        rid = st["id"]
        r = dispatch_out["results"].get(rid, {})
        lines.append(f"## {st['title']}")
        # 截短每个子结果，避免报告过长（沿用 Demo 的截短习惯）
        lines.append((r.get("result") or "")[:600])
        lines.append("")
    lines.append("---")
    lines.append(
        f"并行墙钟：{dispatch_out['wall_clock']}s ｜ 串行估算：{dispatch_out['serial_sum']}s ｜ "
        f"派发加速比：{dispatch_out['speedup']}×"
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 4) 入口
# ──────────────────────────────────────────────────────────────────────────
def run_orchestrator(task: str, max_subagents: int = MAX_SUBAGENTS,
                     worker_type: str = "sim", serial: bool = False) -> dict:
    """完整跑一遍：拆分 → 派发 → 聚合。返回结构化结果字典。"""
    subtasks = split_task(task, max_subagents=max_subagents)
    # 硬约束再次兜底（即便 LLM 返回更多，也截断到上限）
    subtasks = subtasks[:max_subagents]

    dispatch_out = dispatch_workers(
        subtasks, worker_type=worker_type, serial=serial, max_workers=max_subagents
    )
    report = aggregate(task, subtasks, dispatch_out)

    return {
        "task": task,
        "subtasks": subtasks,
        "final_answer": report,
        "workers": {
            rid: {"trace": r["trace"], "duration": r["duration"]}
            for rid, r in dispatch_out["results"].items()
        },
        "parallel_stats": {
            "wall_clock": dispatch_out["wall_clock"],
            "serial_sum": dispatch_out["serial_sum"],
            "dispatch_speedup": dispatch_out["speedup"],
            "n_workers": dispatch_out["n_workers"],
            "max_subagents": max_subagents,
        },
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    sample = "系统调研一下中国新能源汽车市场的整体表现与未来走向"
    out = run_orchestrator(sample, worker_type="sim")
    print(out["final_answer"])
