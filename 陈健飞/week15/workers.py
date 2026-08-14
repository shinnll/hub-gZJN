"""
worker 层：可插拔的叶子 worker（子 agent）

教学定位：
  - worker 是「叶子节点」：只执行，不派发（工具集里绝不注册 dispatch_subagents）
  - 作业要求「子 agent 不允许再下发任务」→ 从机制上杜绝，而非仅靠 prompt 软约束
  - 可插拔：worker_type 注册表，默认 "sim"（模拟），可扩展 "web"（联网调研）

sim（模拟）worker：零依赖、零外部 API，根据子任务返回结构化模拟调研结果。
  用于聚焦「分配机制」本身（拆分→并行→聚合），不卡在外部 key / 网络。
web（联网）worker：复用真实 Demo 的 ReActLoop + Tavily（需 DEEPSEEK_API_KEY + TAVILY_API_KEY）。
"""
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# sim（模拟）worker：纯函数，零依赖、零 LLM
# ──────────────────────────────────────────────────────────────────────────
def make_sim_worker() -> Callable:
    """返回一个符合 run_worker 约定的 sim worker：输入子任务 dict，返回 (result, trace, duration)。

    关键点：sim worker 内部没有任何「派发」语义，它只是本地确定性计算，
    因此天然满足「子 agent 不允许再下发任务」。
    """

    def _run(subtask: dict) -> dict:
        t0 = time.time()
        title = subtask.get("title", "")
        objective = subtask.get("objective", title)
        # 模拟「检索 + 归纳」耗时，让并行加速可被量化（A/B 对比）
        time.sleep(1.2)
        result = "\n".join([
            f"【模拟调研】{title}",
            f"目标：{objective}",
            "",
            "关键发现（模拟内容）：",
            f"  1. 关于「{title}」，行业普遍认为处于快速演进期，相关投入持续增加。",
            f"  2. 主要参与者在「{title}」上的技术路径趋于收敛，头部效应明显。",
            f"  3. 风险点集中在合规、成本与人才供给，需在「{title}」规划中前置考虑。",
            "",
            "（注：此为 sim 模式生成的模拟内容，用于演示分配/并行机制；"
            "设置 worker_type='web' 并配置 TAVILY_API_KEY 可切换为真实联网调研。）",
        ])
        duration = round(time.time() - t0, 2)
        # sim 无 LLM 循环，trace 记为单步，便于统一结构
        trace = [{
            "idx": 0, "agent": "sim-worker", "action": "simulate",
            "action_input": title, "observation": "（本地模拟，无工具调用、无派发）",
        }]
        return {"result": result, "trace": trace, "duration": duration}

    return _run


# ──────────────────────────────────────────────────────────────────────────
# web（联网）worker：复用真实 Demo 的 ReActLoop + Tavily（需 key）
# ──────────────────────────────────────────────────────────────────────────
def make_web_worker() -> Callable:
    """返回一个基于 ReActLoop + Tavily 的 web worker。需要 DEEPSEEK_API_KEY + TAVILY_API_KEY。

    注意：tools 只注册 web_search，绝不注册 dispatch_subagents → 机制层面无派发能力。
    """

    def _run(subtask: dict) -> dict:
        from react_loop import ReActLoop
        from tavily_search import tavily_search, format_search_result

        def web_search(q, **_):
            return format_search_result(tavily_search(q))

        tools = {"web_search": (web_search, "联网搜索，参数是查询词")}
        loop = ReActLoop(f"web-worker:{subtask.get('id', '?')}", tools=tools, max_steps=5)
        r = loop.run(subtask.get("objective") or subtask.get("title", ""))
        return {"result": r["final_answer"], "trace": r["trace"], "duration": r["duration"]}

    return _run


# ──────────────────────────────────────────────────────────────────────────
# 可插拔注册表
# ──────────────────────────────────────────────────────────────────────────
WORKER_REGISTRY: dict[str, Callable] = {
    "sim": make_sim_worker,
    "web": make_web_worker,
}


def get_worker(worker_type: str = "sim") -> Callable:
    """按类型取得 worker 工厂。未知类型回退 sim（保证可跑通）。"""
    factory = WORKER_REGISTRY.get(worker_type)
    if factory is None:
        logger.warning(f"未知 worker_type={worker_type!r}，回退 sim")
        factory = WORKER_REGISTRY["sim"]
    return factory()
