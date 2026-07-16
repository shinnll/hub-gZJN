"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 单轮闭环三步：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填 → 模型生成最终回答
  3. 并行工具调用：模型一次输出多个 tool_call（如同时查年报+查天气），宿主逐个执行后一并回填
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德时代2023年营收和净利润？"

  # 内置示例问题（演示并行工具调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 单轮循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import (  # noqa: E402
    geocode_city,
    get_weather_from_dimension,
    CityDimension,
)


# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_city",
            "description": (
                "把城市名解析为经纬度等结构化维度(CityDimension)。"
                "只需确认城市定位、或要为后续天气查询获取坐标时调用。"
                "例如用户提到某个城市，先调本工具拿到经纬度。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_from_dimension",
            "description": (
                "根据城市维度(经纬度)查询该城市当前天气及未来3天预报。"
                "latitude/longitude/name 等应来自 geocode_city 的返回结果，"
                "不要凭空捏造坐标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，来自 geocode_city"},
                    "longitude": {"type": "number", "description": "经度，来自 geocode_city"},
                    "name": {"type": "string", "description": "城市名，来自 geocode_city"},
                    "country": {"type": "string", "description": "国家，来自 geocode_city"},
                    "admin1": {"type": "string", "description": "省/州，来自 geocode_city"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },

]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

def _weather_from_args(**kw):
    """把模型给的零散字段拼回 CityDimension，再查天气。"""
    dim = CityDimension(
        name=kw.get("name", ""),
        latitude=kw["latitude"],
        longitude=kw["longitude"],
        country=kw.get("country", ""),
        admin1=kw.get("admin1", ""),
        feature_code=kw.get("feature_code", ""),
        population=kw.get("population", 0),
    )
    return get_weather_from_dimension(dim)


def _geocode_to_str(city):
    """把 geocode 结果转成字符串给模型回填（tool 结果必须是字符串）。"""
    dim = geocode_city(city)
    return json.dumps(dim.__dict__, ensure_ascii=False) if not isinstance(dim, str) else dim



TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "geocode_city": _geocode_to_str,
    "get_weather_from_dimension": _weather_from_args,
}



# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时，先调用 geocode_city 获取城市经纬度，再调用 get_weather_from_dimension 查询天气"
    "（后者依赖前者的坐标，需分步调用，不要并行）。"

)


def run(client, model: str, question: str, verbose: bool = True, max_rounds: int = 5) -> dict:
    """
    多轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 若模型继续依赖上一步
    结果调用新工具（如 geocode_city → get_weather_from_dimension），则再跑一轮，
    直到模型不再调用工具（或达到 max_rounds）才生成最终回答。
    这样就能支持"顺序依赖"的工具链，而不是被单轮闭环卡在第一步。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # 多轮迭代：每轮让模型决定是否继续调工具；直到不再调工具才结束
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再调用工具 = 本轮即最终回答，结束循环
        if not msg.tool_calls:
            answer = msg.content or ""
            break

        # 【教学时刻 3】：模型输出了 tool_calls → 逐个执行后端函数，结果以 role=tool 回填
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 工具执行！！
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 进入下一轮：模型可能继续依赖上一步结果调用新工具（如 geocode → weather）

    else:
        # 达到 max_rounds 仍有未执行的工具调用，用最后一条消息兜底
        answer = msg.content or "（工具调用轮次已达上限，未生成最终回答）"

    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}



# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",  # 幻觉控制：比亚迪不在知识库
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
