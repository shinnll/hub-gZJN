"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

◆ 本周改造（多轮问答）：
  请求体新增两个可选字段：
    - session_id：传入相同 id 即复用同一会话历史（记忆层），实现多轮问答
    - reset：      true 时清空该 session_id 对应的历史，开启新会话
  不传 session_id 时退化为原版单轮行为（每次新建会话），保持向后兼容。

请求示例（多轮）：
  POST /query/manual  {"question": "茅台2023毛利率？", "session_id": "u123"}
  POST /query/manual  {"question": "那五粮液呢？",      "session_id": "u123"}  # 复用上文

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: Optional[str] = None   # ◆ 多轮：相同 id 复用历史
    reset:      bool = False           # ◆ 多轮：清空该会话历史


# ── 多轮会话存储（记忆层载体，按 mode + session_id 区分）──────────────────────
SESSIONS: dict = {}

def get_session(mode: str, session_id: Optional[str],
                reset: bool, max_steps: int):
    """
    返回本次请求使用的会话。
    - 未传 session_id：返回 None，由调用方按单轮（每次新建）处理
    - 传了 session_id：从 SESSIONS 取/建对应 ChatSession；reset=true 时重建
    """
    if not session_id:
        return None
    from react_manual import ChatSession as ManualSession
    from react_function_calling import ChatSession as FcSession

    key = (mode, session_id)
    if reset and key in SESSIONS:
        del SESSIONS[key]
    if key not in SESSIONS:
        sess_cls = ManualSession if mode == "manual" else FcSession
        SESSIONS[key] = sess_cls(max_steps=max_steps)
        logger.info("新建会话 %s", key)
    return SESSIONS[key]


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str,
                        session=None):
    """
    同步生成器在独立线程中逐步执行，每产出一步通过 asyncio.Queue 传递给
    异步 SSE 生成器，实现真正的边思考边推送。

    session 为 None -> 单轮（每次新建）；否则 -> 多轮（复用 session 的历史）。
    """
    if mode == "manual":
        from react_manual import run as react_run, ChatSession
    else:
        from react_function_calling import run as react_run, ChatSession

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            if session is None:
                # 单轮：run() 内部每次新建 messages
                gen = react_run(question, max_steps=max_steps)
            else:
                # 多轮：在 session 的持久 messages 上继续
                gen = session.ask(question, max_steps=max_steps)
            for step_data in gen:
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode,
                "multi_turn": session is not None})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    session = get_session("manual", req.session_id, req.reset, req.max_steps)
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    session = get_session("fc", req.session_id, req.reset, req.max_steps)
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
