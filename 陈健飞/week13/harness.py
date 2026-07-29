import os
import re
import sys
import json
import ssl
import socket
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# ───────────────────────── 配置（harness 掌控路径，不依赖 skill md 里的路径）─────────────────────────
SKILLS_DIR = Path(r"E:/SHIN/Desktop/week13/skills")                       # 真实 skill 目录
OUTPUT_DIR = Path(r"E:/SHIN/Desktop/week13/output")                       # HTML 输出目录
PYTHON = sys.executable                                                   # 当前 python 解释器

# 粗略 token 估算（教学用，非精确）：混合中英文约 4 字符/token
def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─────────────────────────────────── 0. LLMClient─────────────────────────────────────
class LLMClient:
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "env_key": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-v4-flash",   # 按 ARCHITECTURE.md 文档命名
            "env_model": "DEEPSEEK_MODEL",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env_key": "DASHSCOPE_API_KEY",
            "default_model": "qwen-plus",
            "env_model": "QWEN_MODEL",
        },
    }

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()
        cfg = self.PROVIDERS.get(self.provider, self.PROVIDERS["deepseek"])
        self.base_url = os.environ.get("LLM_BASE_URL", cfg["base_url"]).rstrip("/")
        self.api_key = os.environ.get(cfg["env_key"], "")
        self.model = os.environ.get(cfg["env_model"], cfg["default_model"])
        self.env_key = cfg["env_key"]
        self.host = urlparse(self.base_url).hostname or ""
        self.configured = bool(self.api_key)
        self.reachable = self._probe() if self.api_key else False

    def _probe(self) -> bool:
        """启动期短超时探测 API host:443；连不上标记不可达，避免每次调用干等超时。"""
        try:
            if not self.host:
                return False
            with socket.create_connection((self.host, 443), timeout=2):
                return True
        except Exception:
            return False

    def chat(self, system_prompt: str, user_prompt: str, timeout: int = 10) -> str:
        if not self.configured:
            raise RuntimeError(f"未配置 {self.provider} 的 API Key（环境变量 {self.env_key}）")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},   # 尽量要求 JSON 输出
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ───────────────────────── 1. 头部解析（启动期，只取 frontmatter）─────────────────────────
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)

def parse_skill(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        # 没有 frontmatter 的退化处理：整篇当头部
        return {"name": md_path.parent.name, "description": "", "header": text, "body": "", "md_path": md_path}
    fm, body = m.group(1), m.group(2)

    name = ""
    nm = re.search(r"^name:\s*(.+)$", fm, re.M)
    if nm:
        name = nm.group(1).strip()

    # description 可能是 YAML 折叠块 `>-`
    dm = re.search(r"^description:\s*(?:>-\s*)?\n((?:[ \t]+.*\n?)*)", fm, re.M)
    if dm:
        desc = " ".join(l.strip() for l in dm.group(1).splitlines() if l.strip())
    else:
        dm2 = re.search(r"^description:\s*(.+)$", fm, re.M)
        desc = dm2.group(1).strip() if dm2 else ""

    return {
        "name": name or md_path.parent.name,
        "description": desc,
        "header": f"---\n{fm}\n---",   # 启动期加载的内容（常驻层）
        "body": body,                  # 按需加载的内容（触发层）
        "md_path": md_path,
    }


class SkillIndex:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.entries = {}
        self._build()

    def _build(self):
        for md in sorted(self.skills_dir.rglob("SKILL.md")):
            info = parse_skill(md)
            info["header_tokens"] = approx_tokens(info["header"])
            info["body_tokens"] = approx_tokens(info["body"])
            self.entries[info["name"]] = info

    def __iter__(self):
        return iter(self.entries.values())

    def __len__(self):
        return len(self.entries)

    def summary(self) -> str:
        lines = []
        for e in self:
            head = e["description"][:56]
            lines.append(f"  · [{e['name']}] {head}…  (头部 {e['header_tokens']} tok)")
        return "\n".join(lines)

    def catalog_for_llm(self) -> str:
        """给 LLM 看的 skill 目录（仅头部，对应渐进式披露的常驻层）。"""
        return "\n".join(
            f"- name: {e['name']}\n  description: {e['description']}" for e in self
        )


# ───────────────────────── 2. 按需读取函数（触发层 primitive）─────────────────────────
def read_skill(index: SkillIndex, name: str):
    e = index.entries.get(name)
    if not e:
        return None
    return e["body"]


# ───────────────────────── 3. 决策层（模型判断要不要加载）─────────────────────────
_STOP_EN = {"a", "an", "the", "of", "to", "and", "for", "my", "is", "are",
            "you", "i", "it", "this", "that", "with", "on", "in", "at", "by", "be", "as", "or"}
_STOP_CN = {"的", "我", "你", "请", "给", "一", "张", "个", "了", "为", "和", "就", "也", "都", "这", "那", "们"}

class DecisionEngine:
    def __init__(self, index: SkillIndex, llm: LLMClient):
        self.index = index
        self.llm = llm

    def decide(self, user_msg: str) -> tuple:
        if self.llm.configured and self.llm.reachable:   # LLM 优先，且启动期联网探测可达
            try:
                print("  ⏳ 正在调用模型做路由决策…")
                skills = self._llm_decide(user_msg)
                return skills, "llm"
            except Exception as e:
                print(f"  ⚠ LLM 决策失败（{type(e).__name__}: {str(e)[:90]}）→ 降级启发式兜底")
                return self._heuristic_decide(user_msg), "heuristic"
        return self._heuristic_decide(user_msg), "heuristic"

    def _llm_decide(self, user_msg: str) -> list:
        system = (
            "你是 Agent harness 的『skill 路由决策器』。系统采用渐进式披露："
            "启动时只加载每个 skill 的【头部摘要】，正文默认不进 context。"
            "你的任务：根据用户消息，判断是否需要加载某个 skill 的正文来执行任务。"
            "只依据头部信息做语义判断（用户未必使用触发词原话，要理解意图）。"
            "必须严格输出 JSON："
            "{\"trigger\": bool, \"skills\": [相关 skill 的 name 列表], \"reason\": \"简短理由\"}。"
            "若不需要任何 skill，trigger 为 false、skills 为空数组。"
        )
        user = f"用户消息：\n«{user_msg}»\n\n可选 skill（仅头部）：\n{self.index.catalog_for_llm()}"
        raw = self.llm.chat(system, user)
        data = _extract_json(raw)
        if not data.get("trigger"):
            return []
        names = data.get("skills", [])
        valid = [n for n in names if n in self.index.entries]   # 只保留真实存在的 skill
        return valid

    def _heuristic_decide(self, user_msg: str) -> list:
        """兜底：用户消息 token 与某 skill 头部 token 有重叠 → 判定相关。"""
        ut = self._tokens(user_msg)
        relevant = []
        for e in self.index:
            st = self._tokens(e["description"])
            if ut & st:
                relevant.append(e["name"])
        return relevant

    def _tokens(self, text: str) -> set:
        toks = re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]", text.lower())
        return {t for t in toks if t not in _STOP_EN and t not in _STOP_CN}


# ───────────────────────── 4. 执行器（flash-card 专用）─────────────────────────
class FlashCardExecutor:
    def __init__(self, skills_dir: Path, output_dir: Path):
        self.skills_dir = skills_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_word(self, user_msg: str):
        """从用户消息里找出目标英语单词，并定位 data/<word>.json。"""
        data_dir = self.skills_dir / "flash-card" / "data"
        if data_dir.exists():
            for jf in sorted(data_dir.glob("*.json")):
                w = jf.stem.lower()
                if re.search(rf"\b{re.escape(w)}\b", user_msg.lower()):
                    return w, jf
        words = [w for w in re.findall(r"[a-zA-Z]{3,}", user_msg.lower()) if w not in _STOP_EN]
        if words:
            w = max(words, key=len)
            jf = data_dir / f"{w}.json"
            if jf.exists():
                return w, jf
        return None, None

    def execute(self, user_msg: str):
        word, jf = self.extract_word(user_msg)
        if not jf:
            print("  ⚠ 找不到单词对应的数据文件，无法生成闪卡。")
            return None
        # 路径由 harness 自己拼，不照抄 SKILL.md 里的 .cursor/skills/...
        script = self.skills_dir / "flash-card" / "scripts" / "make_flashcard.py"
        out = self.output_dir / f"{word}.html"
        cmd = [PYTHON, str(script), str(jf), "-o", str(out)]
        print(f"  ▶ 运行: {' '.join(cmd)}")
        # 子进程（Anaconda python / Windows）管道输出默认 GBK，强制其用 UTF-8，
        # 读取端再用 errors="replace" 双保险，避免 UnicodeDecodeError 刷 traceback
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env)
        if res.returncode == 0:
            print(f"  ✅ 已生成: {out}")
            return out
        print(f"  ❌ 失败: {res.stderr}")
        return None


# ───────────────────────── 5. Harness 主循环 ─────────────────────────
class Harness:
    def __init__(self, skills_dir: Path, output_dir: Path):
        self.index = SkillIndex(skills_dir)          # 启动即构建头部索引（常驻层）
        self.llm = LLMClient()
        self.decider = DecisionEngine(self.index, self.llm)
        self.executor = FlashCardExecutor(skills_dir, output_dir)
        self._print_llm_status()

    def _print_llm_status(self):
        if self.llm.configured and self.llm.reachable:
            print(f"[LLM] 决策层 = 真 LLM（{self.llm.provider} / {self.llm.model}）")
        elif self.llm.configured and not self.llm.reachable:
            print(f"[LLM] 检测到 {self.llm.env_key}，但联网探测失败（{self.llm.host}:443 不可达）→ 决策层降级为启发式兜底")
        else:
            print(f"[LLM] 未检测到 {self.llm.env_key} → 决策层降级为启发式兜底")

    def handle(self, user_msg: str, verbose: bool = True):
        print("\n" + "=" * 64)
        print(f"用户消息: {user_msg}")

        # 启动期 context：只有头部索引（渐进式披露·常驻层）
        ctx_start = sum(e["header_tokens"] for e in self.index)
        print(f"[Context·启动] 仅加载 {len(self.index)} 个 skill 头部，共 {ctx_start} tok")
        if verbose:
            print("  索引:")
            print(self.index.summary())

        # —— 模型决策：只看头部判断要不要加载正文 ——
        relevant, source = self.decider.decide(user_msg)
        if not relevant:
            print(f"[决策·{source}] 模型认为不需要任何 skill，直接回答 / 结束。")
            return
        print(f"[决策·{source}] 模型判定相关 skill → {relevant} → 触发 read 加载正文")

        # —— 按需加载（渐进式披露核心）——
        loaded = ctx_start
        for name in relevant:
            body = read_skill(self.index, name)
            bt = self.index.entries[name]["body_tokens"]
            loaded += bt
            print(f"  ↳ read_skill('{name}') 加载正文 {bt} tok（已进入 context）")

        # 对比：朴素全量加载会怎样
        all_tokens = sum(e["header_tokens"] + e["body_tokens"] for e in self.index)
        saved = all_tokens - loaded
        print(f"[Context·加载后] 实际 {loaded} tok | 朴素全量需 {all_tokens} tok | 省 {saved} tok")
        print(f"  （渐进式披露价值：索引有 {len(self.index)} 个 skill，")
        print(f"   但只加载了命中的正文，未命中的正文始终不进 context）")

        # —— 执行 ——
        for name in relevant:
            if name == "flash-card":
                print(f"[执行] 按 '{name}' 正文逻辑运行:")
                self.executor.execute(user_msg)
            else:
                print(f"[执行] '{name}' 的执行器未实现（本 demo 仅 flash-card）")


# ───────────────────────── 入口：用户驱动 ─────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("HARNESS 启动：构建 skill 索引（仅头部 · 渐进式披露·常驻层）")
    h = Harness(SKILLS_DIR, OUTPUT_DIR)

    # 模式一：命令行参数 —— python harness.py "给我做 thrill 的闪卡"
    args = sys.argv[1:]
    if args:
        user_msg = " ".join(args)
        h.handle(user_msg, verbose=True)
        print("\n" + "=" * 64)
        print(f"处理完毕，HTML 输出目录: {OUTPUT_DIR}")
        return

    # 模式二：交互式 REPL —— 不带参数启动，逐行输入指令
    print("\n进入交互模式（输入 exit / quit / 退出 结束）")
    print("示例：给我做 thrill 的闪卡\n")
    first = True
    while True:
        try:
            user_msg = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit", "q", "退出"):
            print("再见。")
            break
        h.handle(user_msg, verbose=first)
        first = False


if __name__ == "__main__":
    main()
