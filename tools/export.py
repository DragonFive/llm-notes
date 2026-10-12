#!/usr/bin/env python3
"""策展导出：ob_note（私有 vault，只读）→ docs/（公开站内容）。

设计铁律：
- MD 是 source of truth，本脚本只做确定性转换，不写回源 vault（零改动）。
- 任何不确定的情况（无法解析的 obsidian 嵌入、敏感词命中）→ 显式失败，不静默兜底。

用法：
    OB_BUDDY_HOME=/path/to/ob_note python3 tools/export.py [--check]

    --check  只扫描不写文件（用于 CI / 提交前自检）

产物：
    docs/notes/*.md        转换后的笔记
    docs/assets/*          ../../assets/ 相对路径引用的图片（保持相对结构）
    docs/index.md          首页索引（生成）
    mkdocs.yml  nav 块     生成式重写（BEGIN/END GENERATED NAV 之间）
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# 敏感词：命中即失败（防内部信息外泄）。只收内部标识，不收通用技术词
# （例如"SM 内部结构"里的"内部"是正常技术表述，不能列入）。
SENSITIVE = [
    r"coding\.jd",
    r"xingyun",
    r"京东",
    r"工号",
    r"内网",
    r"公司内",
    r"内部仓库",
    r"内部环境",
    r"910C103",
    r"maxiaolong",
    r"DragonFive",
    r"xllm",
    r"xLLM",
    r"glm5_next",
    r"kimi_k3",
    r"qwen3_5",
    r"\bMR[ #]?\d{2,4}\b",
    r"![0-9]{5}",
]
SENSITIVE_RE = re.compile("|".join(SENSITIVE), re.IGNORECASE)

WIKILINK_ALIAS_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")   # [[A|B]] → B
WIKILINK_PLAIN_RE = re.compile(r"\[\[([^\]]*)\]\]")              # [[A]]  → A
EMBED_RE = re.compile(r"!\[\[([^\]]*)\]\]")                      # obsidian 嵌入 ![[x.svg|508]]
STD_SIZE_RE = re.compile(r"!\[([^\]]*?)\|(\d+)\]\(([^)]+)\)")     # ![alt|429](path) → ![alt](path)
ASSET_REF_RE = re.compile(r"\]\((\.\./assets/[^)]+)\)")           # ../assets/xxx.svg

IMAGE_EXT = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")

NAV_BEGIN = "  # BEGIN GENERATED NAV"
NAV_END = "  # END GENERATED NAV"


def fail(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def read_manifest(repo: Path) -> list:
    manifest = repo / "curation.txt"
    if not manifest.is_file():
        fail(f"缺少策展清单：{manifest}")
    entries = []
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    if not entries:
        fail("策展清单为空")
    return entries


def extract_title(text: str, fallback: str) -> str:
    """frontmatter 的 title: 优先，其次第一个一级标题，最后回退文件名。"""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            for line in text[3:end].splitlines():
                if line.strip().startswith("title:"):
                    value = line.split(":", 1)[1].strip()
                    # 只剥"成对包裹"的引号：标题内部/末尾的引号是合法字符
                    # （教训：`...不能迷信"大融合"` 曾被 strip('"') 咬掉收尾引号）
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                        value = value[1:-1]
                    return value
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def resolve_embed(target: str, src: Path, vault: Path) -> Path:
    """obsidian 嵌入路径是 vault 内相对路径（如 assets/x.svg → wiki/assets/x.svg）。"""
    for cand in (vault / "wiki" / target, vault / target, src.parent / target):
        if cand.is_file():
            return cand
    fail(f"{src.name}: 嵌入引用的资产不存在 ![[{target}]]（候选：wiki/{target}、{target}）")


def convert(text: str, name: str, src: Path, vault: Path) -> str:
    """obsidian 方言 → 通用 markdown（确定性、可回归）。"""

    def _embed(m: "re.Match") -> str:
        raw = m.group(1)
        target = raw.split("|", 1)[0].strip()
        if not target.lower().endswith(IMAGE_EXT):
            fail(f"{name}: 含非图片 obsidian 嵌入 ![[{raw}]]，导出脚本不支持——请改写后再收录")
        resolve_embed(target, src, vault)  # 存在性校验（不存在即失败）
        return f"![{Path(target).stem}](../assets/{Path(target).name})"

    text = EMBED_RE.sub(_embed, text)
    text = STD_SIZE_RE.sub(r"![\1](\3)", text)   # 去掉 obsidian 尺寸后缀
    text = WIKILINK_ALIAS_RE.sub(r"\2", text)
    text = WIKILINK_PLAIN_RE.sub(r"\1", text)
    return text


def scan(text: str, name: str) -> list:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        m = SENSITIVE_RE.search(line)
        if m:
            hits.append((i, m.group(0), line.strip()[:90]))
    return hits


def copy_assets(text: str, src_file: Path, vault: Path, repo: Path) -> int:
    """`](../assets/x.svg)` → docs/assets/x.svg（保持 notes/ 与 assets/ 的相对结构）。"""
    copied = 0
    for ref in ASSET_REF_RE.findall(text):
        src = (src_file.parent / ref).resolve()
        dst = repo / "docs" / ref.replace("../", "", 1)
        if not src.is_file():
            fail(f"{src_file.name}: 引用的资产不存在 {ref} → {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def write_index(repo: Path, items: list) -> None:
    lines = [
        "# LLM Notes",
        "",
        "机器学习系统学习笔记的公开策展镜像：GPU 调度、Tensor Core、编译器与指令级实现。",
        "",
        "- **源**：笔记在私有 Obsidian vault 中以 Markdown 维护，本仓是其**只读策展导出**"
        "（What you see is a curated export, not the working vault）。",
        "- **构建**：MkDocs-Material，GitHub Actions 推送后自动构建发布。",
        "- **原则**：Markdown 是 source of truth，HTML 只是派生的阅读层。",
        "",
        f"## 笔记索引（{len(items)} 篇）",
        "",
    ]
    for title, slug in items:
        lines.append(f"- [{title}](notes/{slug})")
    lines += [
        "",
        "> 持续更新中。本站只收录纯通用技术篇；涉及内部工程细节的笔记不在范围。",
        "",
        "<!-- 本文件由 tools/export.py 生成，请勿手改 -->",
    ]
    (repo / "docs" / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_nav(repo: Path, items: list) -> None:
    cfg_path = repo / "mkdocs.yml"
    cfg = cfg_path.read_text(encoding="utf-8")
    if NAV_BEGIN not in cfg or NAV_END not in cfg:
        fail(f"mkdocs.yml 缺少 {NAV_BEGIN.strip()} / {NAV_END.strip()} 标记")
    head, rest = cfg.split(NAV_BEGIN, 1)
    _, tail = rest.split(NAV_END, 1)
    block = [NAV_BEGIN, "  - 笔记:"]
    for title, slug in items:
        safe = title.replace("\\", "\\\\").replace('"', '\\"')   # YAML 双引号标量，防冒号/引号破坏
        block.append(f'      - "{safe}": notes/{slug}')
    block.append(NAV_END)
    cfg_path.write_text(head + "\n".join(block) + tail, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    vault = Path(os.environ.get("OB_BUDDY_HOME", "")).expanduser()
    if not vault.is_dir():
        fail("未设置 OB_BUDDY_HOME（或目录不存在）——先 source ~/dev/dsh-sync/config/env.sh")

    entries = read_manifest(repo)
    docs_notes = repo / "docs" / "notes"
    docs_notes.mkdir(parents=True, exist_ok=True)

    items, copied_assets, all_hits = [], 0, []
    for rel in entries:
        src = vault / rel
        if not src.is_file():
            fail(f"清单条目不存在：{rel}")
        raw = src.read_text(encoding="utf-8")
        hits = scan(raw, rel)
        if hits:
            all_hits.append((rel, hits))
            continue
        converted = convert(raw, rel, src, vault)
        slug = src.name
        title = extract_title(raw, src.stem)
        if not args.check:
            (docs_notes / slug).write_text(converted, encoding="utf-8")
            copied_assets += copy_assets(converted, src, vault, repo)
        items.append((title, slug))

    if all_hits:
        print("❌ 敏感词命中，导出中止：", file=sys.stderr)
        for rel, hits in all_hits:
            for lineno, word, ctx in hits:
                print(f"   {rel}:{lineno}  命中「{word}」  {ctx}", file=sys.stderr)
        sys.exit(1)

    if not args.check:
        items.sort(key=lambda x: x[1])
        write_index(repo, items)
        write_nav(repo, items)

    mode = "扫描" if args.check else "导出"
    print(f"✅ {mode}完成：{len(items)} 篇笔记，{copied_assets} 个资产更新，敏感词零命中")


if __name__ == "__main__":
    main()
