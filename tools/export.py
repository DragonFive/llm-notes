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
import json
import os
import re
import shutil
import sys
from pathlib import Path

# 敏感词：命中即失败（防内部信息外泄）。只收内部标识，不收通用技术词
# （例如"SM 内部结构"里的"内部"是正常技术表述，不能列入）。
# 口径（2026-10-12 用户裁定）：xllm 与所服务的模型均开源——
# 模型代号（glm5_next/qwen3_5/kimi_k3 等）与源码文件名/file:line **不算敏感**，不列入。
# 注意：人名/同事名等更私密的模式不进本表（本表在公开仓），
# 放 vault 私有侧的 .publish-redactions/<笔记名>.json 规则里处理。
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
    r"\bMR[ #]?\d{2,4}\b",
    r"![0-9]{5}",
    # 2026-10-12 脱敏升级新增：内部基础设施与内部项目代号（不含开源模型/框架名）
    r"6\.205\.\d{1,3}\.\d{1,3}",   # 内部网段 A
    r"11\.87\.\d{1,3}\.\d{1,3}",   # 内部网段 B
    r"/export/home",               # 内部路径
    r"\bgpu9\d\b",                 # 内部机器名（gpu90-99）
    r"onerec",
    r"OneRec",
]
SENSITIVE_RE = re.compile("|".join(SENSITIVE), re.IGNORECASE)

# 脱敏规则（vault 私有侧，不进公开仓）：
#   $OB_BUDDY_HOME/.publish-redactions/<笔记basename>.json
#   格式：[{"find": "<regex>", "replace": "<替换文本>"}, ...]  按序应用。
#   应用时机：转换之前、敏感词扫描之前（扫的是脱敏后的文本）。
#   规则文件本身含敏感词 → 只能住在私有 vault，随 obnote-sync 同步。
REDACTION_DIRNAME = ".publish-redactions"

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


def load_redactions(vault: Path, basename: str) -> list:
    """读 vault 私有侧的脱敏规则；无规则文件返回空表。

    规则住在私有 vault（$OB_BUDDY_HOME/.publish-redactions/），永不进公开仓——
    规则文本本身含敏感词，放公开仓等于自我泄露。
    每条规则：{"find": ..., "replace": ..., "literal": true?, "flags": "s"?}
      literal=true → find 按字面匹配（内部 re.escape，写纯文本最稳）
      flags 含 "s" → DOTALL（跨行段落整段替换用）
    """
    rf = vault / REDACTION_DIRNAME / (Path(basename).stem + ".json")
    if not rf.is_file():
        return []
    try:
        rules = json.loads(rf.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"脱敏规则文件解析失败 {rf.name}: {e}")
    if not isinstance(rules, list):
        fail(f"脱敏规则必须是 JSON 数组：{rf.name}")
    compiled = []
    for i, r in enumerate(rules, 1):
        if not isinstance(r, dict) or "find" not in r or "replace" not in r:
            fail(f"脱敏规则第 {i} 条缺 find/replace 字段：{rf.name}")
        flags = re.DOTALL if "s" in str(r.get("flags", "")) else 0
        pattern = re.escape(r["find"]) if r.get("literal") else r["find"]
        try:
            compiled.append((re.compile(pattern, flags), str(r["replace"])))
        except re.error as e:
            fail(f"脱敏规则第 {i} 条正则非法：{rf.name} → {e}")
    return compiled


def apply_redactions(text: str, rules: list) -> str:
    """按序应用脱敏规则（确定性：同一规则同一文本永远同一输出）。"""
    for cre, rep in rules:
        text = cre.sub(rep, text)
    return text


def read_manifest(repo: Path) -> list:
    manifest = repo / "curation.txt"
    if not manifest.is_file():
        fail(f"缺少策展清单：{manifest}")
    entries = []
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 支持改名发布："vault相对路径 -> 新文件名.md"
        # （源笔记名含敏感词时，公开 slug 必须换名——文件名也是泄露面）
        if " -> " in line:
            rel, slug = line.split(" -> ", 1)
            entries.append((rel.strip(), slug.strip()))
        else:
            entries.append((line, None))
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


# 分类体系（2026-10-16 用户需求：首页分类呈现 + 最新在前）。
# 从源笔记 frontmatter 的 tags 驱动，按 CATEGORIES 顺序首匹配归类，无匹配落「其他」。
# 扩展：加新类目就加一行 (显示名, [匹配 tags])；想改优先级就调顺序。
CATEGORIES = [
    ("推理系统与分布式并行", [
        "topic/contextparallel", "topic/kvsplit", "topic/kvcache",
        "topic/pddisagg", "topic/prefix-cache", "topic/linear-attention",
        "topic/mtp", "topic/attention", "domain/systems",
    ]),
    ("GPU Kernel 与指令级", [
        "topic/gpukernel", "topic/cuda", "topic/gemm",
    ]),
    ("编译器与图优化", [
        "topic/compiler", "topic/optimization",
    ]),
]
FALLBACK_CATEGORY = "其他"


def extract_tags(text: str) -> list:
    """frontmatter tags → ['topic/xxx', ...]；兼容块格式与行内数组格式。"""
    tags = []
    if not text.startswith("---"):
        return tags
    end = text.find("\n---", 3)
    if end < 0:
        return tags
    in_tags = False
    for line in text[3:end].splitlines():
        s = line.strip()
        if s.startswith("tags:"):
            in_tags = True
            inline = s[5:].strip()
            if inline.startswith("["):   # 行内数组：tags: [a, b, c]
                tags += [t.strip().strip("'\"") for t in inline.strip("[]").split(",") if t.strip()]
                in_tags = False
            continue
        if in_tags:
            if s.startswith("- "):
                tags.append(s[2:].strip().strip("'\""))
            elif s:
                in_tags = False
    return tags


def categorize(tags: list) -> str:
    # 归一化连字符（topic/gpu-kernel 与 topic/gpukernel 视为同一标签）
    norm = {t.replace("-", "") for t in tags}
    for cat_name, cat_tags in CATEGORIES:
        if any(ct.replace("-", "") in norm for ct in cat_tags):
            return cat_name
    return FALLBACK_CATEGORY


def write_index(repo: Path, items: list) -> None:
    # items: [(title, slug, category)]；类内按 slug 倒序 = 最新在前（slug 前缀是日期）
    by_cat = {}
    for title, slug, cat in items:
        by_cat.setdefault(cat, []).append((title, slug))
    lines = [
        "# LLM Notes",
        "",
        "机器学习系统学习笔记的公开策展镜像：GPU 调度、Tensor Core、编译器与指令级实现。",
        "",
        "- **源**：笔记在私有 Obsidian vault 中以 Markdown 维护，本仓是其**只读策展导出**"
        "（What you see is a curated export, not the working vault）。",
        "- **构建**：MkDocs-Material，GitHub Actions 推送后自动构建发布。",
        "- **原则**：Markdown 是 source of truth，HTML 只是派生的阅读层。",
        "- **方法**：笔记以源码为证据底座（file:line 锚点 + 可复算的手算数字）；新作经我主导的教学流水线产出——勘探定界、分章深潜、冷读验收、勘误裁决。",
        "",
        f"## 笔记索引（{len(items)} 篇，类内最新在前）",
        "",
    ]
    cat_order = [c for c, _ in CATEGORIES] + ([FALLBACK_CATEGORY] if FALLBACK_CATEGORY in by_cat else [])
    for cat in cat_order:
        if cat not in by_cat:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        for title, slug in sorted(by_cat[cat], key=lambda x: x[1], reverse=True):
            date = slug[:10] if slug[:4].isdigit() else ""
            suffix = f" `{date}`" if date else ""
            lines.append(f"- [{title}](notes/{slug}){suffix}")
        lines.append("")
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
    by_cat = {}
    for title, slug, cat in items:
        by_cat.setdefault(cat, []).append((title, slug))
    block = [NAV_BEGIN]
    cat_order = [c for c, _ in CATEGORIES] + ([FALLBACK_CATEGORY] if FALLBACK_CATEGORY in by_cat else [])
    for cat in cat_order:
        if cat not in by_cat:
            continue
        block.append(f"  - {cat}:")
        for title, slug in sorted(by_cat[cat], key=lambda x: x[1], reverse=True):
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

    items, copied_assets, all_hits, redacted_notes = [], 0, [], []
    for rel, slug_override in entries:
        src = vault / rel
        if not src.is_file():
            fail(f"清单条目不存在：{rel}")
        slug = slug_override or src.name
        if SENSITIVE_RE.search(slug):
            fail(f"公开 slug 含敏感词：{slug}（用「路径 -> 新名.md」改名发布）")
        raw = src.read_text(encoding="utf-8")
        rules = load_redactions(vault, src.name)
        if rules:
            raw = apply_redactions(raw, rules)
            redacted_notes.append(rel)
        hits = scan(raw, rel)
        if hits:
            all_hits.append((rel, hits))
            continue
        converted = convert(raw, rel, src, vault)
        title = extract_title(raw, src.stem)
        if not args.check:
            (docs_notes / slug).write_text(converted, encoding="utf-8")
            copied_assets += copy_assets(converted, src, vault, repo)
        items.append((title, slug, categorize(extract_tags(raw))))

    if all_hits:
        print("❌ 敏感词命中，导出中止（可在私有 vault 的 .publish-redactions/<笔记名>.json 写脱敏规则后重试）：", file=sys.stderr)
        for rel, hits in all_hits:
            for lineno, word, ctx in hits:
                print(f"   {rel}:{lineno}  命中「{word}」  {ctx}", file=sys.stderr)
        sys.exit(1)

    if not args.check:
        # 排序在各 write_* 内按分类做（类内 slug 倒序 = 最新在前）
        write_index(repo, items)
        write_nav(repo, items)

    mode = "扫描" if args.check else "导出"
    extra = f"，{len(redacted_notes)} 篇应用了脱敏规则（{', '.join(Path(r).name for r in redacted_notes)}）" if redacted_notes else ""
    print(f"✅ {mode}完成：{len(items)} 篇笔记，{copied_assets} 个资产更新，敏感词零命中{extra}")


if __name__ == "__main__":
    main()
