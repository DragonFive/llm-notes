# llm-notes

机器学习系统学习笔记的公开站：MkDocs-Material + GitHub Pages。

- `docs/` — 策展导出的 Markdown（source of truth 在私有 vault，本仓是只读镜像）
- `mkdocs.yml` — 站点配置
- `.github/workflows/ci.yml` — push main 后 GitHub Actions 自动构建发布到 Pages

本地预览：

```bash
pip install mkdocs-material
mkdocs serve
```
