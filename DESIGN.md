# Rummage 设计文档

> 版本：0.1.0（2026-10-24）
> 面向：后续维护者 / 接手 agent
> 用户文档见 [README.md](README.md)，本次改动的工作记录见本地 `WORKLOG.local.md`（不提交 git）。

## 1. 目标与约束

Rummage 是一个 Windows 本地文档混合检索 TUI：

- 支持 Markdown、DOCX、带文字层的 PDF、OFD；现在对无文字层 PDF 支持可选 OCR
- 完全离线，不依赖 Python、不联网、不修改原始文档
- 索引可持久化，第二次启动只处理变化
- 便携部署：exe + 模型 + DLL 可整体复制
- 目标环境可能存在无法安装的组件（例如没有 Windows OCR 语言包），所以 OCR 必须自带依赖或可降级

## 2. 总体架构

```text
                ┌──────────────────────────────┐
                │            TUI               │
                │  ratatui: 输入/结果/详情/状态 │
                └──────────────┬───────────────┘
                               │ IndexMsg / UiMsg
                ┌──────────────▼───────────────┐
                │         worker 线程          │
                │  扫描 → 文本提取 → 嵌入/OCR  │
                └───────┬──────────────┬───────┘
                        │              │
              ┌─────────▼──────┐  ┌────▼─────────────┐
              │  IndexEngine   │  │ watcher (notify) │
              │ BM25 + 向量 +  │  │ 文件事件 / 防抖  │
              │ RRF + rerank   │  └──────────────────┘
              └───────┬────────┘
                      │
        ┌─────────────┼────────────────┐
        │             │                │
  ┌─────▼─────┐ ┌─────▼─────┐  ┌──────▼──────┐
  │ SQLite    │ │ index.bin │  │ rerank.bin  │
  │ 文件/文本 │ │ BM25+向量 │  │ Gemma 向量  │
  └───────────┘ └───────────┘  └─────────────┘
                      │
              ┌───────▼────────┐
              │ ocr-cache/     │
              │ 按页 OCR 文本  │
              └────────────────┘
```

主线程只负责 TUI；索引、OCR、嵌入、重排都在 worker 线程中执行。TUI 退出时 `IndexHandle::shutdown()` 设置停止标志并等待 worker 落盘退出。

## 3. 模块

| 模块 | 职责 |
|---|---|
| `src/main.rs` | CLI、装配 worker/watcher/TUI、日志与 OCR 配置初始化 |
| `src/config.rs` | TOML 配置、路径解析、忽略/白名单、局部规则、尺寸校验 |
| `src/engine.rs` | 目录扫描、增量更新、文本索引、BM25+向量检索、RRF、rerank、快照协调 |
| `src/extract.rs` | docx/pdf/ofd/纯文本提取；PDF 逐页 `catch_unwind`；无文字层页 OCR 回退 |
| `src/ocr.rs` | PDFium 渲染、RapidOCR/PP-OCRv4 ONNX、页级缓存、OCR 状态 |
| `src/embed.rs` | bge-small-zh-v1.5 ONNX 嵌入 |
| `src/rerank.rs` | 可选 EmbeddingGemma 2 重排与持久化 |
| `src/store.rs` | SQLite `files`/`meta`，文件哈希、mtime、归一化文本、嵌入状态 |
| `src/persist.rs` | `index.bin` 快照的临时文件 + fsync + rename 原子读写 |
| `src/watcher.rs` | notify 文件事件 + 防抖 + 局部规则变化重扫 |
| `src/worker.rs` | 索引线程消息循环、停止信号、嵌入/重排批次 |
| `src/tui.rs` | ratatui 界面、按键、分页、状态栏、剪贴板、详情层 |
| `src/logging.rs` | 本地错误日志、UTC 时间戳、5 MiB 轮转 |
| `src/query.rs` / `tokenize.rs` / `normalize.rs` | 查询语法、jieba、OCR 归一化 |
| `vendor/pdf-extract` | 本地 vendored PDF 文本解析器（含 panic 加固） |
| `vendor/rapidocr-core` | 本地 vendored RapidOCR 管线（适配 ort rc.15） |
| `vendor/scour-search` | BM25 + HNSW + RRF 实现（含 serde/增量补丁） |

## 4. 索引流程

### 4.1 启动扫描

1. `walkdir` 遍历所有 `[watch] roots`，排除索引目录、内置忽略项、局部规则/扩展名不匹配的文件。
2. 对每个文件先取 `metadata`：`mtime_ns` 和 `size`。
3. 查 SQLite 中同一路径的旧记录：
   - `mtime_ns + size` 相同：**快速路径**，不读文件、不算哈希，只补齐缺失的嵌入/重排队列。
   - 不同：读文件，算 `blake3`。
     - 哈希相同：只更新 SQLite 的 `mtime/size`，不重新提取。
     - 哈希不同：重新提取、归一化、切块、更新索引。
4. 扫描结束后，把本次未出现的旧文件从 SQLite/内存索引删除。
5. 扫描完成写日志：`扫描完成：N 个文件，M 个变更，耗时 Xs`。

快速路径的代价：如果文件内容变了但 `mtime` 和 `size` 都没变，会被漏检。需要删除索引或 touch 文件后重启。网络盘、同步盘、杀软导致 mtime 反复变化时，快速路径收益会下降。

### 4.2 文本提取

- Markdown/纯文本：UTF-8 lossy 解码
- DOCX：ZIP + `word/document.xml`，跳过删除修订/页眉页脚
- OFD：ZIP + `OFD.xml` → `Document.xml` → `Content.xml`，按页序取 `TextCode`
- PDF：`vendor/pdf-extract` 逐页提取；单页解析错误或 panic 只跳过该页
  - 页级 `catch_unwind` 保证一份坏 PDF 不会拖垮整个索引
  - 未知 simple-font `/Encoding`、缺失 `Differences`、缺 `MediaBox`、畸形路径操作符等均降级处理
  - 仍有极少数 xref/trailer 损坏的文件无法加载，见“已知限制”

### 4.3 无文字层 PDF 与 OCR

`[index] textless_pdf` 是总开关：

- `false`：无文字层 PDF 直接跳过，不 OCR、不入索引
- `true`：允许 OCR 回退；若 OCR 未启用/模型缺失/失败，仍不建空索引

OCR 管线（`src/ocr.rs`）：

1. PDFium 从字节加载 PDF，按页渲染成 RGB 位图（目标宽度 = 页面宽度 × `dpi / 72`，上限 4096 px）
2. RapidOCR/PP-OCRv4 ONNX：检测 → 识别，按行拼成文本
3. 只处理文字字符数 `< [ocr] min_text_chars` 的页
4. 单个 PDF 最多处理 `[ocr] max_pages` 页；超出的页跳过并写日志
5. 结果缓存到 `<index.dir>/ocr-cache/<file-hash>-<模型大小>-<dpi>-ppocrv4.bin`
   - 每识别完一页立即落盘，退出后可续跑
   - 缓存按页存，重建索引时可复用
6. OCR 状态：`Disabled / Idle / Loading / Ready / Busy / Unavailable`，TUI 显示为 `OCR: 待加载/加载中/就绪/识别中/未就绪`

依赖：

- `pdfium.dll` 7.2 MB（PDFium 7881，BSD-3-Clause）
- `models/ocr/{det.onnx,rec.onnx,dict.txt}` 约 16.4 MB（PP-OCRv4，Apache-2.0）
- 复用现有 ONNX Runtime（`ort` 2.0.0-rc.15）

### 4.4 归一化、切块、持久化

- 归一化：控制字符、全角/半角、OCR 硬换行、页码/页眉页脚、Markdown 标记等
- 切块：约 1024 字节 / 128 字节重叠
- SQLite `files` 保存路径、hash、mtime、size、归一化全文、块数、是否已嵌入
- `index.bin` 保存 BM25 倒排 + 向量图；`rerank.bin` 保存可选 Gemma 向量
- 快照采用 `tmp + fsync + rename`，避免半写损坏

## 5. 检索流程

```text
查询
→ normalize + jieba 分词 + 中文停用词
→ 关键词 BM25 取 Top N
→ 语义 bge 取 Top N
→ RRF（k=60）融合
→ 按文件取最高分、去重
→ 截断到 [search] results
→ 可选 Gemma 对前 [rerank] top_n 名重排
→ TUI 结果列表
```

配置：

- `[search] candidates`：每路分块候选数，默认 50
- `[search] results`：最终文件数，默认 10
- `[rerank] top_n`：只重排基础排序前 N 个文件
- 结果列表按可见行数分页，状态栏显示 `页 X/Y`

## 6. 线程、取消与退出

- worker 线程拥有 `IndexEngine`；TUI 通过 mpsc 发送 Query/FileEvents/Shutdown
- `IndexHandle.stop` 是原子停止标志：
  - 扫描每文件前检查
  - 嵌入/重排每个小批前检查（`embed_documents_cancellable`）
  - 退出时停止调度，当前小批完成后落盘并结束
- OCR 在扫描线程内同步执行；模型懒加载，加载/识别期间状态为 `加载中/识别中`
- 退出时先 `watcher.stop()`，再 `index.shutdown()`，最后写日志 `退出`

## 7. TUI 与状态

- 输入框 / 结果列表 / 详情叠层
- `q` 在查询框是普通字符；`Esc` 关详情或退出；`Ctrl+C` 立即退出
- 状态栏：监控、索引文件数与阶段百分比、模型、重排、OCR、焦点、页
- 80 列窄屏且 OCR 启用时省略“焦点”字段，优先保留 OCR 和页码
- 结果页按当前可见行数动态计算；无结果不显示页信息

## 8. 日志

- 位置优先级：`<exe>/logs/rummage.log` → `%LOCALAPPDATA%/rummage/logs/rummage.log` → `%TEMP%/rummage-logs/rummage.log`
- 格式：`2026-10-24T15:00:00Z [LEVEL] target: message`
- 记录：启动参数、目录、模型开关、扫描耗时/变更数、错误、PDF 单页失败、OCR 状态与页进度
- 超过 5 MiB 启动时轮转 `rummage.log.1`
- 不记录查询词和结果正文

## 9. 配置摘要

```toml
[watch]
roots = ["D:\\docs"]
debounce_ms = 500

[index]
dir = "D:\\docs\\.rummage"
extensions = ["md", "markdown", "docx", "pdf", "ofd"]
textless_pdf = true

[search]
candidates = 50
results = 10

[ocr]
enabled = true
dir = "D:\\tools\\rummage\\models\\ocr"
pdfium = "D:\\tools\\rummage\\pdfium.dll"
dpi = 200
max_pages = 100
min_text_chars = 20

[model]
enabled = true
batch_size = 32

[rerank]
enabled = false
top_n = 5
weight = 0.5
```

目录局部 `.rummage.toml` 可覆盖 `[index] extensions` / `[include]` / `[ignore]`；内置忽略项和 `[normalize]` 仍全局生效。`[index] textless_pdf` 和 `[ocr]` 属于全局策略，不由局部规则覆盖。

## 10. 部署形态

```text
D:\tools\rummage-portable-0.1.0-full\
├── rummage.exe
├── pdfium.dll
├── models\
│   ├── bge-small-zh-v1.5\
│   ├── embeddinggemma-2\        # 可选
│   └── ocr\
│       ├── det.onnx
│       ├── rec.onnx
│       ├── dict.txt
│       ├── README.txt
│       └── PDFIUM_LICENSE.txt
├── docs\
├── index\                       # 可整体删除以重建
├── logs\
├── rummage.toml
└── start.bat
```

更新部署时只替换 `rummage.exe`、`README.md`、`pdfium.dll` 和 `models\ocr\`；不要覆盖 `rummage.toml`、`docs\`、`index\`、已有 bge/Gemma 模型。

## 11. 验证与性能基线

- 单元测试：130 个（`cargo test --offline`）
- 监控集成测试：2 个
- PDF 加固：594 页机考 PDF 由 5 页 panic 降为 0；109 份高风险样本逐页 0 panic（3 条文件级 xref/trailer 损坏仍需专门修复）
- OCR 选型：合成 11 页 + 真实 8 页；RapidOCR 准确率最高，Tesseract 次之，Windows OCR 快但表格顺序差且目标机可能无语言包
- 启动扫描：50 个 Markdown 首次 1.29s，二次 0.02s（0 变更）；真实 1077 文件索引二次扫描约 18.7s（目录遍历/stat 占主导，未读文件内容）
- OCR：3 页真实扫描件 release 约 15～23 秒（含首次加载）；224 页扫描件按 100 页上限约数分钟

## 12. 已知限制 / 后续方向

1. 3 个 xref/trailer 损坏的 PDF 仍无法加载；需要外部扫描 `N 0 obj` 重建 xref 的修复解析
2. `max_pages` 截断后，已索引文件不会自动补剩余页；需要重新建该文件/索引
3. OCR 在扫描线程内同步执行，启动首次 OCR 大扫描件可能很慢；后续可拆成独立 OCR 阶段/后台队列
4. OCR 没有百分比进度，只有 `识别中` 状态和每 10 页日志
5. 视觉向量（图表/图纸/照片）未实现；当前只做文本 OCR
6. 快速扫描依赖 mtime+size；内容变但保留时间戳/大小的文件会漏检
7. 目标内网机 CPU 可能较慢，OCR 耗时需在目标机重测

## 13. 关键设计取舍

- **文件级结果**：检索结果定位到文件，详情层展示最佳块；不引入段落级结果列表
- **页级 OCR 回退**：只 OCR 缺字页，避免整份重做；代价是版面/表格顺序仍由 OCR 决定
- **OCR 缓存放索引目录**：便于重建索引复用，但删除索引会丢缓存；需要时可把缓存目录外置
- **PDFium 而非系统 PDF 渲染**：避免 Server Core/精简系统缺组件；代价是额外 7.2 MB DLL
- **RapidOCR 而非 Tesseract/Windows OCR**：中文和表格顺序最好；代价是模型/集成复杂度
- **mtime+size 快速路径**：用极小漏检风险换启动速度；需要时可通过重建索引兜底
