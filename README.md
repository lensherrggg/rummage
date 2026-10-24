# Rummage

本地文档混合检索 TUI：关键词（BM25）+ 语义向量（bge-small-zh-v1.5）+ RRF 融合，定位到文件，完全离线运行。

面向 PDF OCR 转写产生的、带页码/断行/页眉页脚噪声的中文文档库，同时支持直接索引 `.docx`、带文字层的 `.pdf` 和 `.ofd`。不需要 Python，不需要联网，不会修改任何原始文件。

- 查询用自然语言，也可以加 `dir:` / `name:` / `"短语"` 过滤
- 可按文件名或相对路径通配符限定收录范围，也可按目录/文件名通配符排除
- 可在任意子目录放 `.rummage.toml`，让该子树单独使用一套收录/排除规则
- 启动时全量扫描，之后监听新增/修改/删除并增量更新
- 索引持久化，第二次启动只处理变化的文件
- 模型缺失或监控失败时自动降级，界面明确提示，不会闪退

## 快速开始

```powershell
# 1. 按 §1.1 把 exe 与 bge 模型放好
#    D:\tools\rummage\rummage.exe
#    D:\tools\rummage\models\bge-small-zh-v1.5\...
# 2. 启动并指定要检索的目录（可重复 --dir）
D:\tools\rummage\rummage.exe --dir D:\docs

# 3. 可选：临时排除某些目录/文件（可重复 --ignore）
D:\tools\rummage\rummage.exe --dir D:\docs --ignore drafts --ignore "~$*"
```

第一次启动会建立索引，状态栏出现 `模型: 就绪` 后即可检索。

| 操作 | 按键 |
|---|---|
| 输入查询并检索 | 输入文字后按 `Enter` |
| 在查询框与结果列表间切换 | `Tab` |
| 选择结果 | `↑` / `↓` |
| 复制选中文件路径 | 结果区按 `c` |
| 退出 | `Esc` 或 `Ctrl+C` |

查询可混用过滤语法：`断行修复 dir:docs\ocr`、`会议纪要 name:2025`、`"合并硬换行"`。

可选二次重排默认关闭；需要时把 EmbeddingGemma 2 放到 `models\embeddinggemma-2\`，在配置中写 `[rerank] enabled = true`，详见 §1.1 与 §3。完整说明见 §二、§三，部署到离线机器见 §六。

---

## 一、首次配置（从零开始）

### 1.1 准备程序目录

选一个固定位置，例如 `D:\tools\rummage`，按下面的结构放好 exe 和模型：

```text
D:\tools\rummage\
├── rummage.exe
└── models\
    └── bge-small-zh-v1.5\
        ├── model.onnx
        ├── tokenizer.json
        ├── config.json
        ├── special_tokens_map.json
        └── tokenizer_config.json
```

模型文件来自 `Xenova/bge-small-zh-v1.5` 的 ONNX 导出，五个文件缺一不可。默认情况下程序**只在 exe 同目录**查找 `models\bge-small-zh-v1.5`，不看当前工作目录。

把 exe 和模型拷过去（以从源码构建为例）：

```powershell
New-Item -ItemType Directory -Force D:\tools\rummage\models\bge-small-zh-v1.5 | Out-Null
Copy-Item .\target\release\rummage.exe D:\tools\rummage\
Copy-Item .\models\bge-small-zh-v1.5\* D:\tools\rummage\models\bge-small-zh-v1.5\ -Recurse
```

```bat
mkdir D:\tools\rummage\models\bge-small-zh-v1.5
copy target\release\rummage.exe D:\tools\rummage\
xcopy models\bge-small-zh-v1.5 D:\tools\rummage\models\bge-small-zh-v1.5 /E /I
```

> 直接运行 `target\release\rummage.exe` 也可以，但此时默认模型目录是 `target\release\models\bge-small-zh-v1.5`。建议把 exe 复制出来单独使用。

如果模型必须放在别处，用下面两种方式之一指定，**不必**复制到 exe 旁边：

```powershell
# 方式 A：环境变量（当前窗口有效）
$env:RUMMAGE_MODEL_DIR = "E:\models\bge-small-zh-v1.5"
D:\tools\rummage\rummage.exe --dir D:\docs
```

```toml
# 方式 B：写进 rummage.toml（推荐，长期有效）
[model]
dir = "E:\\models\\bge-small-zh-v1.5"
```

查找顺序：`[model] dir` → 环境变量 `RUMMAGE_MODEL_DIR` → `exe目录\models\bge-small-zh-v1.5`。

如果还要启用可选的二次重排（默认关闭），再把 EmbeddingGemma 2 的导出放到 `models\embeddinggemma-2\`：

```text
D:\tools\rummage\
└── models\
    └── embeddinggemma-2\
        ├── onnx\model_quantized.onnx
        ├── onnx\model_quantized.onnx_data   # 量化模型的外部权重，若有
        └── tokenizer.json
```

重排模型的查找顺序是 `[rerank] dir` → 环境变量 `RUMMAGE_RERANK_MODEL_DIR` → `exe目录\models\embeddinggemma-2`。不启用重排时，这个目录不存在也没关系。

要求：Windows 11 x64，无需安装 Python、Rust 或 GPU。ONNX Runtime 已静态链接进 exe；Windows 11 自带的 `DirectML.dll` 会被导入，系统目录中已存在，无需另行分发。

### 1.2 准备监控目录

待检索的文档放在任意本地目录下，例如 `D:\docs`。默认收录 `.md`、`.markdown`、`.docx`、带文字层的 `.pdf` 和 `.ofd`；其他纯文本格式（如 `.txt`）可自行加进 `[index] extensions`。程序只读取文件，不会写入或改写。

不需要事先整理文档。OCR 产生的页码行、页眉页脚、硬换行等噪声在**索引阶段**归一化处理，原文保持原样。

### 1.3 第一次启动

最小启动方式，直接命令行给目录：

```powershell
D:\tools\rummage\rummage.exe --dir D:\docs
```

`--dir` 可以重复，监控多个根目录：

```powershell
D:\tools\rummage\rummage.exe --dir D:\docs --dir D:\notes
```

启动时也可以临时排除目录或文件，模式按名称匹配、可重复，效果等同于同时写进 `[ignore] dirs` 和 `[ignore] names`：

```powershell
D:\tools\rummage\rummage.exe --dir D:\docs --ignore drafts --ignore "20??_归档" --ignore "~$*"
```

`--ignore` 是本次启动的补充规则；目录里的 `.rummage.toml` 仍会按原有语义整体替换全局 `[ignore]`。

第一次启动会依次发生：

1. 加载 ONNX 模型。状态栏先显示 `模型: 加载中`，成功后变为 `模型: 就绪`。
2. 在**第一个根目录**下创建索引目录 `D:\docs\.rummage`（默认位置）。不想在被索引目录里产生文件时，用 `[index] dir` 指到别处，见[四、索引目录与维护](#四索引目录与维护)。
3. 递归扫描全部根目录，状态栏文件数递增并显示 `· 扫描 NN%`。
4. 对每个文件做归一化、切块、建关键词索引，再按每批 `[model] batch_size` 个块（默认 32）计算语义向量。首次索引耗时主要由这一步决定，文档越多越久；状态栏显示 `· 语义 NN%`。
5. 如果启用了 `[rerank]`，再为每个文件计算一组重排向量并写入 `rerank.bin`；这一步比主模型更慢，首次启用会明显拉长索引时间；状态栏显示 `· 重排 NN%`。
6. 索引落盘。扫描期间的查询会排队，扫描结束后立即处理。

中途可以按 `Esc` 或 `Ctrl+C` 退出。退出时会先停止调度，不再开始新的文件或嵌入批次；正在计算的一小批完成后立即落盘并结束进程，已完成的索引会保存，下次启动从未完成的部分继续。把 `[model] batch_size` / `[rerank] batch_size` 调小可以让退出更快，但索引吞吐会下降。

**启动后先确认状态栏四项**（终端至少 80×20）：

```text
查询: 
────────────────────────────────────────────────────────────
输入查询后按 Enter 检索
────────────────────────────────────────────────────────────
监控: 运行中 | 索引: 0 文件 | 模型: 就绪 | 焦点: 查询
```

- `监控: 运行中` —— 文件监听正常
- `索引: N 文件` —— 已收录的文档数量；索引进行中会追加当前阶段与百分比，如 `· 扫描 67%`。`扫描` 期间查询要排队；`语义` / `重排` 期间可以正常检索
- `模型: 就绪` —— 语义检索可用；若是 `未就绪(仅关键词)` 见[常见问题](#五常见问题)
- `焦点: 查询` —— 键盘输入进入查询框

输入一句话，按 `Enter` 检索；`Tab` 把焦点切到结果列表，`↓` 选择，`c` 复制路径。

### 1.4 写配置文件（可选，但推荐）

日常使用不想每次敲 `--dir`，就把配置写成文件。在任意位置新建 `rummage.toml`，例如 `D:\tools\rummage\rummage.toml`：

```toml
[watch]
roots = ["D:\\docs", "D:\\notes"]
debounce_ms = 500

[index]
# 索引数据存放目录；省略则用第一个根目录下的 .rummage
dir = "D:\\tools\\rummage\\index"
# 参与索引的扩展名；省略时默认 md / markdown / docx / pdf / ofd
extensions = ["md", "markdown", "docx", "pdf", "ofd"]

[search]
# 可选：每路分块候选数与最终结果条数；省略时默认 50 / 10
candidates = 50
results = 10

[include]
# 可选白名单：只收录这些文件；留空或删除本节即全部收录
# patterns = ["会议纪要*", "2025/**"]

[ignore]
# 排除其中的子目录，按目录名匹配，支持通配符
dirs = ["drafts*", "20??_归档", "临时[0-9]"]

[model]
enabled = true
dir = "D:\\tools\\rummage\\models\\bge-small-zh-v1.5"

[rerank]
# 可选：EmbeddingGemma 2 二次重排，默认关闭
enabled = false
# dir = "D:\\tools\\rummage\\models\\embeddinggemma-2"
```

启用重排时，把 `enabled` 改为 `true` 并确认模型目录；状态栏会多一项 `重排: 就绪`。

之后启动只需：

```powershell
D:\tools\rummage\rummage.exe --config D:\tools\rummage\rummage.toml
```

指定了 `--config` 就只读这个文件；不指定则读**当前工作目录**下的 `rummage.toml`。因此也可以先 `cd D:\tools\rummage` 再运行 `rummage.exe`。做成快捷方式时，把「起始位置」设为 `D:\tools\rummage` 即可。

命令行 `--dir` 优先于 `[watch] roots`；两者都没有时程序报错退出：

```text
未指定监控目录：请使用 --dir，或在 rummage.toml 的 [watch] roots 中配置
```

### 1.5 之后的每次启动

直接用同样的命令启动即可：

- 未变化的文件按哈希跳过，不再解析、不再嵌入，启动只需数秒
- 索引上次已落盘，模型与维度没变就原样复用
- 上次退出前没嵌入完的文件会自动续跑；重排向量同样续跑（`重排: 加载中` → `就绪`）
- 模型文件被移走或损坏时，自动降级为纯关键词检索，状态栏显示 `模型: 未就绪(仅关键词)`
- 重排模型被移走或损坏时不影响基础检索，状态栏显示 `重排: 未就绪`，红字给出原因
- 关闭 `[rerank]` 后检索完全回到原来的排序，已有的 `rerank.bin` 保留，重新打开无需重算

---

## 二、日常使用

### 2.1 界面

```text
查询: 如何合并硬换行
────────────────────────────────────────────────────────────
   D:\docs\OCR预处理说明.md
      页眉页脚与页码行会先被清理，再处理硬换行…
>  D:\docs\断行修复笔记.md
      合并硬换行的规则：段内单换行视为软换行，直接拼接…
   D:\notes\页眉页脚清理.md
      清理页眉页脚后再合并断行，避免把标题混进正文…
────────────────────────────────────────────────────────────
监控: 运行中 | 索引: 128 文件 | 模型: 就绪 | 焦点: 结果
```

- 第一行：查询输入框，焦点在此时显示光标
- 中间：结果列表，默认最多 10 条（由 `[search] results` 控制）；每条第一行是完整文件路径，第二行是**最佳匹配块**里的摘录，命中的词以黄色高亮，`>` 为当前选中项
- 最后一行：状态栏；右侧偶尔出现红字提示（最近一次错误，4 秒后消失）
- 焦点切到结果列表时，结果区底部出现提示行 `c 复制路径   j/k 选择   Enter 详情   Tab 返回查询`
- 在结果列表按 `Enter` 会叠一层详情：显示完整路径、关键词/语义名次、融合分、重排分与名次变化，以及最佳匹配块的全文（命中词同样黄色高亮）；详情层内 `j`/`k` 滚动、`Esc` 关闭
- 检索结果为空或尚未检索时：`输入查询后按 Enter 检索`

详情层大致如下（`Esc` 关闭后回到结果列表；没有详情层时 `Esc` 才是退出）：

```text
详情 · 断行修复笔记.md
路径: D:\docs\断行修复笔记.md
来源: 关键词第 1 位 + 语义第 2 位
融合分: 0.0320（RRF，k=60）
重排: 0.8123（权重 0.35，第 2 → 第 1 位）

最佳匹配片段
合并硬换行的规则：段内单换行视为软换行，直接拼接；段落边界才保留…

Esc 关闭   j/k 滚动
```

摘录取自归一化后的文本（页码、页眉页脚、硬换行噪声已清理），中间用空格拼接为一行；命中位置靠后时首尾会出现 `…`。纯语义命中若没有字面命中词，则展示最佳向量块的开头。

终端小于 80×20 时进入降级布局：隐藏状态栏、最多显示 5 条结果（且不超过 `[search] results`）、只显示路径不显示摘录，底部显示红色提示 `终端过小（最小 80×20）`。放大终端后自动恢复。

### 2.2 按键

| 按键 | 行为 |
|---|---|
| 字符 | 焦点在查询框时输入查询 |
| `Backspace` / `Delete` | 删除光标前 / 后一个字符 |
| `←` / `→` | 移动光标 |
| `Home` / `End` | 跳到行首 / 行尾 |
| `Enter` | 焦点在查询框时执行检索；焦点在结果列表时打开选中项的详情层 |
| `Tab` | 在「查询」和「结果」之间切换焦点 |
| `↑` / `↓` | 在结果列表中移动选择（任意焦点）；详情层内滚动 |
| `j` / `k` | 焦点在结果列表时，向下/向上移动选择；详情层内滚动 |
| `c` | 焦点在结果列表时，复制当前选中路径到剪贴板 |
| `q` | 焦点在查询框时输入字母 `q`；焦点在结果列表时无动作 |
| `Esc` | 关闭详情层；没有详情层时退出（**全局**） |
| `Ctrl+C` | 立即退出 |

> 查询框里可以正常输入英文 `q`。剪贴板不可用时只在状态栏提示，不影响其他功能。

### 2.3 查询语法

| 写法 | 含义 | 示例 |
|---|---|---|
| 普通文本 | 自然语言混合检索（关键词 + 语义） | `如何合并硬换行` |
| `dir:子串` | 只保留路径包含该子串的文件 | `OCR预处理 dir:docs\ocr` |
| `name:子串` | 只保留文件名包含该子串的文件 | `断行修复 name:preprocess` |
| `"短语"` | 归一化文本必须包含该短语 | `"页眉页脚"` |

规则细节：

- `dir:` / `name:` 的值可加引号以包含空格：`dir:"docs/ocr 资料"`；同一类过滤器写多次时最后一个生效
- 过滤在结果聚合到文件后执行，路径与文件名均不区分大小写
- `"短语"` 是精确包含匹配，不参与分词，也不区分大小写；可写多个，例如 `"页眉" 与 "页脚"`
- 引号未闭合时，后半段当作普通查询文本
- 只有过滤器、没有正文也可以检索，例如 `dir:docs` 会按过滤器列出已索引文件（顺序不固定，仍受 `[search] results` 上限约束）

检索流程：查询归一化 → jieba 分词并过滤中文虚词 → BM25 取每路候选 Top N + 语义取 Top N（`[search] candidates`，默认 50）→ RRF（k=60）融合 → 按文件取最高分去重 → 最多 M 条（`[search] results`，默认 10）。

`candidates` 是**聚合前的分块候选数**，不是最终结果条数；调大能让更多文件进入融合与重排，但查询会略慢。`results` 才是结果列表显示的文件数上限。

结果详情层（结果区按 `Enter`）把这条链路摊开：`来源` 是该文件在关键词、语义两路候选里各自的第一出现位置；`融合分` 在两路都可用时是 RRF 分，只有一路时显示该路原始分（BM25 或语义相似度）；`重排` 给出 Gemma 分数、配置权重和重排前后名次，只对前 `top_n` 名展示。原始分值受模式影响、彼此不可比，所以默认不放进结果列表，只在详情层用于排查排序。

### 2.4 状态栏

| 字段 | 取值 | 说明 |
|---|---|---|
| 监控 | `运行中` / `未运行` | `未运行` 表示文件监听启动失败；检索仍可用，只是不再自动感知增删改，右侧红字给出原因 |
| 索引 | `N 文件` + 阶段百分比 | 已收录文件数；索引进行中追加 `· 扫描 NN%` / `· 语义 NN%` / `· 重排 NN%`，空闲时不显示百分比。扫描阶段查询排队，语义/重排阶段可正常检索 |
| 模型 | `加载中` / `就绪` / `未就绪(仅关键词)` / `已禁用` | 只有 `就绪` 时启用语义检索；其余情况退化为纯 BM25 |
| 重排 | `加载中` / `就绪` / `未就绪` | 仅在 `[rerank] enabled = true` 时显示；`未就绪` 表示重排被跳过，基础检索不受影响 |
| 焦点 | `查询` / `结果` | 当前键盘焦点，决定字符输入还是 `c` 复制 |

启用重排且正在补索引时状态栏形如：

```text
监控: 运行中 | 索引: 1434 文件 · 语义 32% | 模型: 就绪 | 重排: 就绪 | 焦点: 查询
```

降级行为都可见、可恢复：模型文件修好后重启即恢复语义检索；重排模型不可用只影响前几名的微调，基础排序照常；监控失败时重启程序即可重试。

---

## 三、配置文件参考

所有字段都可省略，省略时使用默认值。完整带注释示例见 [rummage.toml.example](rummage.toml.example)。

```toml
[watch]
roots = ["D:\\docs", "D:\\notes"] # 未给 --dir 时使用
debounce_ms = 500                 # 文件事件防抖窗口（毫秒）

[index]
dir = "D:\\tools\\rummage\\index" # 省略则为第一个根目录下的 .rummage
extensions = ["md", "markdown", "docx", "pdf", "ofd"] # 参与索引的扩展名，省略即默认值

[search]
candidates = 50                   # 关键词/语义每路先取多少个分块候选（默认 50）
results = 10                      # 最终最多显示多少个文件结果（默认 10；小终端最多 5）

[include]
# 可选白名单；留空即全部收录
# patterns = ["会议纪要*", "2025/**"]

[ignore]
# 追加到内置忽略列表；内置项不可删除。
# 按目录名 / 文件名匹配，不区分大小写，支持 * ? [abc] [!abc]
dirs = ["drafts*", "20??_归档", "临时[0-9]"]
names = ["desktop.ini", "~$*"]

[normalize]
enabled = true                    # 总开关；关闭后所有归一化规则失效
control_chars = true
width_fold = true                 # 全角数字/字母转半角，中文全角标点保留
merge_hard_breaks = true          # 合并 OCR 硬换行
drop_page_number_lines = true
drop_page_markers = true
drop_repeated_headers = true
collapse_blank_lines = true
collapse_spaces = true
strip_markdown = true
links = true
table_pipes = true
front_matter = true
page_number_max_len = 20
header_footer_max_len = 40
header_footer_min_repeat = 3
header_footer_zone_lines = 20

[model]
enabled = true                    # false = 只建关键词索引，状态栏显示“已禁用”
dir = "D:\\tools\\rummage\\models\\bge-small-zh-v1.5"
file = "model.onnx"
batch_size = 32                   # 每批嵌入的文档块数
max_length = 512                  # 分词最大长度

[rerank]
enabled = false                   # 可选二次重排，默认关闭
dir = "D:\\tools\\rummage\\models\\embeddinggemma-2"
file = "onnx/model_quantized.onnx" # 相对 dir 的 ONNX 路径
top_n = 5                         # 只重排基础排序的前 N 个文件
weight = 0.5                      # Gemma 分数权重；0 等于不重排，1 等于完全按 Gemma 排
batch_size = 16                   # 每批嵌入的文档块数
max_length = 1024                 # 分词最大长度
```

`[search]` 控制检索规模：`candidates` 是关键词、语义两路各自先取的**分块**候选数，聚合到文件后再截断到 `results` 条。两者都必须大于 0；省略时保持旧行为（50 / 10）。把 `results` 调大后，`Enter` 详情、`j`/`k` 选择和翻页照常工作，终端高度不够时仍按页滚动。

`[rerank]` 用 EmbeddingGemma 2 对**基础排序的前 `top_n` 个文件**做混合重排：基础分与 Gemma 相似度各自归一化后按 `weight` 加权。第 `top_n + 1` 名及以后完全不动，所以它只影响“哪份文件排第一”，不会把弱相关文件拉进结果。

- 默认关闭：不加载重排模型、不写 `rerank.bin`、检索行为与没有这个功能时完全一致
- 打开后首次索引会为每个文件额外计算重排向量，耗时约为仅主模型的数倍；日常启动按文件哈希与已有向量续跑
- 若前 `top_n` 个文件的重排向量不全、Gemma 分数全部相同或查询为空，则保持基础排序
- 重排模型缺失/损坏/推理失败时只把状态标为 `重排: 未就绪` 并在红字里说明，基础检索不受影响
- 实测（125 个 OFD、手写评测集）：难查询的 MRR 从 0.556 提升到 0.641，简单查询持平；整体提升约 1.2 个百分点，统计上不显著，单次查询多约 0.3 秒。是否值得开由使用场景决定
- 关闭后 `rerank.bin` 不会被删除，再次打开时若模型标识与维度未变可直接复用

路径解析：

- 配置文件中的**相对路径**按配置文件所在目录解析
- `--dir` 给的相对路径也按配置文件所在目录解析；没有配置文件时按当前工作目录
- `--dir` 目录必须存在，否则启动报错；重复目录会自动去重
- `--ignore` 只接受名称模式（目录名或文件名，支持 `*` `?` `[abc]`），不接受完整路径；可重复，本次启动生效

显式指定索引范围：

- 要索引哪些目录：`[watch] roots`（命令行 `--dir` 优先）
- 索引数据存放在哪：`[index] dir`
- 重排模型放在哪：`[rerank] dir`（仅启用重排时需要）
- 哪些扩展名参与索引：`[index] extensions`，默认 `["md", "markdown", "docx", "pdf", "ofd"]`
- 只收录哪些文件：`[include] patterns`，留空表示全部收录
- 其中哪些子目录不索引：`[ignore] dirs`，对全部根目录生效
- 临时排除：命令行 `--ignore`（同时按目录名和文件名匹配，可重复）

`extensions` 的写法可带或不带点、不区分大小写（`"md"`、`".MD"` 等价），重复项会自动去重。`md` / `markdown` / 自定义文本扩展名按 UTF-8 纯文本读取；`docx` 解包正文 XML 取文字；`pdf` 只读取文字层；`ofd` 按国标容器解包页面 XML 取文字。旧版 `.doc`、Excel、PPT 仍需额外的解析层，填写 `doc`、`xls`、`xlsx`、`ppt`、`pptx` 会在启动时报错，不会把二进制内容错误地建进索引。

`docx` 提取正文段落、表格和文本框中的文字，忽略页眉页脚、图片和修订中已删除的文字。`pdf` 使用内置的纯 Rust 解析器，不依赖外部程序或 DLL；支持 `Identity-H`、`UniGB-UCS2-H`、`UniGB-UTF16-H`、`GBK-EUC-H`、`GB-EUC-H` 等常见中文字体编码（有 `ToUnicode` 时优先使用映射表），逐页提取，单页解析失败只跳过该页并保留其余页文字。扫描件、纯图片 PDF 没有文字层，这类文件会被直接跳过（不做 OCR）；加密，或所有页都无法解析的 PDF，会在状态栏报错。PDF 的文字顺序与表格结构由生成器决定，复杂排版可能不如 docx 规整。

`ofd` 同样是 ZIP + XML 容器：按 `OFD.xml` → `Document.xml` 的页序读取每页 `Content.xml`，提取 `TextCode` 文字。页序以 `Document.xml` 的 `BaseLoc` 为准，不按文件名排序；文字按 XML 中的对象顺序拼接。只有图片、没有文字层的扫描版 OFD 会被跳过（不做 OCR），加密或损坏的 OFD 会在状态栏报错。内嵌字体与数字签名不影响文字读取。

`[include] patterns` 是索引白名单，留空或删除该节即全部收录；非空时文件至少命中一条才建索引：

- 不含 `/` 或 `\` 的模式匹配**文件名**，如 `会议纪要*`、`QMS-??.*`，可匹配任意子目录下的文件
- 含分隔符的模式匹配**相对监控根目录的路径**，如 `2025/**`、`制度/*.md`；`*` 不跨目录，`**` 可跨任意层
- 目录写法 `2025/` 等价于 `2025/**`；`./` 前缀和多余的首尾斜杠会被忽略，反斜杠自动转成 `/`
- 不区分大小写；多个模式是“或”的关系
- `[ignore]` 优先于 `[include]`：同一文件既命中白名单又命中忽略规则时，不收录

白名单修改后重启生效；全量扫描完成时，原先已收录但不再命中的文件会自动移出索引。

`[ignore] dirs` / `[ignore] names` 按**单个目录名或文件名**匹配，不匹配完整路径，不区分大小写：

| 通配符 | 含义 | 示例 |
|---|---|---|
| `*` | 任意长度的字符 | `drafts*` 排除 `drafts`、`drafts-2024` |
| `?` | 恰好一个字符 | `20??_归档` 排除 `2024_归档` |
| `[abc]` | 集合中的一个字符 | `临时[0-9]` 排除 `临时3` |
| `[!abc]` | 不在集合中的一个字符 | `备份[!a]` 排除 `备份b`，保留 `备份a` |

精确名称照常可用（如 `归档`）；内置忽略项按精确名称匹配，用户项可与通配符混用。未闭合的 `[` 按字面字符处理。只匹配名称意味着无法只排除“某个特定层级下的同名目录”，需要更具体的命名或单独组织根目录。

内置忽略规则（按路径组件匹配、不区分大小写）：

```text
.git .svn .hg node_modules target __pycache__ .venv venv
.obsidian .trash .idea .vscode attachments assets images 附件 .rummage
```

忽略文件名：`.DS_Store`、`Thumbs.db`。是否收录由 `[index] extensions` 决定，默认索引 `.md` / `.markdown` / `.docx` / `.pdf` / `.ofd`；扫描版 PDF 和纯图片 OFD（无文字层）会被跳过，不做 OCR。

### 3.1 目录局部规则（`.rummage.toml`）

在任意被监控的子目录里新建一个 `.rummage.toml`，该目录及其所有子目录就改按这个文件的规则收录，**不再继承全局的 `[include]` / `[ignore]` / `[index] extensions`**。更深的子目录再放一个规则文件时，以最近的那个为准。内置忽略项（`.git`、`node_modules` 等）始终生效。

局部文件只支持这三段，其他键会解析失败：

```toml
# D:\docs\归档\.rummage.toml
[index]
extensions = ["md", "txt"]

[include]
patterns = ["2024*"]

[ignore]
dirs = ["drafts*"]
names = ["~$*"]
```

- 未写的段取默认值：`extensions` 为 `["md", "markdown", "docx", "pdf", "ofd"]`，`include` 留空表示全部收录，`ignore` 只有内置项。所以局部文件只写 `[ignore]` 时，`include` 和 `extensions` 也会回到默认值。
- 写 `[watch]`、`[model]`、`[normalize]` 或拼错键名会导致解析失败；此时该目录沿用上级规则，状态栏红字给出文件与原因，不影响程序运行。
- 规则文件本身永远不建索引。
- 运行中新增、修改、删除或重命名规则文件，会立即重扫该目录并自动移出不再命中的文件，不需要重启。
- 新建或移入一个目录时，会整棵子树扫描，其中的规则文件同样生效。
- `[normalize]` 是内容处理规则，不属于收录范围，仍然全局统一。

局部文件里的路径模式同样相对**监控根目录**，而不是相对规则文件所在目录。例如在 `D:\docs\归档\.rummage.toml` 中写 `patterns = ["归档/2024/**"]` 表示 `D:\docs` 下的路径。

---

## 四、索引目录与维护

索引默认位于第一个根目录下的 `.rummage\`：

```text
D:\docs\.rummage\
├── index.sqlite3   # 路径、大小、mtime、内容哈希、归一化文本、分块数、嵌入状态（WAL）
├── index.bin       # BM25 倒排 + 向量图快照（含 magic、版本、模型标识、维度）
└── rerank.bin      # 可选：每个文件的重排块向量（仅在启用 [rerank] 时生成）
```

- 快照采用临时文件 + fsync + rename 原子落盘；文件事件处理完、每批嵌入完成后都会更新
- 启动时校验快照与 SQLite 是否一致：分块数不符会重建词法索引，向量缺失会重新嵌入，模型或维度变化会整体重建
- `rerank.bin` 带模型标识与维度；重排模型换了会整体重建重排向量，文件增删改只重算受影响的文件
- 原文 `.md` 只读，索引内容与原始文件完全分开

### 4.1 重建索引

**关闭程序**（`Esc` 或 `Ctrl+C`），确认没有 `rummage.exe` 进程占用索引文件，删除整个索引目录，再按原命令启动即可全量重建：

```powershell
# 默认位置：第一个监控根目录下的 .rummage
Remove-Item -Recurse -Force D:\docs\.rummage

# 如果配置里用 [index] dir 指到了别处，就删那个目录
Remove-Item -Recurse -Force D:\tools\rummage\index

# 重新启动，会重新扫描、切块、嵌入
D:\tools\rummage\rummage.exe --dir D:\docs
```

需要重建的典型情况：

- 更换嵌入模型，或修改了 `[normalize]` 规则
- 根目录被移动、重命名，或大量文件在程序未运行时被移动
- 怀疑索引损坏（状态栏异常、检索结果明显不全）
- 调整了切块/归一化等会影响原文表示的实现版本

删除索引目录是安全操作：程序只读原始文档，索引里没有任何原始文件的唯一副本。删除后原文件不受影响。

### 4.2 清理索引

“清理索引”和“重建索引”是同一个动作：删除索引目录即可把已建立的索引全部清空。区别只在于删完后是否马上重启：

```powershell
# 只清掉索引，不再使用这个文档库
Remove-Item -Recurse -Force D:\docs\.rummage

# 清掉后重新建立
D:\tools\rummage\rummage.exe --dir D:\docs
```

- 默认索引目录是**第一个根目录**下的 `.rummage`；配置了 `[index] dir` 时以配置为准
- 多个根目录共用同一个索引目录（默认是第一个根目录下的 `.rummage`），删一次即可全部清空
- 索引目录里还有 SQLite 的 `-wal` / `-shm` 和临时文件，随目录一起删除即可
- 程序没有单独的 `purge` / `rebuild` 子命令；删除目录再启动就是官方做法

只想重建重排向量时，删除 `rerank.bin` 后重启即可，不必重建整个索引：

```powershell
Remove-Item -Force D:\docs\.rummage\rerank.bin
```

只想让某些目录/文件退出索引、但保留其余索引时，不要删目录：用 `[ignore]` 或命令行 `--ignore` 排除后重启，扫描完成时会把已收录但不再命中的文件移出索引。

**不要**把 `.rummage` 放进同步盘或放进被监控目录的子目录中——它已在内部忽略列表里，但同步工具仍可能争用文件。需要把索引放在别处时用 `[index] dir`；写成绝对路径（如 `D:\\rummage-index`）后，被索引的目录里不会产生任何文件。

索引约占原文大小的一到数倍（取决于分块数量与向量维度）。删除 `.rummage` 不影响任何原始文档。

---

## 五、常见问题

| 现象 | 原因与处理 |
|---|---|
| 启动即退出，提示 `未指定监控目录` | 既没给 `--dir`，配置里也没有 `[watch] roots`。补上其一 |
| 提示 `监控目录不存在或不可访问` | `--dir` 路径写错，或网络盘未挂载 |
| 提示配置文件不存在 / 解析失败 | `--config` 路径写错，或 TOML 语法错误（注意 Windows 路径要写双反斜杠） |
| 状态栏 `模型: 未就绪(仅关键词)` | 模型目录里缺文件或 `model.onnx` 损坏。按 1.1 的五个文件核对；右侧红字有具体原因。检索仍可用，只是没有语义能力 |
| 状态栏 `重排: 未就绪` | 只影响可选的二次重排：检查 `[rerank] dir` / `file` 与 `tokenizer.json` 是否存在；基础关键词+语义检索不受影响。右侧红字有具体原因 |
| 启用重排后首次索引很慢 | 重排模型要为每个文件再算一遍分块向量，耗时约为仅主模型的数倍；完成后按哈希与已有向量续跑，后续启动不会重算 |
| 状态栏 `监控: 未运行` | 文件监听启动失败（多为权限或路径问题）。检索可用，增删改需重启程序才生效；右侧红字给出原因 |
| 状态栏一直显示 `· 扫描` / `· 语义` / `· 重排`、文件数不动 | 正在首次全量索引，看百分比即可判断是否在推进；`Esc` 可中断，下次续跑 |
| 首次启动很慢 | 语义向量计算是主要开销；文档多时可先用 `[model] enabled = false` 只建关键词索引 |
| 检索结果少于预期 | 默认最多 10 条（小终端 5 条，可用 `[search] results` 调整）；确认文件后缀在 `[index] extensions` 中（默认 `.md` / `.markdown` / `.docx` / `.pdf` / `.ofd`），且不在忽略目录中 |
| 想调整候选数量 / 结果条数 | 在 `[search]` 写 `candidates`（每路分块候选，默认 50）和 `results`（最终文件数，默认 10），重启生效；两者必须大于 0 |
| 怎么重建 / 清理索引 | 关闭程序后删除索引目录再启动，见 [4.1](#41-重建索引) 与 [4.2](#42-清理索引)；默认是第一个根目录下的 `.rummage`，配置了 `[index] dir` 时删那个目录 |
| 退出时还在索引，进程要等一会 | 退出会先停止调度，正在计算的一小批（`[model]` / `[rerank]` 的 `batch_size`）完成后落盘退出；调小 `batch_size` 可更快退出，但索引吞吐会下降 |
| 能否索引 Word / PDF / OFD？ | `.docx`、带文字层的 `.pdf` 和 `.ofd` 默认就会索引；旧版 `.doc`、Excel、PPT 暂不支持。扫描件 PDF 和纯图片 OFD 需要先做 OCR 转成文字，本程序不内置 OCR |
| 扫描版 PDF / 纯图片 OFD 搜不到 | 设计如此：没有文字层就跳过，不建空索引。先用 OCR 工具生成文字版文档，或转成 Markdown |
| PDF 文字顺序乱 / 表格串行 | PDF 只存绘制指令，文字顺序取决于生成器；对排版要求高时优先用 docx 或转成 Markdown |
| 个别 PDF 少了几页文字 | 单页解析失败会被跳过，其余页照常入库；状态栏只在整份 PDF 一页都取不到文字时报错 |
| OFD 页序不对 | 程序按 `Document.xml` 的 `BaseLoc` 顺序读页，不按 `Page_N` 文件名排序；若仍不对，是生成工具写错了页序 |
| 改了 PDF / OFD 但索引没变 | 文件内容哈希未变时不会重解析；确认文件确实保存成功。索引只读原文件，不会改动 PDF / OFD |
| 改成 `extensions = ["txt"]` 后没生效 | 配置只在启动时读取，需重启；扩展名不带点也可以，扫描完成时原先不在白名单内的文件会自动移出索引 |
| 想排除某些子目录 | 在 `[ignore] dirs` 写目录名或通配符（如 `drafts*`）后重启，或启动时加 `--ignore drafts*`；扫描完成时已收录的相关文件会自动移出索引，无需手动删除 `.rummage` |
| 某个子目录想用另一套规则 | 在该目录放一个 `.rummage.toml`，只写 `[index] extensions` / `[include] patterns` / `[ignore]`；它会完全替换全局规则，见 [3.1](#31-目录局部规则rummagetoml) |
| 局部规则文件写错了 | 状态栏红字提示文件名与原因，该目录沿用上级规则；修好后保存即自动重扫，不用重启 |
| 只想索引部分文件 | 用 `[include] patterns`，如 `["会议纪要*"]` 或 `["2025/**"]`；`*` 不跨目录，`**` 跨目录，`ignore` 优先。重启后生效 |
| 按 `q` 退不出去 | 退出键是 `Esc` 或 `Ctrl+C`；`q` 现在只作为普通字符输入，避免查询框里打不出英文 `q` |
| `c` 没有复制 | 焦点需在结果列表（`Tab` 切换），且要有选中项；剪贴板被占用时状态栏会提示 |
| 界面错乱或最小尺寸提示 | 使用 Windows Terminal，窗口至少 80×20；放大后自动恢复 |
| 中文全角标点被改成半角？ | 不会。`width_fold` 只转全角数字和字母，中文全角标点保留 |

---

## 六、迁移到内网 / 离线运行

运行时完全离线，联网只发生在**构建**阶段。

1. **Rust 工具链**：Windows 离线安装包（rustup + MSVC 组件）。
2. **crate 依赖**：外网机器执行 `cargo vendor`，把生成的 `vendor/` 与 `.cargo/config.toml`（source replacement）带入内网。本仓库的 `vendor/scour-search` 是打了持久化与增量 API 补丁的本地路径依赖，需原样保留。
3. **ONNX Runtime 静态库**：`ort` 开启了 `download-binaries`，默认构建时会下载静态库。内网构建前，把外网机器上的
   `%LOCALAPPDATA%\ort.pyke.io\dfbin\x86_64-pc-windows-msvc\<哈希>\` 目录整体带入内网，并设置 `ORT_LIB_LOCATION` 指向它。
   然后 `cargo build --release --offline` 即可，不再联网。
4. **模型文件**：按 1.1 的目录布局整体复制；启用重排时再带上 `models\embeddinggemma-2\`。
5. **最简单的方式**：直接分发外网编译好的 `rummage.exe` + `models\` 目录。运行时不需要 Rust，也不需要额外分发 ONNX Runtime DLL。

---

## 七、从源码构建与测试

环境：Windows 11 x64、Rust 1.87+（MSVC 工具链，本机以 Visual Studio 2022 验证）。

```powershell
cargo build --release
```

产物为 `target\release\rummage.exe`。注意它默认在自身同目录找模型，见 1.1。

```powershell
cargo test                              # 123 个单元测试 + 2 个监控集成测试，不需要模型
cargo test --test e2e_model -- --ignored # 真实 ONNX 模型端到端（需 models\ 下的文件）
cargo clippy --all-targets
```

本机已实测：

- 真实模型下中文自然语言查询的语义排序正确；重启后快照复用，仍可检索
- 启用重排后真实 TUI 实测（10 份 OFD）：状态栏出现 `重排: 就绪`，`rerank.bin` 生成 768 维向量且文件数、块数与 SQLite 一致；query 前 5 名顺序相对关闭重排时发生变化
- 重启后重排向量直接复用（`rerank.bin` 内容哈希不变、无重新嵌入）；`enabled = false` 时不显示重排状态、结果回到基础排序；模型目录缺失时显示 `重排: 未就绪` 并在红字给出原因，基础检索与结果列表不受影响
- 真实 TUI（Windows PTY，110×30）：查询、结果路径、状态栏（监控/索引数/模型/焦点）渲染正常
- 结果摘录（90×20 真实 PTY，12 份文档）：每条结果下显示最佳块摘录，命中词为黄色加粗；连续下移 9 次后按页滚动，选中项与可见条数正确，索引仍只写在 `index\` 内
- `Tab` 切换焦点（结果区底部出现 `c 复制路径 / j/k 选择 / Enter 详情` 提示）、`↑/↓/j/k` 选择、`c` 复制路径（读取系统剪贴板核对内容）、`Esc` 退出
- 详情叠层（110×30 真实 PTY，4 份文档 + 真实 bge/Gemma）：结果区按 `Enter` 显示关键词/语义名次、RRF 融合分、Gemma 重排分与名次变化、最佳块全文；`j` 滚动不误移动列表，第一次 `Esc` 只关闭叠层且进程存活，第二次 `Esc` 才退出，索引仍写在监控目录外
- 索引进度（120×20 真实 PTY，1200 份 Markdown + 真实 bge/Gemma）：状态栏依次出现 `· 扫描 0%…98%`、`· 语义 0%…98%`、`· 重排 0%…98%`；全部完成后百分比消失，回到 `索引: 1200 文件`，随后的查询结果正常
- 79×19 降级布局：隐藏状态栏、只显示 5 条、显示“终端过小”
- 运行中新增/删除 Markdown：状态栏文件数 1→2→1，新增文件可立即检索
- 运行中新增/删除目录局部规则文件：状态栏文件数 2→1→2，无需重启
- 索引 docx 正文、PDF 文字层与 OFD 文字层；无文字层的 PDF / OFD 被跳过，不产生空索引
- 三份实测 PDF（`UniGB-UCS2-H` / `GBK-EUC-H` 编码、含损坏内容流）均能提取中文正文：原报错的文件分别得到 1.1 万、57 万、0.5 万字，过程中无 panic 输出；单页损坏时其余页照常保留
- 594 页机考题库 PDF（含 5 个异常路径操作符页面）：修复后逐页提取 0 次 panic，原先被跳过的 5 页恢复出 426/249/249/224/434 字正文；全文 468851 字（修复前 467951）
- 109 份高风险 PDF（异常 CMap/编码/损坏内容流）回归：106 份正常提取，3 条错误来自 2 个文件（文件级 xref/trailer 损坏，非 panic，需专门的修复解析），逐页 panic 为 0
- 索引进行中按 `Esc` 可及时退出，不阻塞在扫描上
- 语义索引进行中按 `Esc`（250 份文档、真实 bge）：停止信号发出后约 0.3 秒进程退出，不会等整批文件算完
- `--ignore` + `[search] results`（真实 PTY，含 `skipdir` 与 `~$temp.md`）：被忽略的目录/文件不进入 SQLite，查询只显示配置的 3 条结果
- 原文文件字节在索引前后完全一致

目录结构：

```text
src/
├── main.rs        CLI 入口、装配监控/索引/TUI、退出清理
├── config.rs      配置、路径解析、收录白名单、忽略规则、文件类型、目录局部规则
├── extract.rs     docx / pdf / ofd / 纯文本提取
├── normalize.rs   12 条 OCR 归一化规则
├── query.rs       dir:/name:/"短语" 解析
├── tokenize.rs    jieba 分词 + 中文停用词
├── store.rs       SQLite 元数据
├── embed.rs       fastembed/ONNX 嵌入
├── rerank.rs      EmbeddingGemma 2 二次重排：加载、块向量、混合排序、持久化
├── persist.rs     bincode 快照原子读写
├── engine.rs      混合索引、增量更新、检索与聚合
├── watcher.rs     notify + 防抖
├── worker.rs      索引线程与消息协议
└── tui.rs         ratatui 界面、按键、剪贴板
vendor/scour-search  打上 serde/增量 API 补丁的 BM25+HNSW+RRF 实现
tests/               模型端到端、监控集成测试
```

---

## 八、与需求 v1.3 的差异

- **向量维度是 512，不是 384**：`BAAI/bge-small-zh-v1.5` 与 `Xenova/bge-small-zh-v1.5` 的 `hidden_size` 都是 512（`config.json` 可查）。代码在加载时动态探测维度并写入快照，不硬编码。
- **退出键是 `Esc` 而非 `q`**：需求表格写的是 `q` 全局退出，但那样查询框内无法输入英文 `q`。现改为 `Esc`（详情层内先关闭叠层）或 `Ctrl+C` 退出；`c` 仅在结果焦点复制。
- **文档分块**：约 1024 字节、重叠 128 字节；检索结果按文件聚合取最高分，满足“定位到文件、不要求段落级”的要求。
- **中文停用词**：jieba 分词后过滤“的/了/是/如何/怎样”等虚词，避免 BM25 被“的”这类高频字带偏、压过语义排序。
- **索引快照版本**：首版快照格式带版本号，不兼容时自动重建；SQLite schema 同样有版本校验。
- **状态栏降级**：监控启动失败时仍可检索，状态栏显示“未运行”并在错误区给出原因；模型不可用时显示“未就绪(仅关键词)”。
- **检索规模可配置**：需求固定“Top 50 候选、最多 10 条结果”，现改为 `[search] candidates` / `results`，默认值保持 50 / 10；另增命令行 `--ignore`，便于临时排除目录或文件。
- **目录局部规则**：需求之外增加了 `.rummage.toml`，允许按目录覆盖 `extensions` / `include` / `ignore`；内置忽略项与 `[normalize]` 仍全局生效。
- **docx / pdf / ofd 直接索引**：需求之外的格式扩展。docx 解包正文 XML 取文字；pdf 只读文字层；ofd 按国标容器读取页面 `TextCode`，页序以 `Document.xml` 为准。无文字层（扫描件、纯图片页）按“不用管”直接跳过，不引入 OCR，也不依赖外部程序或 DLL。
- **PDF 中文字体编码**：内置解析器原本只认 `Identity-H`，遇到 `UniGB-UCS2-H` / `GBK-EUC-H` 会直接 panic。现用仓库内 vendored 版本补上这几类预定义 CMap 的直接解码，并把逐份提取改为逐页提取：损坏页计入跳过，其余页文字保留，解析器 panic 不再打印到终端。
- **PDF 解析健壮性**：继续收口 vendored 解析器的 panic 分支：未知 simple-font `/Encoding` 回退到默认表、`Differences` 越界跳过、缺 `MediaBox` 用 Letter 尺寸、路径操作符缺操作数时忽略而不是 abort、`Tf`/`gs`/`Do` 引用缺失对象时跳过、损坏 UTF-16 不 panic、颜色空间与函数解析异常降级。594 页实测文件由 5 页 panic 降为 0；109 份高风险样本逐页 panic 为 0。
- **可选二次重排**：需求之外的排序增强。用 EmbeddingGemma 2 对基础排序前 5 名做混合重排，默认关闭；通过 `[rerank]` 打开，关闭时零开销。它不替代主检索链路，模型不可用时只降级重排本身。
