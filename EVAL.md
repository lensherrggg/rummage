# Rummage 检索质量评估

这是离线评测工具，不会启动 TUI，也不会自动建索引。先正常用 Rummage 建好索引、退出程序，再跑评估。

## 1. 构建

```powershell
cargo build --release --bin rummage-eval --offline
```

产物：

```text
target\release\rummage-eval.exe
```

## 2. 查询集格式

标注流程、相关性判定、给其他 AI agent 的提示词见 [ANNOTATION.md](ANNOTATION.md)。

一行一个 JSON 对象（JSONL），路径写索引中的真实绝对路径：

```json
{"id":"q001","query":"党委会纪要 2025","type":"keyword","relevant":["D:\\退休就删\\资料文档\\a.md"]}
{"id":"q002","query":"光伏电站运维中的树线矛盾","type":"semantic","relevant":["D:\\退休就删\\资料文档\\b.pdf"]}
{"id":"q003","query":"结算管理质效","type":"keyword","relevant":[],"grades":{"D:\\退休就删\\资料文档\\c.docx":2,"D:\\退休就删\\资料文档\\d.docx":1}}
```

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 查询唯一标识 |
| `query` | 是 | 查询文本 |
| `type` | 否 | 分组标签，默认 `default`；建议 `keyword` / `semantic` / `ocr` / `filter` |
| `relevant` | 否 | 相关文件路径数组，二元相关 |
| `grades` | 否 | 分级相关：`2` 高度相关、`1` 部分相关；存在时优先于 `relevant` |

没有标签的查询会保留在逐查询输出里，但不计入汇总指标。

## 3. 运行

三路一起跑：

```powershell
.\target\release\rummage-eval.exe `
  --config D:\tools\rummage-portable-0.1.0-full\rummage.toml `
  --queries eval\queries.jsonl `
  --top 10 `
  --out eval\result.json
```

只跑单路：

```powershell
--modes lexical
--modes semantic
--modes hybrid
```

扫描 RRF 参数：

```powershell
--rrf-k 10
--lexical-weight 1.0 --semantic-weight 0.5
```

默认关闭重排，避免单路基线被交叉编码器干扰；要测完整链路加：

```powershell
--rerank
```

`--rerank` 只影响 `hybrid`：交叉编码器按 `[rerank] top_n` / `chunks` / `weight` 调整混合排序，单路基线不受影响。

`--top 10` 会同时把 `[search] results` 调到 10，并把 `candidates` 至少调到 10。评估时建议所有模式使用相同的 `--top` 和 `candidates`。

## 4. 指标

| 指标 | 含义 |
|---|---|
| `Recall@K` | 前 K 个结果覆盖了多少比例的相关文件 |
| `MRR@K` | 第一个相关文件排名的倒数；没有则 0 |
| `nDCG@K` | 考虑位置和分级相关性的排序质量 |
| `Success@K` | 前 K 是否至少命中一个相关文件 |

输出会按 `type` 分组，并保留每条查询的 `first_relevant_rank` 和 Top K 路径。三路同时运行时，末尾会额外输出“融合税”统计：混合结果的首个相关排名劣于较好单路的查询数 / 至少一路命中的查询数；JSON 的 `fusion_tax` 字段包含明细样本。

## 5. 建议流程

1. 先做 50 条分层查询，不要一开始标 100–200 条
2. 跑 `lexical` / `semantic` / `hybrid` 三组基线
3. 对每条查询比较 `first_relevant_rank`，统计融合后排名劣于较好单路的查询
4. 只保留 5–10 条 OCR 噪声查询做专项对照
5. RRF 参数在 dev 集上选，test 集只做最终确认

## 6. 注意

- 评测前关闭 Rummage，避免 SQLite 写入竞争
- `index` 必须已经建好；`rummage-eval` 不会扫描文档
- 标注路径必须和索引中的路径一致；大小写、`/` 与 `\` 会自动归一化
- `Recall@K` 是相对标注集计算的；未标注文件按不相关处理，不要在报告里把它说成严格全库 Recall
- 语义模式需要 `[model]` 指定的句向量模型可用；模型缺失时语义结果为空
- `--rrf-k` / 权重覆盖发生在加载配置之后；非法值会被 `rummage-eval` 拒绝
- 对比 OCR 噪声时，干净版和 OCR 版是两个索引，查询文件里的 `relevant` 路径也要分别对应各自索引
