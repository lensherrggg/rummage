use rummage::config::Config;
use rummage::engine::{IndexEngine, ModelStatus};
use rummage::query;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("bge-small-zh-v1.5")
}

fn engine_for(root: &Path) -> IndexEngine {
    IndexEngine::new(
        Config::default(),
        vec![root.to_path_buf()],
        root.join(".rummage"),
        model_dir(),
        model_dir().join("rerank"),
    )
    .unwrap()
}

fn embed_all(engine: &mut IndexEngine) {
    while engine.has_pending_embeddings() {
        let batch = engine.next_embed_batch().expect("pending batch");
        engine.embed_documents(&batch).unwrap();
    }
}

#[test]
#[ignore = "需要 models/ 下的真实 ONNX 模型，运行：cargo test --test e2e_model -- --ignored"]
fn semantic_search_with_real_model() {
    let dir = tempdir().unwrap();
    let ocr =
        "# OCR 预处理\n\n本文介绍如何把被 OCR 截断的句子重新接成完整段落，属于文本后处理流程。\n";
    let finance = "本季度财务报表显示营业收入同比增长百分之十。\n";
    let patrol = "设备巡检记录：变压器温度正常，无渗漏。\n";
    std::fs::write(dir.path().join("ocr.md"), ocr).unwrap();
    std::fs::write(dir.path().join("finance.md"), finance).unwrap();
    std::fs::write(dir.path().join("patrol.md"), patrol).unwrap();

    let mut engine = engine_for(dir.path());
    match &engine.status(false).model {
        ModelStatus::Ready { dims } => assert_eq!(*dims, 512),
        other => panic!("模型未就绪：{other:?}"),
    }
    engine.scan_lexical(|_| {}).unwrap();
    embed_all(&mut engine);

    let hits = engine
        .search(&query::parse("怎样把断开的文字拼接起来"))
        .unwrap();
    assert!(!hits.is_empty(), "语义检索没有结果");
    assert!(
        hits[0].path.ends_with("ocr.md"),
        "语义检索首选错误：{:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    let hits = engine
        .search(&query::parse("公司今年的经营收入情况"))
        .unwrap();
    assert!(
        hits[0].path.ends_with("finance.md"),
        "语义检索首选错误：{:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    engine.persist().unwrap();
    drop(engine);

    let mut reopened = engine_for(dir.path());
    assert_eq!(reopened.status(false).indexed_files, 3);
    let hits = reopened.search(&query::parse("变压器检查结果")).unwrap();
    assert!(
        hits[0].path.ends_with("patrol.md"),
        "重启后语义检索失败：{:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    assert_eq!(
        std::fs::read(dir.path().join("ocr.md")).unwrap(),
        ocr.as_bytes()
    );
    assert_eq!(
        std::fs::read(dir.path().join("finance.md")).unwrap(),
        finance.as_bytes()
    );
}
