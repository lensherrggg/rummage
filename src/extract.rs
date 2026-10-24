use crate::ocr;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io::{Cursor, Read, Seek};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

/// Extract plain text from a document. `None` means the format was parsed but
/// has no text layer (for example a scanned PDF), so the file is skipped
/// instead of being stored as an empty document.
pub fn extract_text(path: &Path, bytes: &[u8]) -> Result<Option<String>> {
    match extension(path).as_deref() {
        Some("docx") => non_empty(docx_text(bytes)?),
        Some("ofd") => non_empty(ofd_text(bytes)?),
        Some("pdf") => non_empty(pdf_text(bytes)?),
        _ => Ok(Some(String::from_utf8_lossy(bytes).into_owned())),
    }
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(OsStr::to_str)
        .map(str::to_lowercase)
}

fn non_empty(text: String) -> Result<Option<String>> {
    Ok((!text.trim().is_empty()).then_some(text))
}

fn docx_text(bytes: &[u8]) -> Result<String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).context("不是有效的 docx（ZIP 解包失败）")?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .context("docx 缺少 word/document.xml")?
        .read_to_string(&mut xml)
        .context("读取 docx 正文失败")?;
    let document = roxmltree::Document::parse(&xml).context("docx 正文 XML 解析失败")?;
    let mut out = String::new();
    collect_word_text(document.root(), &mut out);
    Ok(out)
}

fn collect_word_text(node: roxmltree::Node, out: &mut String) {
    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "t" => {
                for text in child.descendants().filter(|n| n.is_text()) {
                    out.push_str(text.text().unwrap_or(""));
                }
            }
            // Tracked deletions and field instructions are not document text.
            "del" | "delText" | "moveFrom" | "instrText" | "rPr" | "pPr" => {}
            "tab" => out.push('\t'),
            "br" | "cr" => out.push('\n'),
            "noBreakHyphen" => out.push('-'),
            "p" => {
                collect_word_text(child, out);
                out.push('\n');
            }
            _ => collect_word_text(child, out),
        }
    }
}

fn ofd_text(bytes: &[u8]) -> Result<String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).context("不是有效的 OFD（ZIP 解包失败）")?;
    let ofd = read_zip_text(&mut archive, "OFD.xml").context("OFD 缺少 OFD.xml")?;
    let mut pages = Vec::new();

    for doc_root in ofd_doc_roots(&ofd)? {
        let doc_path = ofd_path("", &doc_root);
        let doc = read_zip_text(&mut archive, &doc_path)
            .with_context(|| format!("读取 OFD 文档失败：{doc_path}"))?;
        let base = doc_path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");

        for loc in ofd_page_refs(&doc)? {
            let page_path = ofd_path(base, &loc);
            let page = read_zip_text(&mut archive, &page_path)
                .with_context(|| format!("读取 OFD 页面失败：{page_path}"))?;
            let text = ofd_page_text(&page)?;
            if !text.trim().is_empty() {
                pages.push(text);
            }
        }
    }

    Ok(pages.join("\n"))
}

fn ofd_doc_roots(xml: &str) -> Result<Vec<String>> {
    let document = roxmltree::Document::parse(xml).context("OFD.xml 解析失败")?;
    let roots: Vec<String> = document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "DocRoot")
        .map(|node| element_text(node).trim().to_string())
        .filter(|root| !root.is_empty())
        .collect();
    if roots.is_empty() {
        bail!("OFD.xml 缺少 DocRoot");
    }
    Ok(roots)
}

fn ofd_page_refs(xml: &str) -> Result<Vec<String>> {
    let document = roxmltree::Document::parse(xml).context("OFD Document.xml 解析失败")?;
    Ok(document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "Page")
        .filter_map(|node| node.attribute("BaseLoc"))
        .map(str::to_string)
        .collect())
}

fn ofd_page_text(xml: &str) -> Result<String> {
    let document = roxmltree::Document::parse(xml).context("OFD 页面 XML 解析失败")?;
    let mut out = String::new();
    // TextObject order in the XML is reading order. TextCode X/Y are relative
    // to their own TextObject, so they cannot be used for global sorting.
    for node in document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "TextCode")
    {
        out.push_str(&element_text(node));
    }
    Ok(out)
}

fn element_text(node: roxmltree::Node) -> String {
    node.descendants()
        .filter(|child| child.is_text())
        .filter_map(|child| child.text())
        .collect()
}

fn ofd_path(base: &str, loc: &str) -> String {
    let loc = loc.trim().replace('\\', "/");
    if let Some(absolute) = loc.strip_prefix('/') {
        return absolute.to_string();
    }
    let loc = loc.strip_prefix("./").unwrap_or(&loc);
    if base.is_empty() {
        loc.to_string()
    } else {
        format!("{}/{}", base.trim_end_matches('/'), loc)
    }
}

fn read_zip_text<R: Read + Seek>(archive: &mut zip::ZipArchive<R>, name: &str) -> Result<String> {
    let mut text = String::new();
    archive
        .by_name(name)
        .with_context(|| format!("缺少 {name}"))?
        .read_to_string(&mut text)
        .with_context(|| format!("读取 {name} 失败"))?;
    Ok(text)
}

fn pdf_text(bytes: &[u8]) -> Result<String> {
    // pdf-extract can panic on malformed files (and on fonts it cannot map).
    // Extract page by page so one broken page does not discard the rest, and
    // keep panic text off the terminal by suppressing the hook while we run.
    silence_pdf_panics(|| pdf_text_inner(bytes))
        .unwrap_or_else(|_| Err(anyhow!("PDF 提取失败：解析器内部错误")))
}

fn pdf_text_inner(bytes: &[u8]) -> Result<String> {
    let mut doc = pdf_extract::Document::load_mem(bytes).map_err(|err| anyhow!("PDF 提取失败：{err}"))?;
    if doc.is_encrypted() {
        doc.decrypt("")
            .map_err(|_| anyhow!("PDF 已加密，无法提取文字"))?;
    }
    let pages = doc.get_pages();
    let mut page_texts: BTreeMap<u32, String> = BTreeMap::new();
    let mut failed = 0usize;

    for &page_num in pages.keys() {
        let mut page = String::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut output = pdf_extract::PlainTextOutput::new(&mut page);
            pdf_extract::output_doc_page(&doc, &mut output, page_num)
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                failed += 1;
                log::warn!("PDF 第 {page_num} 页解析失败，已跳过：{err}");
            }
            Err(_) => {
                failed += 1;
                log::warn!("PDF 第 {page_num} 页触发解析器 panic，已跳过");
            }
        }
        // A panic mid-page can still leave the text written so far, so keep it.
        if !page.trim().is_empty() {
            page_texts.insert(page_num, page.trim_end().to_string());
        }
    }

    if ocr::textless_pdf_enabled() && ocr::is_enabled() {
        let min_chars = ocr::min_text_chars();
        let missing: Vec<u32> = pages
            .keys()
            .copied()
            .filter(|page| {
                page_texts
                    .get(page)
                    .map(|text| text.trim().chars().count() < min_chars)
                    .unwrap_or(true)
            })
            .collect();
        if !missing.is_empty() {
            log::info!(
                "PDF 有 {} 页缺少文字层，尝试 OCR（文件哈希缓存）",
                missing.len()
            );
            let ocr_texts = ocr::recognize_pages(bytes, &missing);
            for (page, text) in ocr_texts {
                if !text.trim().is_empty() {
                    page_texts.insert(page, text);
                }
            }
        }
    }

    let out = page_texts
        .values()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");

    if failed > 0 {
        log::warn!("PDF 提取完成：{} 页中 {failed} 页失败，其余页已保留", pages.len());
    }
    if out.trim().is_empty() && failed > 0 {
        bail!("PDF 提取失败：{failed}/{} 页解析错误", pages.len());
    }
    Ok(out)
}

thread_local! {
    static SUPPRESS_PANIC_OUTPUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn silence_pdf_panics<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    static INSTALL_HOOK: std::sync::Once = std::sync::Once::new();
    INSTALL_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !SUPPRESS_PANIC_OUTPUT.with(std::cell::Cell::get) {
                previous(info);
            }
        }));
    });

    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SUPPRESS_PANIC_OUTPUT.with(|flag| flag.set(self.0));
        }
    }

    let previous = SUPPRESS_PANIC_OUTPUT.with(|flag| flag.replace(true));
    let _restore = Restore(previous);
    catch_unwind(AssertUnwindSafe(f))
}

#[cfg(test)]
pub(crate) fn make_docx(xml: &str) -> Vec<u8> {
    use std::io::Write as _;

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("word/document.xml", options).unwrap();
        zip.write_all(xml.as_bytes()).unwrap();
        zip.finish().unwrap();
    }
    cursor.into_inner()
}

#[cfg(test)]
pub(crate) fn make_ofd(pages: &[(&str, &str)]) -> Vec<u8> {
    use std::fmt::Write as _;
    use std::io::Write as _;

    let mut refs = String::new();
    for (index, (loc, _)) in pages.iter().enumerate() {
        write!(refs, r#"<ofd:Page ID="{index}" BaseLoc="{loc}"/>"#).unwrap();
    }
    let document = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><ofd:Document xmlns:ofd="http://www.ofdspec.org/2016"><ofd:Pages>{refs}</ofd:Pages></ofd:Document>"#
    );
    let ofd = r#"<?xml version="1.0" encoding="UTF-8"?><ofd:OFD xmlns:ofd="http://www.ofdspec.org/2016" DocType="OFD" Version="1.1"><ofd:DocBody><ofd:DocRoot>Doc_0/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>"#;

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("OFD.xml", options).unwrap();
        zip.write_all(ofd.as_bytes()).unwrap();
        zip.start_file("Doc_0/Document.xml", options).unwrap();
        zip.write_all(document.as_bytes()).unwrap();
        for (loc, text) in pages {
            let body = if text.is_empty() {
                r#"<ofd:ImageObject ID="9" Boundary="0 0 10 10"/>"#.to_string()
            } else {
                format!(
                    r#"<ofd:TextObject ID="2" Boundary="0 0 10 10"><ofd:TextCode X="0" Y="5">{text}</ofd:TextCode></ofd:TextObject>"#
                )
            };
            let page = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?><ofd:Page xmlns:ofd="http://www.ofdspec.org/2016"><ofd:Content><ofd:Layer>{body}</ofd:Layer></ofd:Content></ofd:Page>"#
            );
            zip.start_file(format!("Doc_0/{loc}"), options).unwrap();
            zip.write_all(page.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }
    cursor.into_inner()
}

/// Build a one-page PDF with a text layer using the built-in Helvetica font.
#[cfg(test)]
pub(crate) fn make_pdf(content: &str) -> Vec<u8> {
    make_pdf_pages(&[content])
}

/// Build a one-page PDF with a caller-supplied font dictionary. Used to
/// exercise malformed or unusual simple-font encodings.
#[cfg(test)]
fn make_pdf_with_font(font: &str, content: &str) -> Vec<u8> {
    let objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_string(),
        format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        ),
        font.to_string(),
    ];
    assemble_pdf(&objects)
}

#[cfg(test)]
fn make_pdf_pages(contents: &[&str]) -> Vec<u8> {
    let font_id = 3 + 2 * contents.len();
    let mut kids = Vec::new();
    let mut objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        String::new(),
    ];
    for (index, content) in contents.iter().enumerate() {
        let page_id = 3 + 2 * index;
        let content_id = page_id + 1;
        kids.push(format!("{page_id} 0 R"));
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        ));
    }
    objects[1] = format!(
        "<< /Type /Pages /Kids [{}] /Count {} >>",
        kids.join(" "),
        contents.len()
    );
    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string());
    assemble_pdf(&objects)
}

/// Build a one-page PDF whose text uses a predefined CJK CMap (`Encoding`).
#[cfg(test)]
pub(crate) fn make_type0_pdf(encoding: &str, hex: &str) -> Vec<u8> {
    let content = format!("BT /F1 24 Tf 72 720 Td <{hex}> Tj ET");
    let objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_string(),
        format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        ),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /SimSun /Encoding /{encoding} \
             /DescendantFonts [6 0 R] >>"
        ),
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /SimSun \
         /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> \
         /FontDescriptor 7 0 R /DW 1000 >>"
            .to_string(),
        "<< /Type /FontDescriptor /FontName /SimSun /Flags 4 /FontBBox [0 0 1000 1000] \
         /ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 880 /StemV 80 >>"
            .to_string(),
    ];
    assemble_pdf(&objects)
}

#[cfg(test)]
fn assemble_pdf(objects: &[String]) -> Vec<u8> {
    let mut out = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (index, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.push_str(&format!("{} 0 obj\n{body}\nendobj\n", index + 1));
    }
    let xref = out.len();
    out.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
    out.push_str("0000000000 65535 f \n");
    for offset in offsets {
        out.push_str(&format!("{offset:010} 00000 n \n"));
    }
    out.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    ));
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docx_extracts_paragraphs_runs_tabs_and_skips_deleted_text() {
        let xml = r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
              <w:p><w:r><w:t>光伏电站</w:t></w:r><w:r><w:t>巡检记录</w:t></w:r></w:p>
              <w:p><w:r><w:t>组件&amp;清洁</w:t></w:r><w:tab/><w:t>温度正常</w:t></w:p>
              <w:p><w:del><w:r><w:delText>已删除内容</w:delText></w:r></w:del><w:r><w:t>保留内容</w:t></w:r></w:p>
            </w:body>
        </w:document>"#;
        let bytes = make_docx(xml);
        let text = extract_text(Path::new("a.docx"), &bytes)
            .unwrap()
            .expect("docx text");
        assert!(text.contains("光伏电站巡检记录"), "{text}");
        assert!(text.contains("组件&清洁\t温度正常"), "{text}");
        assert!(!text.contains("已删除内容"), "{text}");
        assert!(text.contains("保留内容"), "{text}");
    }

    #[test]
    fn textless_docx_is_skipped() {
        let xml = r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p/></w:body></w:document>"#;
        assert!(extract_text(Path::new("a.docx"), &make_docx(xml))
            .unwrap()
            .is_none());
    }

    #[test]
    fn corrupt_docx_reports_an_error() {
        let err = extract_text(Path::new("a.docx"), b"not a zip").unwrap_err();
        assert!(err.to_string().contains("docx"), "{err}");
    }

    #[test]
    fn ofd_extracts_pages_in_document_order() {
        let bytes = make_ofd(&[
            ("Pages/Page_1/Content.xml", "第二页&amp;续"),
            ("Pages/Page_0/Content.xml", "第一页"),
        ]);
        let text = extract_text(Path::new("a.ofd"), &bytes)
            .unwrap()
            .expect("ofd text");
        assert_eq!(text, "第二页&续\n第一页");
    }

    #[test]
    fn textless_ofd_is_skipped() {
        let bytes = make_ofd(&[("Pages/Page_0/Content.xml", "")]);
        assert!(extract_text(Path::new("a.ofd"), &bytes).unwrap().is_none());
    }

    #[test]
    fn corrupt_ofd_reports_an_error() {
        let err = extract_text(Path::new("a.ofd"), b"not a zip").unwrap_err();
        assert!(err.to_string().contains("OFD"), "{err}");
    }

    #[test]
    fn pdf_with_text_layer_is_extracted() {
        let bytes = make_pdf("BT /F1 24 Tf 72 720 Td (Hello PDF) Tj ET");
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("Hello PDF"), "{text}");
    }

    #[test]
    fn pdf_gbk_cmap_is_decoded_without_tounicode() {
        let bytes = make_type0_pdf("GBK-EUC-H", "C4E3BAC3");
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("你好"), "{text}");
    }

    #[test]
    fn pdf_ucs2_cmap_is_decoded_without_tounicode() {
        let bytes = make_type0_pdf("UniGB-UCS2-H", "4F60597D");
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("你好"), "{text}");
    }

    #[test]
    fn pdf_utf16_cmap_is_decoded_without_tounicode() {
        let bytes = make_type0_pdf("UniGB-UTF16-H", "4F60597D");
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("你好"), "{text}");
    }

    #[test]
    fn unknown_simple_font_encoding_falls_back_without_panic() {
        let bytes = make_pdf_with_font(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /GBK >>",
            "BT /F1 24 Tf 72 720 Td (Hello PDF) Tj ET",
        );
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("Hello PDF"), "{text}");
    }

    #[test]
    fn malformed_path_operators_do_not_lose_text() {
        let bytes = make_pdf(
            "BT /F1 12 Tf 72 720 Td (Before) Tj ET\nv\ny\nBT /F1 12 Tf 72 700 Td (After) Tj ET",
        );
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("Before"), "{text}");
        assert!(text.contains("After"), "{text}");
    }

    #[test]
    fn text_without_a_selected_font_is_skipped() {
        let bytes = make_pdf("BT (Before Tf) Tj ET");
        assert!(extract_text(Path::new("a.pdf"), &bytes).unwrap().is_none());
    }

    #[test]
    fn broken_pdf_page_does_not_discard_other_pages() {
        let bytes = make_pdf_pages(&[
            "BT /F1 24 Tf 72 720 Td (Good Page) Tj ET",
            "BI /W 10",
        ]);
        let text = extract_text(Path::new("a.pdf"), &bytes)
            .unwrap()
            .expect("pdf text");
        assert!(text.contains("Good Page"), "{text}");
    }

    #[test]
    fn pdf_with_only_broken_pages_reports_an_error() {
        let bytes = make_pdf_pages(&["BI /W 10"]);
        let err = extract_text(Path::new("a.pdf"), &bytes).unwrap_err();
        assert!(err.to_string().contains("PDF"), "{err}");
    }

    #[test]
    fn textless_pdf_is_skipped() {
        let bytes = make_pdf("q Q");
        assert!(extract_text(Path::new("a.pdf"), &bytes).unwrap().is_none());
    }

    #[test]
    fn plain_text_is_decoded_lossily() {
        let text = extract_text(Path::new("a.md"), "合并断行".as_bytes())
            .unwrap()
            .expect("text");
        assert_eq!(text, "合并断行");
    }
}
