use crate::engine::{FileHit, IndexPhase, IndexStatus, ModelStatus};
use crate::ocr::{self, OcrBackend, OcrStatus};
use crate::query::{self, ParsedQuery};
use crate::worker::{IndexMsg, UiMsg};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const MIN_WIDTH: u16 = 80;
const MIN_HEIGHT: u16 = 20;
const TOAST_TTL: Duration = Duration::from_secs(4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Input,
    Results,
}

#[derive(Debug)]
pub enum KeyOutcome {
    None,
    Submit(ParsedQuery),
    Copy(PathBuf),
    Quit,
}

pub struct App {
    input: Vec<char>,
    cursor: usize,
    focus: Focus,
    hits: Vec<FileHit>,
    selected: usize,
    detail: Option<DetailView>,
    status: IndexStatus,
    watching: bool,
    toast: Option<(String, Instant)>,
    query_id: u64,
    awaiting: Option<u64>,
    max_results: usize,
    ocr_status: OcrStatus,
    ocr_backend: OcrBackend,
    pub quit: bool,
}

#[derive(Debug, Clone, Copy)]
struct DetailView {
    hit: usize,
    scroll: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageInfo {
    current: usize,
    pages: usize,
}

impl PageInfo {
    fn label(self) -> String {
        format!("页 {}/{}", self.current, self.pages)
    }
}

/// 结果页按当前可见行数分页；`selected` 决定当前页。
fn page_info(app: &App, max_hits: usize, visible: usize) -> Option<PageInfo> {
    let total = app.hits.len().min(max_hits);
    if total == 0 {
        return None;
    }
    let visible = visible.max(1);
    let pages = total.div_ceil(visible);
    let selected = app.selected.min(total - 1);
    Some(PageInfo {
        current: selected / visible + 1,
        pages,
    })
}

impl App {
    pub fn new(watching: bool, initial_error: Option<String>) -> Self {
        Self {
            input: Vec::new(),
            cursor: 0,
            focus: Focus::Input,
            hits: Vec::new(),
            selected: 0,
            detail: None,
            status: IndexStatus {
                model: ModelStatus::Loading,
                rerank: ModelStatus::Disabled,
                indexed_files: 0,
                indexing: true,
                progress: None,
                last_error: None,
            },
            watching,
            toast: initial_error.map(|e| (e, Instant::now())),
            query_id: 0,
            awaiting: None,
            max_results: 10,
            ocr_status: ocr::status(),
            ocr_backend: ocr::backend(),
            quit: false,
        }
    }

    /// 由配置 `[search] results` 决定结果列表最多显示多少个文件。
    pub fn set_result_limit(&mut self, limit: usize) {
        self.max_results = limit.max(1);
    }

    pub fn input_text(&self) -> String {
        self.input.iter().collect()
    }

    fn set_toast(&mut self, message: impl Into<String>) {
        self.toast = Some((message.into(), Instant::now()));
    }

    pub fn apply_ui(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Status(status) => self.status = status,
            UiMsg::QueryResult { id, hits } => {
                if self.awaiting == Some(id) || self.awaiting.is_none() {
                    self.hits = hits;
                    self.selected = 0;
                    self.detail = None;
                    self.awaiting = None;
                }
            }
            UiMsg::Error(message) => self.set_toast(message),
        }
    }

    pub fn tick(&mut self) {
        if let Some((_, at)) = &self.toast {
            if at.elapsed() > TOAST_TTL {
                self.toast = None;
            }
        }
        self.ocr_status = ocr::status();
        self.ocr_backend = ocr::backend();
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> KeyOutcome {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return KeyOutcome::Quit;
        }
        if key.code == KeyCode::Esc && self.detail.is_none() {
            return KeyOutcome::Quit;
        }
        if self.detail.is_some() {
            return self.handle_detail_key(key);
        }
        match key.code {
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Results,
                    Focus::Results => Focus::Input,
                };
                return KeyOutcome::None;
            }
            KeyCode::Enter => {
                if self.focus == Focus::Results {
                    if self.hits.get(self.selected).is_some() {
                        self.detail = Some(DetailView {
                            hit: self.selected,
                            scroll: 0,
                        });
                    }
                    return KeyOutcome::None;
                }
                return self.submit();
            }
            KeyCode::Up => {
                self.select_prev();
                return KeyOutcome::None;
            }
            KeyCode::Down => {
                self.select_next();
                return KeyOutcome::None;
            }
            _ => {}
        }

        match self.focus {
            Focus::Input => self.handle_input_key(key),
            Focus::Results => self.handle_results_key(key),
        }
    }

    fn handle_detail_key(&mut self, key: KeyEvent) -> KeyOutcome {
        match key.code {
            KeyCode::Esc => {
                self.detail = None;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                if let Some(view) = &mut self.detail {
                    view.scroll = view.scroll.saturating_add(1);
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if let Some(view) = &mut self.detail {
                    view.scroll = view.scroll.saturating_sub(1);
                }
            }
            _ => {}
        }
        KeyOutcome::None
    }

    fn submit(&mut self) -> KeyOutcome {
        let parsed = query::parse(&self.input_text());
        if parsed.is_empty() {
            return KeyOutcome::None;
        }
        self.query_id += 1;
        self.awaiting = Some(self.query_id);
        KeyOutcome::Submit(parsed)
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> KeyOutcome {
        match key.code {
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                }
                KeyOutcome::None
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                }
                KeyOutcome::None
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                KeyOutcome::None
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
                KeyOutcome::None
            }
            KeyCode::Home => {
                self.cursor = 0;
                KeyOutcome::None
            }
            KeyCode::End => {
                self.cursor = self.input.len();
                KeyOutcome::None
            }
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.input.insert(self.cursor, ch);
                self.cursor += 1;
                KeyOutcome::None
            }
            _ => KeyOutcome::None,
        }
    }

    fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    fn select_next(&mut self) {
        if self.selected + 1 < self.hits.len() {
            self.selected += 1;
        }
    }

    fn handle_results_key(&mut self, key: KeyEvent) -> KeyOutcome {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return KeyOutcome::None;
        }
        match key.code {
            KeyCode::Char('c') => match self.hits.get(self.selected) {
                Some(hit) => KeyOutcome::Copy(hit.path.clone()),
                None => KeyOutcome::None,
            },
            KeyCode::Char('j') => {
                self.select_next();
                KeyOutcome::None
            }
            KeyCode::Char('k') => {
                self.select_prev();
                KeyOutcome::None
            }
            _ => KeyOutcome::None,
        }
    }
}

pub fn run(mut app: App, index_tx: Sender<IndexMsg>, ui_rx: Receiver<UiMsg>) -> Result<()> {
    let mut terminal = ratatui::try_init().map_err(|e| anyhow::anyhow!("初始化终端失败：{e}"))?;
    let result = run_loop(&mut terminal, &mut app, &index_tx, &ui_rx);
    ratatui::restore();
    result
}

fn run_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    index_tx: &Sender<IndexMsg>,
    ui_rx: &Receiver<UiMsg>,
) -> Result<()> {
    loop {
        while let Ok(msg) = ui_rx.try_recv() {
            app.apply_ui(msg);
        }
        app.tick();
        terminal.draw(|frame| draw(frame, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.handle_key(key) {
                        KeyOutcome::None => {}
                        KeyOutcome::Quit => {
                            app.quit = true;
                        }
                        KeyOutcome::Submit(parsed) => {
                            let (reply_tx, _reply_rx) = mpsc::channel();
                            if index_tx
                                .send(IndexMsg::Query {
                                    id: app.query_id,
                                    parsed,
                                    reply: reply_tx,
                                })
                                .is_err()
                            {
                                log::error!("索引线程已退出");
                                app.set_toast("索引线程已退出");
                            }
                        }
                        KeyOutcome::Copy(path) => {
                            if let Err(err) = copy_path(&path) {
                                log::warn!("{err}");
                                app.set_toast(err);
                            }
                        }
                    }
                }
            }
        }

        if app.quit {
            break;
        }
    }
    Ok(())
}

fn copy_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy().to_string();
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("剪贴板不可用：{e}"))?;
    clipboard
        .set_text(text)
        .map_err(|e| format!("复制路径失败：{e}"))?;
    drop(clipboard);
    Ok(())
}

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }
    let small = area.width < MIN_WIDTH || area.height < MIN_HEIGHT;
    let max_hits = if small {
        app.max_results.min(5)
    } else {
        app.max_results
    };

    let chunks = if small {
        Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area)
    } else {
        Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area)
    };

    draw_input(frame, app, chunks[0]);

    let results_area = if small { chunks[1] } else { chunks[2] };
    let results_height = if !small && app.focus == Focus::Results {
        results_area.height.saturating_sub(1)
    } else {
        results_area.height
    };
    let per_hit = if small { 1 } else { 2 };
    let visible = ((results_height as usize) / per_hit).max(1);
    let page = page_info(app, max_hits, visible);

    if !small && app.focus == Focus::Results {
        let parts =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(results_area);
        draw_results(frame, app, parts[0], max_hits, small, visible);
        draw_results_hint(frame, parts[1]);
    } else {
        draw_results(frame, app, results_area, max_hits, small, visible);
    }

    if small {
        let hint = Paragraph::new(Line::from(Span::styled(
            "终端过小（最小 80×20）",
            Style::default().fg(Color::Red),
        )));
        frame.render_widget(hint, chunks[2]);
    } else {
        draw_separator(frame, chunks[1]);
        draw_status(frame, app, chunks[3], page);
        draw_message(frame, app, chunks[4]);
    }

    if let Some(view) = &app.detail {
        if let Some(hit) = app.hits.get(view.hit) {
            draw_detail(frame, app, area, hit, *view);
        }
    }
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let text = app.input_text();
    let line = Line::from(vec![
        Span::styled(
            "查询: ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(text),
    ]);
    frame.render_widget(Paragraph::new(line), area);

    if app.focus == Focus::Input {
        let prompt_width = UnicodeWidthStr::width("查询: ");
        let prefix: String = app.input[..app.cursor].iter().collect();
        let x = area
            .x
            .saturating_add(prompt_width as u16)
            .saturating_add(UnicodeWidthStr::width(prefix.as_str()) as u16)
            .min(area.right().saturating_sub(1));
        frame.set_cursor_position(Position::new(x, area.y));
    }
}

fn draw_separator(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    frame.render_widget(block, area);
}

fn draw_results(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    max_hits: usize,
    small: bool,
    visible: usize,
) {
    let mut lines: Vec<Line> = Vec::new();
    if app.hits.is_empty() {
        lines.push(Line::from(Span::styled(
            "输入查询后按 Enter 检索",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let width = area.width.saturating_sub(2) as usize;
        let show_snippets = !small;
        let visible = visible.max(1);
        let total = app.hits.len().min(max_hits);
        let selected = app.selected.min(total.saturating_sub(1));
        let offset = if total > visible {
            (selected / visible) * visible
        } else {
            0
        };
        for (idx, hit) in app.hits[..total]
            .iter()
            .enumerate()
            .skip(offset)
            .take(visible)
        {
            let selected = app.focus == Focus::Results && idx == app.selected;
            let prefix = if selected { "> " } else { "  " };
            let text = fit_left(&hit.path.to_string_lossy(), width);
            let style = if selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(Color::Cyan)),
                Span::styled(text, style),
            ]));
            if show_snippets && !hit.snippet.is_empty() {
                let mut spans = vec![Span::raw("    ")];
                spans.extend(snippet_spans(
                    &hit.snippet,
                    area.width.saturating_sub(4) as usize,
                ));
                lines.push(Line::from(spans));
            }
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_results_hint(frame: &mut Frame, area: Rect) {
    let key = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let line = Line::from(vec![
        Span::styled("c", key),
        Span::raw(" 复制路径   "),
        Span::styled("j/k", key),
        Span::raw(" 选择   "),
        Span::styled("Enter", key),
        Span::raw(" 详情   "),
        Span::styled("Tab", key),
        Span::raw(" 返回查询"),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn draw_detail(frame: &mut Frame, app: &App, area: Rect, hit: &FileHit, view: DetailView) {
    let popup = centered_rect(area, 98, 90);
    frame.render_widget(Clear, popup);
    let name = hit
        .path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            format!(" 详情 · {name} "),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.height < 2 || inner.width == 0 {
        return;
    }

    let hint_area = Rect::new(inner.x, inner.bottom() - 1, inner.width, 1);
    let body = Rect::new(inner.x, inner.y, inner.width, inner.height - 1);
    let lines = wrap_detail_lines(detail_lines(app, hit), body.width as usize);
    let max_scroll = lines.len().saturating_sub(body.height as usize);
    let scroll = (view.scroll as usize).min(max_scroll) as u16;
    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((scroll, 0)),
        body,
    );

    let key = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let hint = Line::from(vec![
        Span::styled("Esc", key),
        Span::raw(" 关闭   "),
        Span::styled("j/k", key),
        Span::raw(" 滚动"),
    ]);
    frame.render_widget(
        Paragraph::new(hint).style(Style::default().fg(Color::DarkGray)),
        hint_area,
    );
}

fn detail_lines(app: &App, hit: &FileHit) -> Vec<Line<'static>> {
    let label = |text: &'static str| {
        Span::styled(
            text,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    };
    let score_label = match (hit.detail.lexical_rank, hit.detail.semantic_rank) {
        (Some(_), Some(_)) => "融合分: ",
        (Some(_), None) => "关键词分: ",
        (None, Some(_)) => "语义相似度: ",
        (None, None) => "",
    };
    let source = match (hit.detail.lexical_rank, hit.detail.semantic_rank) {
        (Some(lexical), Some(semantic)) => {
            format!("关键词第 {lexical} 位 + 语义第 {semantic} 位")
        }
        (Some(lexical), None) => format!("仅关键词（第 {lexical} 位）"),
        (None, Some(semantic)) => format!("仅语义（第 {semantic} 位）"),
        (None, None) => "空查询：按路径列出全部文件".to_string(),
    };
    let score = match (hit.detail.lexical_rank, hit.detail.semantic_rank) {
        (Some(_), Some(_)) => {
            format!("{:.4}（RRF，k={:.0}）", hit.detail.fused_score, hit.detail.rrf_k)
        }
        (Some(_), None) => format!("{:.4}（BM25）", hit.detail.fused_score),
        (None, Some(_)) => format!("{:.4}", hit.detail.fused_score),
        (None, None) => String::new(),
    };

    let mut lines = vec![
        Line::from(vec![
            label("路径: "),
            Span::raw(hit.path.to_string_lossy().to_string()),
        ]),
        Line::from(vec![label("来源: "), Span::raw(source)]),
    ];
    if !score_label.is_empty() {
        lines.push(Line::from(vec![label(score_label), Span::raw(score)]));
    }
    lines.push(Line::from(vec![
        label("重排: "),
        Span::raw(rerank_line(app, hit)),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(label("最佳匹配片段")));
    if hit.detail.spans.is_empty() {
        lines.push(Line::from(Span::styled(
            "（无可展示文本）",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.push(Line::from(
            hit.detail
                .spans
                .iter()
                .map(|(text, highlight)| {
                    let style = if *highlight {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    Span::styled(text.clone(), style)
                })
                .collect::<Vec<_>>(),
        ));
    }

    if !hit.related.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(label("其他相关片段")));
        for chunk in &hit.related {
            lines.push(Line::from(Span::styled(
                format!(
                    "片段 {} · 分数 {:.4}",
                    chunk.chunk_index + 1,
                    chunk.score
                ),
                Style::default().fg(Color::DarkGray),
            )));
            if chunk.spans.is_empty() {
                lines.push(Line::from(Span::styled(
                    "（无可展示文本）",
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                lines.push(Line::from(
                    chunk
                        .spans
                        .iter()
                        .map(|(text, highlight)| {
                            let style = if *highlight {
                                Style::default()
                                    .fg(Color::Yellow)
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default()
                            };
                            Span::styled(text.clone(), style)
                        })
                        .collect::<Vec<_>>(),
                ));
            }
        }
    }
    lines
}

fn rerank_line(app: &App, hit: &FileHit) -> String {
    if let Some(rerank) = &hit.detail.rerank {
        return format!(
            "{:.4}（权重 {:.2}，第 {} → 第 {} 位）",
            rerank.score, rerank.weight, rerank.before, rerank.after
        );
    }
    if hit.detail.rerank_applied {
        return "该结果未参与本次重排".to_string();
    }
    match &app.status.rerank {
        ModelStatus::Disabled => "未启用",
        ModelStatus::Loading => "加载中",
        ModelStatus::Unavailable { .. } => "未就绪",
        ModelStatus::Ready { .. } => "本次未执行",
    }
    .to_string()
}

fn wrap_detail_lines(lines: Vec<Line<'static>>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut out = Vec::new();
    for line in lines {
        let mut current: Vec<Span<'static>> = Vec::new();
        let mut used = 0usize;
        for span in line.spans {
            let style = span.style;
            let mut buf = String::new();
            for ch in span.content.chars() {
                let w = UnicodeWidthChar::width(ch).unwrap_or(0);
                if used + w > width && used > 0 {
                    if !buf.is_empty() {
                        current.push(Span::styled(std::mem::take(&mut buf), style));
                    }
                    out.push(Line::from(std::mem::take(&mut current)));
                    used = 0;
                }
                buf.push(ch);
                used += w;
            }
            if !buf.is_empty() {
                current.push(Span::styled(buf, style));
            }
        }
        out.push(Line::from(current));
    }
    out
}

fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let width = area.width.saturating_mul(percent_x) / 100;
    let height = area.height.saturating_mul(percent_y) / 100;
    Rect::new(
        area.x + (area.width.saturating_sub(width)) / 2,
        area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    )
}

fn snippet_spans(spans: &[(String, bool)], max_width: usize) -> Vec<Span<'static>> {
    let style = |highlight: bool| {
        if highlight {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        }
    };
    let total_width: usize = spans
        .iter()
        .map(|(text, _)| UnicodeWidthStr::width(text.as_str()))
        .sum();
    if total_width <= max_width {
        return spans
            .iter()
            .map(|(text, highlight)| Span::styled(text.clone(), style(*highlight)))
            .collect();
    }
    if max_width == 0 {
        return Vec::new();
    }

    let limit = max_width - 1;
    let mut used = 0usize;
    let mut out = Vec::new();
    for (text, highlight) in spans {
        let mut buf = String::new();
        for ch in text.chars() {
            let w = UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + w > limit {
                break;
            }
            buf.push(ch);
            used += w;
        }
        if !buf.is_empty() {
            out.push(Span::styled(buf, style(*highlight)));
        }
        if used >= limit {
            break;
        }
    }
    out.push(Span::styled("…", style(false)));
    out
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect, page: Option<PageInfo>) {
    let watching = if app.watching {
        "运行中"
    } else {
        "未运行"
    };
    let model = match &app.status.model {
        ModelStatus::Disabled => "已禁用",
        ModelStatus::Loading => "加载中",
        ModelStatus::Ready { .. } => "就绪",
        ModelStatus::Unavailable { .. } => "未就绪(仅关键词)",
    };
    let focus = match app.focus {
        Focus::Input => "查询",
        Focus::Results => "结果",
    };
    let indexing = match (app.status.indexing, app.status.progress) {
        (true, Some(progress)) => {
            let phase = match progress.phase {
                IndexPhase::Scan => "扫描",
                IndexPhase::Embed => "语义",
                IndexPhase::Rerank => "重排",
            };
            format!(" · {phase} {}%", progress.percent())
        }
        (true, None) => " (索引中)".to_string(),
        (false, _) => String::new(),
    };
    let mut base = format!(
        "监控: {watching} | 索引: {} 文件{indexing} | 模型: {model}",
        app.status.indexed_files
    );
    if app.ocr_status != OcrStatus::Disabled {
        let backend = app.ocr_backend.label();
        let suffix = if backend.is_empty()
            || !matches!(app.ocr_status, OcrStatus::Ready | OcrStatus::Busy)
        {
            String::new()
        } else {
            format!("·{backend}")
        };
        // 窄屏下省掉冒号后的空格，给 OCR 后端和页信息留一位。
        base.push_str(&format!(" | OCR:{}{suffix}", app.ocr_status.label()));
    }
    if app.status.rerank != ModelStatus::Disabled {
        let rerank = match &app.status.rerank {
            ModelStatus::Disabled => "已关闭",
            ModelStatus::Loading => "加载中",
            ModelStatus::Ready { .. } => "就绪",
            ModelStatus::Unavailable { .. } => "未就绪",
        };
        base.push_str(&format!(" | 重排: {rerank}"));
    }
    // 80 列窄屏且 OCR 已启用时，优先保留 OCR/翻页等新信息，焦点仍可从光标和结果区提示判断。
    if app.ocr_status == OcrStatus::Disabled || area.width >= 90 {
        base.push_str(&format!(" | 焦点: {focus}"));
    }
    if let Some(page) = page {
        base.push_str(&format!(" | {}", page.label()));
    }

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            base,
            Style::default().fg(Color::Gray),
        ))),
        area,
    );
}

fn draw_message(frame: &mut Frame, app: &App, area: Rect) {
    let message = app
        .toast
        .as_ref()
        .map(|(message, _)| message.clone())
        .or_else(|| app.status.last_error.clone());
    if let Some(message) = message {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                fit_left(&message, area.width as usize),
                Style::default().fg(Color::Red),
            ))),
            area,
        );
    }
}

fn fit_left(text: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    let mut out: Vec<char> = Vec::new();
    let mut width = 0;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > max_width.saturating_sub(1) {
            break;
        }
        width += w;
        out.push(ch);
    }
    let mut s: String = out.into_iter().collect();
    s.push('…');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{HitDetail, IndexProgress};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn app_with_hits(n: usize) -> App {
        let mut app = App::new(true, None);
        app.focus = Focus::Input;
        app.status = IndexStatus {
            model: ModelStatus::Ready { dims: 512 },
            rerank: ModelStatus::Ready { dims: 768 },
            indexed_files: 1000,
            indexing: false,
            progress: None,
            last_error: None,
        };
        app.hits = (0..n)
            .map(|i| FileHit {
                path: PathBuf::from(format!("D:\\docs\\ocr\\file{i}.md")),
                snippet: vec![
                    ("相关摘录：".to_string(), false),
                    ("合并".to_string(), true),
                    ("断行。".to_string(), false),
                ],
                detail: HitDetail {
                    spans: vec![
                        ("结算管理质效：统一台账，".to_string(), false),
                        ("按月核对".to_string(), true),
                        ("并归档。".to_string(), false),
                    ],
                    lexical_rank: Some(i + 1),
                    semantic_rank: Some(i + 2),
                    fused_score: 0.0321,
                    rerank: Some(crate::engine::RerankDetail {
                        score: 0.8123,
                        weight: 0.35,
                        before: i + 1,
                        after: i + 1,
                    }),
                    rerank_applied: true,
                    rrf_k: 60.0,
                },
                related: Vec::new(),
            })
            .collect();
        app
    }

    fn render(app: &App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            let mut x = 0;
            while x < buffer.area.width {
                let symbol = buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" ");
                out.push_str(symbol);
                let width = UnicodeWidthStr::width(symbol).max(1) as u16;
                x += width;
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn renders_normal_layout() {
        let app = app_with_hits(3);
        let text = render(&app, 80, 20);
        assert!(text.contains("查询:"), "{text}");
        assert!(text.contains("D:\\docs\\ocr\\file0.md"), "{text}");
        assert!(text.contains("监控: 运行中"), "{text}");
        assert!(text.contains("模型: 就绪"), "{text}");
        assert!(text.contains("焦点: 查询"), "{text}");
        assert!(text.contains("重排: 就绪"), "{text}");
        assert!(text.contains("索引: 1000 文件"), "{text}");
    }

    #[test]
    fn status_bar_shows_ocr_status_when_enabled() {
        let mut app = app_with_hits(3);
        app.ocr_status = OcrStatus::Ready;
        app.ocr_backend = OcrBackend::DirectMl;
        let text = render(&app, 140, 30);
        assert!(text.contains("OCR:就绪·GPU"), "{text}");

        // 80 列时优先保留 OCR 和页信息，焦点字段让位。
        let narrow = render(&app, 80, 20);
        assert!(narrow.contains("OCR:就绪·GPU"), "{narrow}");
        assert!(narrow.contains("页 1/1"), "{narrow}");

        app.ocr_status = OcrStatus::Disabled;
        app.ocr_backend = OcrBackend::Unknown;
        let text = render(&app, 140, 30);
        assert!(!text.contains("OCR:"), "{text}");
    }

    #[test]
    fn status_bar_shows_active_phase_percent() {
        let mut app = app_with_hits(1);
        app.status.indexing = true;
        app.status.progress = Some(IndexProgress {
            phase: IndexPhase::Scan,
            done: 2,
            total: 4,
        });
        let text = render(&app, 100, 20);
        assert!(text.contains("索引: 1000 文件 · 扫描 50%"), "{text}");
        assert!(!text.contains("(索引中)"), "{text}");
    }

    #[test]
    fn status_bar_uses_embed_and_rerank_labels() {
        let mut app = app_with_hits(1);
        app.status.indexing = true;
        app.status.progress = Some(IndexProgress {
            phase: IndexPhase::Embed,
            done: 1,
            total: 3,
        });
        assert!(render(&app, 100, 20).contains("· 语义 33%"));
        app.status.progress = Some(IndexProgress {
            phase: IndexPhase::Rerank,
            done: 4,
            total: 5,
        });
        assert!(render(&app, 100, 20).contains("· 重排 80%"));
    }

    #[test]
    fn status_bar_falls_back_to_indexing_without_progress() {
        let mut app = app_with_hits(1);
        app.status.indexing = true;
        app.status.progress = None;
        let text = render(&app, 100, 20);
        assert!(text.contains("(索引中)"), "{text}");
    }

    #[test]
    fn idle_status_hides_progress() {
        let mut app = app_with_hits(1);
        app.status.progress = Some(IndexProgress {
            phase: IndexPhase::Embed,
            done: 1,
            total: 2,
        });
        let text = render(&app, 100, 20);
        assert!(!text.contains("语义"), "{text}");
        assert!(!text.contains("(索引中)"), "{text}");
    }

    #[test]
    fn rerank_status_hidden_when_disabled() {
        let mut app = app_with_hits(1);
        app.status.rerank = ModelStatus::Disabled;
        let text = render(&app, 80, 20);
        assert!(!text.contains("重排:"), "{text}");
    }

    #[test]
    fn rerank_failure_is_visible_in_status_bar() {
        let mut app = app_with_hits(1);
        app.status.rerank = ModelStatus::Unavailable {
            reason: "模型文件缺失".to_string(),
        };
        let text = render(&app, 80, 20);
        assert!(text.contains("重排: 未就绪"), "{text}");
    }

    #[test]
    fn results_focus_shows_copy_hint() {
        let mut app = app_with_hits(2);
        let input_focus = render(&app, 80, 20);
        assert!(!input_focus.contains("复制路径"), "{input_focus}");
        app.focus = Focus::Results;
        let results_focus = render(&app, 80, 20);
        assert!(results_focus.contains("复制路径"), "{results_focus}");
        assert!(results_focus.contains("j/k"), "{results_focus}");
        assert!(results_focus.contains("Enter"), "{results_focus}");
        let degraded = render(&app, 79, 19);
        assert!(!degraded.contains("复制路径"), "{degraded}");
    }

    #[test]
    fn j_and_k_move_selection_only_in_results_focus() {
        let mut app = app_with_hits(3);
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert_eq!(app.selected, 0);
        assert_eq!(app.input_text(), "j");
        app.input.clear();
        app.cursor = 0;
        app.focus = Focus::Results;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert_eq!(app.selected, 1);
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert_eq!(app.selected, 2);
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert_eq!(app.selected, 1);
    }

    #[test]
    fn degraded_layout_hides_status_and_limits_five() {
        let app = app_with_hits(8);
        let text = render(&app, 79, 19);
        assert!(text.contains("终端过小"), "{text}");
        assert!(!text.contains("监控:"), "{text}");
        let shown = text.matches("file").count();
        assert_eq!(shown, 5, "{text}");
    }

    #[test]
    fn configured_result_limit_caps_visible_hits() {
        let mut app = app_with_hits(8);
        app.set_result_limit(3);
        let text = render(&app, 80, 20);
        assert_eq!(text.matches("file").count(), 3, "{text}");
    }

    #[test]
    fn results_show_snippet_under_path() {
        let app = app_with_hits(2);
        let text = render(&app, 80, 20);
        assert!(text.contains("相关摘录：合并断行。"), "{text}");
    }

    #[test]
    fn degraded_layout_omits_snippets() {
        let app = app_with_hits(2);
        let text = render(&app, 79, 19);
        assert!(!text.contains("相关摘录"), "{text}");
    }

    #[test]
    fn status_bar_shows_current_and_total_pages() {
        let mut app = app_with_hits(10);
        let text = render(&app, 80, 20);
        assert!(text.contains("页 1/2"), "{text}");

        app.focus = Focus::Results;
        app.selected = 8;
        let text = render(&app, 80, 20);
        assert!(text.contains("页 2/2"), "{text}");

        let empty = app_with_hits(0);
        assert!(!render(&empty, 80, 20).contains("页 "), "empty results should not show a page");
    }

    #[test]
    fn selection_scrolls_results_pages() {
        let mut app = app_with_hits(10);
        app.focus = Focus::Results;
        app.selected = 8;
        let text = render(&app, 80, 20);
        assert!(text.contains("file8.md"), "{text}");
        assert!(text.contains("file9.md"), "{text}");
        assert!(!text.contains("file0.md"), "{text}");
    }

    #[test]
    fn snippet_match_is_yellow() {
        let app = app_with_hits(1);
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut found = false;
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                if let Some(cell) = buffer.cell((x, y)) {
                    if cell.symbol() == "合" {
                        assert_eq!(cell.style().fg, Some(Color::Yellow));
                        found = true;
                    }
                }
            }
        }
        assert!(found, "highlighted snippet not rendered");
    }

    #[test]
    fn selection_only_highlighted_with_results_focus() {
        let mut app = app_with_hits(3);
        app.selected = 1;
        let input_focus = render(&app, 80, 20);
        assert!(
            !input_focus.contains("> D:\\docs\\ocr\\file1.md"),
            "{input_focus}"
        );
        app.focus = Focus::Results;
        let results_focus = render(&app, 80, 20);
        assert!(
            results_focus.contains("> D:\\docs\\ocr\\file1.md"),
            "{results_focus}"
        );
    }

    #[test]
    fn q_types_and_esc_quits_while_input_focused() {
        let mut app = app_with_hits(1);
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE)),
            KeyOutcome::None
        ));
        assert_eq!(app.input_text(), "q");
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            KeyOutcome::Quit
        ));
    }

    #[test]
    fn c_copies_only_from_results_focus() {
        let mut app = app_with_hits(2);
        app.focus = Focus::Results;
        app.selected = 1;
        match app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE)) {
            KeyOutcome::Copy(path) => assert!(path.ends_with("file1.md")),
            _ => panic!("expected copy"),
        }
    }

    #[test]
    fn tab_toggles_focus_and_enter_submits() {
        let mut app = app_with_hits(1);
        app.input = "合并 断行 dir:docs".chars().collect();
        app.cursor = app.input.len();
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            KeyOutcome::None
        ));
        assert_eq!(app.focus, Focus::Results);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.focus, Focus::Input);
        match app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)) {
            KeyOutcome::Submit(parsed) => {
                assert_eq!(parsed.text, "合并 断行");
                assert_eq!(parsed.dir_filter.as_deref(), Some("docs"));
            }
            _ => panic!("expected submit"),
        }
    }

    #[test]
    fn arrows_move_and_enter_opens_detail_from_results_focus() {
        let mut app = app_with_hits(3);
        app.focus = Focus::Results;
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.selected, 1);
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.selected, 1);
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            KeyOutcome::None
        ));
        assert_eq!(app.detail.map(|view| view.hit), Some(1));
    }

    #[test]
    fn detail_overlay_shows_ranking_and_best_chunk() {
        let mut app = app_with_hits(3);
        app.focus = Focus::Results;
        app.selected = 1;
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let text = render(&app, 100, 30);
        assert!(text.contains("详情 · file1.md"), "{text}");
        assert!(text.contains("关键词第 2 位 + 语义第 3 位"), "{text}");
        assert!(text.contains("0.0321（RRF，k=60）"), "{text}");
        assert!(text.contains("0.8123（权重 0.35，第 2 → 第 2 位）"), "{text}");
        assert!(text.contains("最佳匹配片段"), "{text}");
        assert!(text.contains("结算管理质效：统一台账，按月核对并归档。"), "{text}");
        assert!(text.contains("Esc 关闭"), "{text}");
    }

    #[test]
    fn detail_shows_other_related_chunks() {
        let mut app = app_with_hits(1);
        app.hits[0].related.push(crate::engine::RelatedChunk {
            chunk_index: 2,
            score: 0.0201,
            spans: vec![("另一个相关段落".to_string(), false)],
        });
        app.focus = Focus::Results;
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let text = render(&app, 100, 30);
        assert!(text.contains("其他相关片段"), "{text}");
        assert!(text.contains("片段 3"), "{text}");
        assert!(text.contains("另一个相关段落"), "{text}");
    }

    #[test]
    fn esc_closes_detail_before_quitting() {
        let mut app = app_with_hits(2);
        app.focus = Focus::Results;
        app.input = "查询".chars().collect();
        app.cursor = app.input.len();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.detail.is_some());
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            KeyOutcome::None
        ));
        assert!(app.detail.is_none());
        assert_eq!(app.input_text(), "查询");
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            KeyOutcome::Quit
        ));
    }

    #[test]
    fn detail_scrolls_with_j_and_k_and_stays_open() {
        let mut app = app_with_hits(1);
        app.focus = Focus::Results;
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert_eq!(app.detail.map(|view| view.scroll), Some(2));
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert_eq!(app.detail.map(|view| view.scroll), Some(1));
        assert!(app.detail.is_some());
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn detail_overlay_works_in_degraded_layout() {
        let mut app = app_with_hits(1);
        app.focus = Focus::Results;
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let text = render(&app, 79, 19);
        assert!(text.contains("详情 · file0.md"), "{text}");
        assert!(text.contains("最佳匹配片段"), "{text}");
        assert!(text.contains("Esc 关闭"), "{text}");
    }

    #[test]
    fn detail_explains_when_rerank_did_not_run() {
        let mut app = app_with_hits(1);
        app.focus = Focus::Results;
        app.hits[0].detail.rerank = None;
        app.hits[0].detail.rerank_applied = false;
        app.status.rerank = ModelStatus::Ready { dims: 768 };
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let text = render(&app, 100, 30);
        assert!(text.contains("重排: 本次未执行"), "{text}");

        app.status.rerank = ModelStatus::Disabled;
        let text = render(&app, 100, 30);
        assert!(text.contains("重排: 未启用"), "{text}");
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = app_with_hits(0);
        assert!(matches!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            KeyOutcome::Quit
        ));
    }

    #[test]
    fn error_is_visible_in_status_bar() {
        let mut app = app_with_hits(0);
        app.toast = Some(("剪贴板不可用".to_string(), Instant::now()));
        let text = render(&app, 100, 20);
        assert!(text.contains("剪贴板不可用"), "{text}");
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            lines.last().is_some_and(|line| line.contains("剪贴板不可用")),
            "{text}"
        );
    }
}
