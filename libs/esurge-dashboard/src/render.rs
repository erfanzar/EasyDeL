//! Server-rendered HTML shell, styled after px0.ai's light workbench.
//!
//! The charts are drawn in the browser by `assets/dashboard.js`, which polls `/api/history`. The markup carries no
//! inline script or style, so it runs under the strict Content-Security-Policy set in `routes`. Every string that
//! comes from the API server is escaped.

use std::fmt::Write as _;

use crate::metrics::{RecentRequest, ServerMetrics};

/// Stylesheet served at `/assets/dashboard.css`.
pub const CSS: &str = include_str!("../assets/dashboard.css");
/// Client script served at `/assets/dashboard.js`.
pub const JS: &str = include_str!("../assets/dashboard.js");

/// Characters of a request id shown in the log; the full id is in the cell's `title`.
const REQUEST_ID_TAIL: usize = 12;

const FONTS: &str = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap";

const LOGO: &str = "e<b>Surge</b><span class=\"cur\"></span><em>// workbench</em>";

/// Escape text for HTML element content and double-quoted attributes.
pub fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// Number with precision that shrinks as magnitude grows; non-finite values render as `—`.
pub fn number(value: f64) -> String {
    if !value.is_finite() {
        return "—".into();
    }
    let magnitude = value.abs();
    if magnitude >= 1000.0 {
        format!("{value:.0}")
    } else if magnitude >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

pub fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), number)
}

pub fn seconds(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.2}s"),
        _ => "—".into(),
    }
}

pub fn percent(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{:.1}%", value * 100.0),
        _ => "—".into(),
    }
}

pub fn duration(total_seconds: f64) -> String {
    let total = if total_seconds.is_finite() && total_seconds > 0.0 {
        total_seconds as u64
    } else {
        0
    };
    format!("{:02}h {:02}m {:02}s", total / 3600, (total % 3600) / 60, total % 60)
}

fn page_start(html: &mut String, title: &str, refresh_seconds: Option<u64>) {
    html.push_str("<!doctype html><html lang=\"en\" data-theme=\"light\"><head><meta charset=\"utf-8\">");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    html.push_str("<meta name=\"theme-color\" content=\"#f4f5f8\"><meta name=\"robots\" content=\"noindex\">");
    if let Some(seconds) = refresh_seconds {
        let _ = write!(html, "<meta http-equiv=\"refresh\" content=\"{seconds}\">");
    }
    let _ = write!(
        html,
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\
         <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\
         <link rel=\"stylesheet\" href=\"{}\"><link rel=\"stylesheet\" href=\"/assets/dashboard.css\">\
         <title>{}</title></head><body>\
         <div class=\"atmos\" aria-hidden=\"true\"><div class=\"atmos-glow\"></div><div class=\"atmos-grid\"></div></div>",
        escape(FONTS),
        escape(title),
    );
}

fn section_start(html: &mut String, id: &str, n: &str, title: &str, sub: &str) {
    let _ = write!(
        html,
        "<section class=\"sec\" id=\"{id}\"><div class=\"wrap\"><div class=\"sec-head\"><div class=\"sec-id\">\
         <span class=\"sec-n\">{n}</span><span>{id}</span></div><div><h2 class=\"sec-title\">{title}</h2>\
         <p class=\"sec-sub\">{sub}</p></div></div>"
    );
}

fn section_end(html: &mut String) {
    html.push_str("</div></section>");
}

fn chart(html: &mut String, id: &str, title: &str, wide: bool, note: &str) {
    let class = if wide { "pnl ticks wide" } else { "pnl" };
    let _ = write!(
        html,
        "<div class=\"{class}\"><div class=\"pnl-h\"><span>{title}</span><b data-k=\"{id}-v\">—</b></div>\
         <div class=\"pnl-b\"><div data-chart=\"{id}\"></div>"
    );
    if !note.is_empty() {
        let _ = write!(html, "<p class=\"note\">{note}</p>");
    }
    html.push_str("</div></div>");
}

fn meters(html: &mut String, id: &str, title: &str) {
    let _ = write!(
        html,
        "<div class=\"pnl\"><div class=\"pnl-h\"><span>{title}</span><b data-k=\"{id}-v\"></b></div>\
         <div class=\"pnl-b\"><div class=\"meters\" data-meters=\"{id}\"></div></div></div>"
    );
}

fn stat(html: &mut String, key: &str, label: &str, value: &str, highlight: bool) {
    let class = if highlight { "stat hi" } else { "stat" };
    let _ = write!(
        html,
        "<div class=\"{class}\"><span class=\"stat-k\">{label}</span><span class=\"stat-v\" data-k=\"{key}\">{value}</span>\
         <span class=\"stat-d\" data-k=\"{key}-d\">&nbsp;</span><div class=\"spark\" data-spark=\"{key}\"></div></div>"
    );
}

fn big(html: &mut String, key: &str, label: &str, value: &str, highlight: bool) {
    let class = if highlight { "big hi" } else { "big" };
    let _ = write!(
        html,
        "<div class=\"{class}\"><span class=\"big-k\">{label}</span><span class=\"big-v\" data-k=\"{key}\">{value}</span>\
         <span class=\"big-d\" data-k=\"{key}-d\">&nbsp;</span></div>"
    );
}

/// The dashboard page. `metrics` fills the first paint; the script takes over from there.
pub fn dashboard(metrics: &ServerMetrics, refresh_seconds: u64) -> String {
    let mut html = String::with_capacity(32 * 1024);
    page_start(&mut html, "eSurge // workbench", None);
    let status = if metrics.status.is_empty() {
        "unknown"
    } else {
        metrics.status.as_str()
    };
    let dot = if status == "ready" { "ok" } else { "warn" };
    let first = metrics.engine_metrics.iter().next();
    let model = first.map_or("—", |(name, _)| name.as_str());
    let engine = first.map(|(_, engine)| engine);
    let aggregate = engine.map(|engine| &engine.aggregate);
    let uptime = duration(metrics.uptime_seconds);

    let _ = write!(
        html,
        "<nav class=\"nav\"><div class=\"wrap nav-inner\"><a class=\"logo\" href=\"/\">{LOGO}</a>\
         <div class=\"nav-mid\"><a href=\"#throughput\">throughput</a><a href=\"#load\">load</a>\
         <a href=\"#latency\">latency</a><a href=\"#cache\">cache</a><a href=\"#requests\">requests</a>\
         <a href=\"#access\">access</a></div><div class=\"nav-right\">\
         <span class=\"pill\"><span class=\"dot {dot}\" data-role=\"dot\"></span><span data-k=\"conn\">{status}</span></span>\
         <span class=\"pill\">up&nbsp;<span data-k=\"uptime\">{uptime}</span></span>\
         <form method=\"post\" action=\"/logout\"><button class=\"btn btn-ghost\" type=\"submit\">log out</button></form>\
         </div></div></nav>",
        status = escape(status),
    );

    let _ = write!(
        html,
        "<main id=\"app\" data-refresh=\"{refresh_seconds}\"><div class=\"wrap hero\"><div class=\"hero-meta\">\
         <span>model <b data-k=\"model\">{model}</b></span><span>status <b data-k=\"status\">{status}</b></span>\
         <span>uptime <b data-k=\"uptime\">{uptime}</b></span><span>window <b data-k=\"window\">{window}</b></span>\
         <span>models <b data-k=\"models\">{models}</b></span><span>updated <b data-k=\"updated\">—</b></span></div>\
         <div class=\"hero-grid\"><div class=\"hero-copy\"><div class=\"kicker\"><span>$</span> esurge serve · \
         openai-compatible</div><h1>Inference <span class=\"hl\">cockpit</span><span class=\"cur\"></span></h1>\
         <p class=\"hero-sub\">Live throughput, load, latency and prefix-cache telemetry for <code>{model}</code>, \
         read from the API server's <code>/metrics</code> with your key every {refresh_seconds}s.</p>\
         <div class=\"toolbar\"><span class=\"lbl\">range</span>",
        model = escape(model),
        status = escape(status),
        window = engine.map_or_else(
            || "—".to_string(),
            |engine| format!("{}s", number(engine.window_seconds))
        ),
        models = metrics.engine_metrics.len(),
    );
    for range in ["5m", "15m", "1h", "6h", "24h", "all"] {
        let _ = write!(
            html,
            "<button class=\"kbd\" type=\"button\" data-range=\"{range}\">{range}</button>"
        );
    }
    html.push_str(
        "<button class=\"kbd\" type=\"button\" data-action=\"pause\">pause</button>\
         <select data-role=\"model\" aria-label=\"model\" hidden></select></div></div><div class=\"hero-stats\">",
    );
    big(&mut html, "out-big", "output tok/s", "—", true);
    big(
        &mut html,
        "run-big",
        "running",
        &engine.map_or_else(|| "—".to_string(), |engine| engine.running_requests.to_string()),
        false,
    );
    big(
        &mut html,
        "ttft-big",
        "ttft p50",
        &seconds(aggregate.and_then(|agg| agg.ttft_p50_s)),
        false,
    );
    big(
        &mut html,
        "hit-big",
        "cache hit",
        &percent(aggregate.and_then(|agg| agg.token_cache_hit_rate)),
        true,
    );
    html.push_str("</div></div><div class=\"stats\">");
    stat(&mut html, "out", "output", "—", true);
    stat(&mut html, "run", "running", "—", false);
    stat(&mut html, "wait", "waiting", "—", false);
    stat(&mut html, "dec", "decode", "—", false);
    stat(&mut html, "ttft", "ttft p50", "—", false);
    stat(&mut html, "lat", "latency p50", "—", false);
    stat(&mut html, "hit", "cache hit", "—", true);
    stat(&mut html, "req", "requests", &metrics.total_requests.to_string(), false);
    html.push_str("</div></div>");

    section_start(
        &mut html,
        "throughput",
        "01",
        "Tokens <em>out</em>",
        "Generated tokens per second from the server's counter (30 s trailing rate), the engine's output rate over its \
         5-minute window, per-request decode speed, prefill speed and request rate.",
    );
    html.push_str("<div class=\"grid\">");
    chart(&mut html, "gen", "generation throughput", true, "");
    chart(
        &mut html,
        "prefill",
        "prefill tok/s · 5 min",
        false,
        "Uncached prompt tokens over time to first token. TTFT includes queueing, so this is a lower bound.",
    );
    chart(&mut html, "reqrate", "request rate", false, "");
    html.push_str("</div>");
    section_end(&mut html);

    section_start(
        &mut html,
        "load",
        "02",
        "Engine <em>load</em>",
        "Requests running in the engine, waiting in the scheduler and open at the HTTP layer, plus the shape of the \
         work: prompt against output length, and how decode speed spreads across requests.",
    );
    html.push_str("<div class=\"grid\">");
    chart(&mut html, "conc", "concurrency", true, "");
    chart(&mut html, "lengths", "prompt vs output tokens", false, "");
    chart(&mut html, "dist_decode", "decode tok/s distribution", false, "");
    html.push_str("</div>");
    section_end(&mut html);

    section_start(
        &mut html,
        "latency",
        "03",
        "Time to <em>first token</em>",
        "p50 and p99 over the engine's 5-minute window (log scale), with every completed request plotted against its \
         prompt and output length.",
    );
    html.push_str("<div class=\"grid\">");
    chart(&mut html, "ttft", "ttft p50 / p99", false, "");
    chart(&mut html, "lat", "latency p50 / p99", false, "");
    chart(&mut html, "ttft_scatter", "ttft vs prompt tokens", false, "");
    chart(&mut html, "lat_scatter", "latency vs output tokens", false, "");
    chart(&mut html, "dist_ttft", "ttft distribution", true, "");
    html.push_str("</div>");
    section_end(&mut html);

    section_start(
        &mut html,
        "cache",
        "04",
        "Prefix <em>cache</em>",
        "How much of each prompt the prefix cache served. Pages are 128 tokens, so a prompt shorter than one page can \
         never hit, and a hit covers whole pages only.",
    );
    html.push_str("<div class=\"grid\">");
    chart(&mut html, "hit", "hit rate · 5 min", false, "");
    meters(&mut html, "cache", "cache summary");
    chart(
        &mut html,
        "cache_bars",
        "per-request cached vs prefilled tokens",
        true,
        "",
    );
    html.push_str("</div>");
    section_end(&mut html);

    section_start(
        &mut html,
        "requests",
        "05",
        "Request <em>log</em>",
        "Every completed request the dashboard has seen since it started, newest first. Requests that finish between \
         two polls are still logged as long as they stay in the engine's recent list.",
    );
    html.push_str("<div class=\"grid\">");
    chart(&mut html, "dist_prompt", "prompt tokens distribution", false, "");
    chart(&mut html, "dist_output", "output tokens distribution", false, "");
    meters(&mut html, "finish", "finish reasons");
    meters(&mut html, "outcomes", "http outcomes since start");
    html.push_str(
        "<div class=\"pnl wide\"><div class=\"pnl-h\"><span>request log</span><b data-k=\"log-v\">—</b></div>\
         <div class=\"tbl-wrap\"><table class=\"tbl\"><thead><tr><th>seen</th><th>request</th><th>finish</th>\
         <th class=\"num\">prompt</th><th class=\"num\">cached</th><th class=\"num\">prefilled</th>\
         <th class=\"num\">output</th><th class=\"num\">ttft</th><th class=\"num\">latency</th>\
         <th class=\"num\">decode tok/s</th><th>cache</th></tr></thead><tbody data-table=\"log\">",
    );
    match engine {
        Some(engine) if !engine.recent_requests.is_empty() => {
            for request in &engine.recent_requests {
                request_row(&mut html, request);
            }
        }
        _ => html.push_str("<tr><td colspan=\"11\" class=\"muted\">no completed requests in the window</td></tr>"),
    }
    html.push_str("</tbody></table></div></div></div>");
    section_end(&mut html);

    section_start(
        &mut html,
        "access",
        "06",
        "API <em>keys</em>",
        "Usage across every key the API server knows. The dashboard holds no key of its own: each poll is checked \
         upstream with the viewer's key.",
    );
    html.push_str("<div class=\"cells\" data-cells=\"keys\"></div><div class=\"grid\">");
    chart(&mut html, "keyrate", "tokens per second across keys", true, "");
    html.push_str("</div>");
    section_end(&mut html);

    let _ = write!(
        html,
        "</main><footer><div class=\"wrap\">esurge-dashboard <b>v{}</b> · reads <code>GET /metrics</code> with the \
         viewer's key · chart history lives in memory and grows while a viewer is open · refresh {refresh_seconds}s\
         </div></footer><noscript><div class=\"wrap\"><div class=\"error\">Charts need JavaScript. The request log \
         above is a server-rendered snapshot; reload to update it.</div></div></noscript>\
         <script src=\"/assets/dashboard.js\" defer></script></body></html>",
        env!("CARGO_PKG_VERSION"),
    );
    html
}

fn request_row(html: &mut String, request: &RecentRequest) {
    let id = request.request_id.as_str();
    let start = id
        .char_indices()
        .rev()
        .nth(REQUEST_ID_TAIL - 1)
        .map_or(0, |(index, _)| index);
    let short = if start > 0 {
        format!("…{}", &id[start..])
    } else {
        id.to_string()
    };
    let finish = match (&request.error, request.finish_reason.as_deref()) {
        (Some(error), _) => format!(
            "<span class=\"mark mark--danger\" title=\"{}\">error</span>",
            escape(error)
        ),
        (None, Some("length")) => "<span class=\"mark mark--write\">length</span>".to_string(),
        (None, reason) => format!(
            "<span class=\"mark mark--read\">{}</span>",
            escape(reason.unwrap_or("—"))
        ),
    };
    let (cache_class, cache_label) = if request.cache_hit {
        ("hit", "HIT")
    } else {
        ("miss", "miss")
    };
    let _ = write!(
        html,
        "<tr><td>—</td><td class=\"id\" title=\"{}\">{}</td><td>{finish}</td><td class=\"num\">{}</td>\
         <td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
         <td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"{cache_class}\">{cache_label}</td></tr>",
        escape(id),
        escape(&short),
        request.prompt_tokens,
        request.cache_read_tokens,
        request.uncached_input_tokens,
        request.completion_tokens,
        seconds(request.ttft_s),
        seconds(request.latency_s),
        optional_number(request.decode_tps),
    );
}

/// Login form, with an optional error line.
pub fn login(error: Option<&str>) -> String {
    let mut html = String::new();
    page_start(&mut html, "eSurge // sign in", None);
    let _ = write!(
        html,
        "<div class=\"center\"><form class=\"pnl ticks login\" method=\"post\" action=\"/login\">\
         <div class=\"pnl-h\"><span>sign in</span><b>esurge-dashboard</b></div><div class=\"pnl-b\">\
         <div class=\"logo\">{LOGO}</div><p class=\"muted\">Sign in with an API key for this server. The key is \
         checked against the API server and kept in memory; the cookie carries only a random session id.</p>"
    );
    if let Some(error) = error {
        let _ = write!(html, "<div class=\"error\">{}</div>", escape(error));
    }
    html.push_str(
        "<input type=\"password\" name=\"api_key\" autocomplete=\"current-password\" placeholder=\"sk-…\" \
         aria-label=\"API key\" required autofocus><button class=\"btn btn-primary\" type=\"submit\">Sign in</button>\
         </div></form></div></body></html>",
    );
    html
}

/// Page shown when the API server cannot be read. It reloads itself every 15 seconds.
pub fn upstream_error(message: &str) -> String {
    let mut html = String::new();
    page_start(&mut html, "eSurge // unavailable", Some(15));
    let _ = write!(
        html,
        "<div class=\"center\"><div class=\"pnl ticks login\"><div class=\"pnl-h\"><span>upstream</span>\
         <b>retrying every 15s</b></div><div class=\"pnl-b\"><div class=\"logo\">{LOGO}</div>\
         <h2 class=\"sec-title\">API server unavailable</h2><div class=\"error\">{}</div></div></div></div>\
         </body></html>",
        escape(message)
    );
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ServerMetrics {
        serde_json::from_str(
            r#"{"status":"ready","uptime_seconds":3725,"total_requests":3,"engine_metrics":{"<script>m</script>":{
              "running_requests":2,"window_seconds":300,"aggregate":{"prefill_tps":2100.0,"decode_tps":47.1,
              "token_cache_hit_rate":0.5,"ttft_p50_s":null,"ttft_p99_s":2.1},
              "recent_requests":[{"request_id":"chat-<b>0123456789abcdef0123","finish_reason":"stop",
              "prompt_tokens":332,"completion_tokens":31,"ttft_s":1.88,"cache_hit":true,"cache_read_tokens":256,
              "uncached_input_tokens":76},{"request_id":"short","cache_hit":false,"error":"<boom>"}]}},
              "auth_stats":{"active_keys":1,"total_keys":1}}"#,
        )
        .unwrap()
    }

    #[test]
    fn dashboard_has_every_chart_and_the_first_paint() {
        let html = dashboard(&sample(), 5);
        for id in [
            "gen",
            "prefill",
            "reqrate",
            "conc",
            "lengths",
            "dist_decode",
            "ttft",
            "lat",
            "ttft_scatter",
            "lat_scatter",
            "dist_ttft",
            "hit",
            "cache_bars",
            "dist_prompt",
            "dist_output",
            "keyrate",
        ] {
            assert!(html.contains(&format!("data-chart=\"{id}\"")), "missing chart {id}");
        }
        assert!(html.contains("data-refresh=\"5\""));
        assert!(html.contains("01h 02m 05s"));
        assert!(html.contains("50.0%"));
        assert!(html.contains("<td class=\"num\">256</td><td class=\"num\">76</td>"));
        assert!(html.contains("<td class=\"hit\">HIT</td>"));
        assert!(html.contains("<td class=\"miss\">miss</td>"));
    }

    #[test]
    fn markup_has_no_inline_script_or_style() {
        for html in [dashboard(&sample(), 5), login(Some("x")), upstream_error("x")] {
            assert!(!html.contains("<script>"), "inline script");
            assert!(!html.contains(" style="), "inline style attribute");
            assert!(!html.contains("<style"), "inline style element");
            assert!(!html.contains("onclick"));
        }
        assert!(dashboard(&sample(), 5).contains("<script src=\"/assets/dashboard.js\" defer></script>"));
    }

    #[test]
    fn upstream_strings_are_escaped() {
        let html = dashboard(&sample(), 5);
        assert!(!html.contains("<script>m</script>"));
        assert!(html.contains("&lt;script&gt;m&lt;/script&gt;"));
        assert!(!html.contains("<b>0123"));
        assert!(html.contains("title=\"chat-&lt;b&gt;0123456789abcdef0123\""));
        assert!(html.contains("title=\"&lt;boom&gt;\">error</span>"));
        assert!(!upstream_error("<img src=x>").contains("<img"));
        assert!(!login(Some("<x>")).contains("<x>"));
    }

    #[test]
    fn null_values_render_as_dash() {
        let html = dashboard(&sample(), 5);
        assert!(html.contains("data-k=\"ttft-big\">—</span>"));
        assert_eq!(optional_number(None), "—");
        assert_eq!(seconds(Some(f64::NAN)), "—");
        assert_eq!(percent(None), "—");
    }

    #[test]
    fn number_formatting() {
        assert_eq!(number(2100.4), "2100");
        assert_eq!(number(47.12), "47.1");
        assert_eq!(number(1.234), "1.23");
        assert_eq!(number(-12.34), "-12.3");
        assert_eq!(number(f64::INFINITY), "—");
        assert_eq!(duration(-1.0), "00h 00m 00s");
    }

    #[test]
    fn long_ids_are_truncated_on_char_boundaries() {
        let mut metrics = sample();
        let engine = metrics.engine_metrics.values_mut().next().unwrap();
        engine.recent_requests[0].request_id = "ééééééééééééééééééééééééé".into();
        let html = dashboard(&metrics, 5);
        assert!(html.contains("…éééééééééééé<"));
    }

    #[test]
    fn assets_are_embedded() {
        assert!(CSS.contains("--bg: #f4f5f8"));
        assert!(JS.contains("/api/history"));
        assert!(!JS.contains("innerHTML"));
    }
}
