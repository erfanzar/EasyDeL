//! Server-rendered HTML. Every string that comes from the API server is escaped.

use std::fmt::Write as _;

use crate::metrics::{Aggregate, AuthStats, EngineMetrics, RecentRequest, ServerMetrics};

/// Characters of a request id shown in the matrix; the full id is in the cell's `title`.
const REQUEST_ID_TAIL: usize = 18;

const STYLE: &str = ":root{color-scheme:dark;--bg:#0b0e13;--panel:#121720;--line:#263041;--text:#e8edf5;\
--muted:#8d99ac;--blue:#8ba7ff;--green:#55d69e;--amber:#ffc76b;--red:#ff9ca8}*{box-sizing:border-box}\
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,\
Segoe UI,sans-serif}header{height:54px;display:flex;align-items:center;gap:12px;padding:0 22px;\
border-bottom:1px solid var(--line);background:#0e1219;position:sticky;top:0}.logo{font-weight:750;\
letter-spacing:.04em}.logo b{color:var(--blue)}.pill{border:1px solid var(--line);border-radius:5px;\
padding:3px 8px;color:var(--muted);font:11px ui-monospace,monospace}.pill.ok{color:var(--green);\
border-color:#285c4b}.pill.bad{color:var(--red);border-color:#704049}.right{margin-left:auto;display:flex;\
gap:10px;align-items:center;color:var(--muted);font:11px ui-monospace,monospace}main{padding:24px;\
max-width:1700px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}\
h2{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin:0 0 12px}\
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:20px}\
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px}.card span{\
display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em}.card strong{\
display:block;font:600 22px ui-monospace,SFMono-Regular,monospace;margin:4px 0 2px}.card small{\
color:var(--muted)}section{background:var(--panel);border:1px solid var(--line);border-radius:9px;\
padding:18px;margin-bottom:18px}.title{display:flex;align-items:center;gap:10px;margin-bottom:14px}\
.title code{color:var(--blue)}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;\
font:12px ui-monospace,SFMono-Regular,Consolas,monospace}th{text-align:left;color:var(--muted);\
font-weight:500;border-bottom:1px solid var(--line);padding:8px 7px}td{padding:8px 7px;\
border-bottom:1px solid #1c2431;white-space:nowrap}tr:last-child td{border-bottom:0}.hit{color:var(--green)}\
.miss{color:var(--amber)}.muted{color:var(--muted)}.error{border:1px solid #704049;background:#28191c;\
color:var(--red);padding:14px;border-radius:8px;font-family:ui-monospace,monospace}button{background:none;\
border:1px solid var(--line);color:var(--muted);border-radius:5px;padding:3px 9px;font:11px ui-monospace,\
monospace;cursor:pointer}form.login{max-width:420px;margin:12vh auto;background:var(--panel);\
border:1px solid var(--line);border-radius:9px;padding:24px}form.login input{width:100%;margin:12px 0;\
padding:9px;background:#0b0e13;color:var(--text);border:1px solid var(--line);border-radius:6px;\
font:13px ui-monospace,monospace}form.login button{color:var(--text);padding:7px 14px}\
@media(max-width:1000px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr)}table{font-size:11px}}";

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

fn card(html: &mut String, label: &str, value: &str, note: &str) {
    let _ = write!(
        html,
        "<div class=\"card\"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>"
    );
}

fn page_start(html: &mut String, title: &str, refresh_seconds: Option<u64>) {
    html.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    if let Some(seconds) = refresh_seconds {
        let _ = write!(html, "<meta http-equiv=\"refresh\" content=\"{seconds}\">");
    }
    let _ = write!(
        html,
        "<title>{}</title><style>{STYLE}</style></head><body>",
        escape(title)
    );
}

/// The full dashboard page.
pub fn dashboard(metrics: &ServerMetrics, refresh_seconds: u64) -> String {
    let mut html = String::with_capacity(16 * 1024);
    page_start(&mut html, "eSurge // workbench", Some(refresh_seconds));
    let status = if metrics.status.is_empty() {
        "unknown"
    } else {
        metrics.status.as_str()
    };
    let status_class = if status == "ready" { "ok" } else { "bad" };
    let _ = write!(
        html,
        "<header><div class=\"logo\">e<b>Surge</b> // workbench</div><span class=\"pill {status_class}\">{}</span>\
         <div class=\"right\"><span>uptime {}</span><span>refresh {refresh_seconds}s</span>\
         <form method=\"post\" action=\"/logout\"><button type=\"submit\">log out</button></form></div></header><main>",
        escape(status),
        duration(metrics.uptime_seconds),
    );
    html.push_str(
        "<h1>Inference cockpit</h1><p class=\"sub\">Prefill, decode and prefix-cache telemetry from the API server.</p>",
    );
    html.push_str("<div class=\"cards\">");
    card(&mut html, "Status", &escape(status), "API server");
    card(
        &mut html,
        "Requests",
        &metrics.total_requests.to_string(),
        &format!("{} failed", metrics.failed_requests),
    );
    card(
        &mut html,
        "Active",
        &metrics.active_requests.to_string(),
        "HTTP requests in flight",
    );
    card(
        &mut html,
        "Tokens generated",
        &metrics.total_tokens_generated.to_string(),
        "completion tokens",
    );
    card(
        &mut html,
        "Generation tok/s",
        &number(metrics.average_tokens_per_second),
        "per-request average",
    );
    html.push_str("</div>");

    for (name, engine) in &metrics.engine_metrics {
        model_section(&mut html, name, engine);
    }
    if let Some(auth) = &metrics.auth_stats {
        access_section(&mut html, auth);
    }
    html.push_str("<p class=\"muted\" style=\"font:11px ui-monospace,monospace\">server-rendered · no javascript</p>");
    html.push_str("</main></body></html>");
    html
}

fn model_section(html: &mut String, name: &str, engine: &EngineMetrics) {
    let agg: &Aggregate = &engine.aggregate;
    let _ = write!(
        html,
        "<section><div class=\"title\"><h2 style=\"margin:0\">Model</h2><code>{}</code>\
         <span class=\"pill\">running {} · waiting {}</span><span class=\"pill\">window {}s · {} done · {} failed</span></div>",
        escape(name),
        engine.running_requests,
        engine.waiting_requests,
        number(engine.window_seconds),
        agg.completed,
        agg.failed,
    );
    html.push_str("<div class=\"cards\">");
    card(
        html,
        "Prefill tok/s",
        &optional_number(agg.prefill_tps),
        "uncached input / TTFT",
    );
    card(
        html,
        "Decode tok/s",
        &optional_number(agg.decode_tps),
        "per request, after first token",
    );
    card(
        html,
        "Output tok/s",
        &optional_number(agg.output_tps),
        "all completions in window",
    );
    card(
        html,
        "TTFT p50 / p99",
        &format!("{} / {}", seconds(agg.ttft_p50_s), seconds(agg.ttft_p99_s)),
        "time to first token",
    );
    card(
        html,
        "Latency p50 / p99",
        &format!("{} / {}", seconds(agg.latency_p50_s), seconds(agg.latency_p99_s)),
        "end-to-end",
    );
    card(
        html,
        "Request hit rate",
        &percent(agg.cache_hit_rate),
        &format!("{} cache hits", agg.cache_hits),
    );
    card(
        html,
        "Token hit rate",
        &percent(agg.token_cache_hit_rate),
        "prompt tokens read from cache",
    );
    card(html, "Cache read", &agg.cache_read_tokens.to_string(), "tokens");
    card(
        html,
        "Uncached input",
        &agg.uncached_input_tokens.to_string(),
        "tokens prefilled",
    );
    html.push_str("</div><h2>Decode / prefill cache matrix</h2><div class=\"scroll\"><table><thead><tr>");
    for heading in [
        "Request",
        "Finish",
        "Prompt",
        "Output",
        "TTFT",
        "Latency",
        "Decode tok/s",
        "Cache",
        "Read",
        "Uncached",
    ] {
        let _ = write!(html, "<th>{heading}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    if engine.recent_requests.is_empty() {
        html.push_str("<tr><td colspan=\"10\" class=\"muted\">no completed requests in the window</td></tr>");
    }
    for request in &engine.recent_requests {
        request_row(html, request);
    }
    html.push_str("</tbody></table></div></section>");
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
    let (cache_class, cache_label) = if request.cache_hit {
        ("hit", "HIT")
    } else {
        ("miss", "miss")
    };
    let _ = write!(
        html,
        "<tr><td title=\"{}\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\
         <td class=\"{cache_class}\">{cache_label}</td><td>{}</td><td>{}</td></tr>",
        escape(id),
        escape(&short),
        escape(request.finish_reason.as_deref().unwrap_or("—")),
        request.prompt_tokens,
        request.completion_tokens,
        seconds(request.ttft_s),
        seconds(request.latency_s),
        optional_number(request.decode_tps),
        request.cache_read_tokens,
        request.uncached_input_tokens,
    );
}

fn access_section(html: &mut String, auth: &AuthStats) {
    html.push_str("<section><h2>Access</h2><div class=\"cards\">");
    card(
        html,
        "Active keys",
        &auth.active_keys.to_string(),
        &format!("{} total", auth.total_keys),
    );
    card(
        html,
        "Key requests",
        &auth.total_requests_all_keys.to_string(),
        "authenticated calls",
    );
    card(
        html,
        "Key tokens",
        &auth.total_tokens_all_keys.to_string(),
        "tokens across keys",
    );
    card(
        html,
        "Audit entries",
        &auth.audit_log_entries.to_string(),
        "retained audit log",
    );
    html.push_str("</div></section>");
}

/// Login form, with an optional error line.
pub fn login(error: Option<&str>) -> String {
    let mut html = String::new();
    page_start(&mut html, "eSurge // sign in", None);
    html.push_str(
        "<form class=\"login\" method=\"post\" action=\"/login\"><div class=\"logo\">e<b>Surge</b> // workbench</div>",
    );
    html.push_str("<p class=\"muted\">Sign in with an API key for this server.</p>");
    if let Some(error) = error {
        let _ = write!(html, "<div class=\"error\">{}</div>", escape(error));
    }
    html.push_str(
        "<input type=\"password\" name=\"api_key\" autocomplete=\"current-password\" placeholder=\"API key\" \
         required autofocus><button type=\"submit\">Sign in</button></form></body></html>",
    );
    html
}

/// Page shown when the API server cannot be read.
pub fn upstream_error(message: &str) -> String {
    let mut html = String::new();
    page_start(&mut html, "eSurge // unavailable", Some(15));
    let _ = write!(
        html,
        "<main><h1>API server unavailable</h1><div class=\"error\">{}</div></main></body></html>",
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
              "running_requests":2,"aggregate":{"prefill_tps":2100.0,"decode_tps":47.1,"cache_hit_rate":0.5,
              "ttft_p50_s":null,"ttft_p99_s":2.1},
              "recent_requests":[{"request_id":"chat-<b>0123456789abcdef0123","finish_reason":"stop",
              "prompt_tokens":332,"completion_tokens":31,"ttft_s":1.88,"cache_hit":true,"cache_read_tokens":256,
              "uncached_input_tokens":76},{"request_id":"short","cache_hit":false}]}},
              "auth_stats":{"active_keys":1,"total_keys":1}}"#,
        )
        .unwrap()
    }

    #[test]
    fn dashboard_renders_matrix_rows() {
        let html = dashboard(&sample(), 5);
        assert!(html.contains("Decode / prefill cache matrix"));
        assert!(html.contains("<td class=\"hit\">HIT</td><td>256</td><td>76</td>"));
        assert!(html.contains("<td class=\"miss\">miss</td>"));
        assert!(html.contains("content=\"5\""));
        assert!(html.contains("01h 02m 05s"));
        assert!(html.contains("50.0%"));
    }

    #[test]
    fn upstream_strings_are_escaped() {
        let html = dashboard(&sample(), 5);
        assert!(!html.contains("<script>m</script>"));
        assert!(html.contains("&lt;script&gt;m&lt;/script&gt;"));
        assert!(!html.contains("<b>0123"));
        assert!(html.contains("title=\"chat-&lt;b&gt;0123456789abcdef0123\""));
        assert!(!upstream_error("<img src=x>").contains("<img"));
        assert!(!login(Some("<x>")).contains("<x>"));
    }

    #[test]
    fn null_values_render_as_dash() {
        let html = dashboard(&sample(), 5);
        assert!(html.contains("— / 2.10s"));
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
        assert!(html.contains("…éééééééééééééééééé<"));
    }
}
