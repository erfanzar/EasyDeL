//! End-to-end tests against a fake API server.

use std::net::SocketAddr;

use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use clap::Parser;
use esurge_dashboard::{build_app, Config};
use reqwest::{redirect, Client};

const GOOD_KEY: &str = "good-Key_123";

const SAMPLE: &str = r#"{"status":"ready","uptime_seconds":848.5,"total_requests":3,"engine_metrics":{
  "GLM":{"running_requests":2,"waiting_requests":0,"window_seconds":300.0,
    "aggregate":{"completed":2,"cache_hits":1,"cache_hit_rate":0.5,"ttft_p50_s":1.9},
    "recent_requests":[{"request_id":"chat-1","finish_reason":"stop","prompt_tokens":332,"completion_tokens":31,
      "ttft_s":1.88,"cache_hit":true,"cache_read_tokens":256,"uncached_input_tokens":76}]}}}"#;

async fn fake_metrics(headers: HeaderMap) -> axum::response::Response {
    let authorized = headers.get("authorization").and_then(|value| value.to_str().ok()) == Some("Bearer good-Key_123");
    if authorized {
        ([("content-type", "application/json")], SAMPLE).into_response()
    } else {
        (StatusCode::UNAUTHORIZED, r#"{"detail":"Missing or invalid API key"}"#).into_response()
    }
}

async fn spawn(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    addr
}

async fn dashboard_for(upstream: &str) -> (SocketAddr, Client) {
    let config = Config::parse_from([
        "esurge-dashboard",
        "--upstream",
        upstream,
        "--upstream-timeout-seconds",
        "2",
    ]);
    let addr = spawn(build_app(&config).unwrap()).await;
    let client = Client::builder().redirect(redirect::Policy::none()).build().unwrap();
    (addr, client)
}

async fn setup() -> (SocketAddr, Client) {
    let upstream = spawn(Router::new().route("/metrics", get(fake_metrics))).await;
    dashboard_for(&format!("http://{upstream}")).await
}

fn location(response: &reqwest::Response) -> &str {
    response.headers().get("location").unwrap().to_str().unwrap()
}

#[tokio::test]
async fn header_credentials_are_forwarded_and_judged_by_upstream() {
    let (addr, client) = setup().await;
    let url = format!("http://{addr}/");

    let anonymous = client.get(&url).send().await.unwrap();
    assert_eq!(anonymous.status(), StatusCode::SEE_OTHER);
    assert_eq!(location(&anonymous), "/login");

    let good = client.get(&url).bearer_auth(GOOD_KEY).send().await.unwrap();
    assert_eq!(good.status(), StatusCode::OK);
    assert_eq!(good.headers()["cache-control"], "no-store");
    let body = good.text().await.unwrap();
    assert!(body.contains("Inference <span class=\"hl\">cockpit</span>") && body.contains("HIT"));
    assert!(body.contains("data-chart=\"gen\"") && body.contains("/assets/dashboard.js"));

    let lowercase_scheme = client
        .get(&url)
        .header("authorization", format!("bearer {GOOD_KEY}"))
        .send()
        .await
        .unwrap();
    assert_eq!(lowercase_scheme.status(), StatusCode::OK);

    assert_eq!(
        client.get(&url).bearer_auth("wrong").send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .get(&url)
            .bearer_auth("good-key_123")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .get(&url)
            .header("X-API-Key", GOOD_KEY)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );

    let query = client.get(format!("{url}?api_key={GOOD_KEY}")).send().await.unwrap();
    assert_eq!(
        query.status(),
        StatusCode::SEE_OTHER,
        "keys in the query string must be ignored"
    );
}

#[tokio::test]
async fn login_session_logout_cycle() {
    let (addr, client) = setup().await;
    let base = format!("http://{addr}");

    assert_eq!(
        client.get(format!("{base}/login")).send().await.unwrap().status(),
        StatusCode::OK
    );

    let bad = client
        .post(format!("{base}/login"))
        .form(&[("api_key", "nope")])
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), StatusCode::UNAUTHORIZED);
    assert!(bad.headers().get("set-cookie").is_none());

    let good = client
        .post(format!("{base}/login"))
        .form(&[("api_key", GOOD_KEY)])
        .send()
        .await
        .unwrap();
    assert_eq!(good.status(), StatusCode::SEE_OTHER);
    assert_eq!(location(&good), "/");
    let set_cookie = good.headers()["set-cookie"].to_str().unwrap().to_string();
    assert!(set_cookie.contains("HttpOnly") && set_cookie.contains("SameSite=Strict"));
    assert!(!set_cookie.contains(GOOD_KEY), "the cookie must not carry the key");
    let cookie = set_cookie.split(';').next().unwrap().to_string();

    let page = client
        .get(format!("{base}/"))
        .header("cookie", &cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), StatusCode::OK);
    let api = client
        .get(format!("{base}/api/metrics"))
        .header("cookie", &cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(api.status(), StatusCode::OK);
    let api_body: serde_json::Value = serde_json::from_str(&api.text().await.unwrap()).unwrap();
    assert_eq!(api_body["status"], "ready");

    let logout = client
        .post(format!("{base}/logout"))
        .header("cookie", &cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(logout.status(), StatusCode::SEE_OTHER);
    assert!(logout.headers()["set-cookie"].to_str().unwrap().contains("Max-Age=0"));

    let after = client
        .get(format!("{base}/"))
        .header("cookie", &cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), StatusCode::SEE_OTHER);
    assert_eq!(location(&after), "/login");
    let forged = client
        .get(format!("{base}/"))
        .header("cookie", "esurge_session=deadbeef")
        .send()
        .await
        .unwrap();
    assert_eq!(forged.status(), StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn api_metrics_passthrough_health_and_unknown_paths() {
    let (addr, client) = setup().await;
    let base = format!("http://{addr}");

    let anonymous = client.get(format!("{base}/api/metrics")).send().await.unwrap();
    assert_eq!(anonymous.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(anonymous.text().await.unwrap(), r#"{"error":"api key required"}"#);

    let good = client
        .get(format!("{base}/api/metrics"))
        .bearer_auth(GOOD_KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(good.status(), StatusCode::OK);
    assert_eq!(good.text().await.unwrap(), SAMPLE);

    let bad = client
        .get(format!("{base}/api/metrics"))
        .bearer_auth("x")
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), StatusCode::UNAUTHORIZED);

    let health = client.get(format!("{base}/healthz")).send().await.unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.text().await.unwrap(), "ok\n");
    assert_eq!(
        client.get(format!("{base}/nope")).send().await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn history_is_gated_by_upstream_and_returns_deltas() {
    let (addr, client) = setup().await;
    let base = format!("http://{addr}");

    let anonymous = client.get(format!("{base}/api/history")).send().await.unwrap();
    assert_eq!(anonymous.status(), StatusCode::UNAUTHORIZED);
    let rejected = client
        .get(format!("{base}/api/history"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);

    let first = client
        .get(format!("{base}/api/history?after=0"))
        .bearer_auth(GOOD_KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(first.headers()["cache-control"], "no-store");
    let body: serde_json::Value = serde_json::from_str(&first.text().await.unwrap()).unwrap();
    assert_eq!(body["metrics"]["status"], "ready");
    let history = &body["history"];
    assert_eq!(history["samples"].as_array().unwrap().len(), 1);
    assert_eq!(history["samples"][0]["models"]["GLM"]["running_requests"], 2);
    let requests = history["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["request_id"], "chat-1");
    assert_eq!(requests[0]["model"], "GLM");
    assert_eq!(requests[0]["cache_read_tokens"], 256);

    // The same request is logged once; a second poll inside one second adds no sample either.
    let seq = history["seq"].as_u64().unwrap();
    let second = client
        .get(format!("{base}/api/history?after={seq}"))
        .bearer_auth(GOOD_KEY)
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_str(&second.text().await.unwrap()).unwrap();
    assert!(body["history"]["requests"].as_array().unwrap().is_empty());
    assert!(body["history"]["samples"].as_array().unwrap().is_empty());
    assert_eq!(body["history"]["reset"], false);
}

#[tokio::test]
async fn assets_and_security_headers() {
    let (addr, client) = setup().await;
    let base = format!("http://{addr}");

    let css = client.get(format!("{base}/assets/dashboard.css")).send().await.unwrap();
    assert_eq!(css.status(), StatusCode::OK);
    assert!(css.headers()["content-type"].to_str().unwrap().starts_with("text/css"));
    assert!(css.text().await.unwrap().contains("--bg: #f4f5f8"));

    let js = client.get(format!("{base}/assets/dashboard.js")).send().await.unwrap();
    assert_eq!(js.status(), StatusCode::OK);
    assert!(js.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/javascript"));

    let page = client.get(format!("{base}/login")).send().await.unwrap();
    let csp = page.headers()["content-security-policy"].to_str().unwrap();
    assert!(csp.contains("script-src 'self'") && csp.contains("frame-ancestors 'none'"));
    assert!(!csp.contains("unsafe-inline"));
    assert_eq!(page.headers()["x-content-type-options"], "nosniff");
    assert_eq!(page.headers()["x-frame-options"], "DENY");
}

#[tokio::test]
async fn unreachable_upstream_is_a_bad_gateway() {
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let (addr, client) = dashboard_for(&format!("http://{closed}")).await;
    let page = client
        .get(format!("http://{addr}/"))
        .bearer_auth(GOOD_KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), StatusCode::BAD_GATEWAY);
    assert!(page.text().await.unwrap().contains("API server unavailable"));
    let login = client
        .post(format!("http://{addr}/login"))
        .form(&[("api_key", GOOD_KEY)])
        .send()
        .await
        .unwrap();
    assert_eq!(login.status(), StatusCode::BAD_GATEWAY);
    assert!(login.headers().get("set-cookie").is_none());
}

#[tokio::test]
async fn invalid_upstream_json_is_a_bad_gateway() {
    let upstream = spawn(Router::new().route("/metrics", get(|| async { "not json" }))).await;
    let (addr, client) = dashboard_for(&format!("http://{upstream}")).await;
    let page = client
        .get(format!("http://{addr}/"))
        .bearer_auth(GOOD_KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), StatusCode::BAD_GATEWAY);
}
