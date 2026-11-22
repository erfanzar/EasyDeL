//! HTTP routes. Every page that shows data forwards the caller's key to the API server and lets it decide.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{middleware, Form, Router};
use serde::Deserialize;

use crate::auth::{clear_session_cookie, presented_key, session_cookie, session_id};
use crate::config::Config;
use crate::history::History;
use crate::metrics::ServerMetrics;
use crate::render;
use crate::session::SessionStore;
use crate::upstream::Upstream;

/// Pages load only their own script and stylesheet, plus the JetBrains Mono webfont.
const CONTENT_SECURITY_POLICY: &str = "default-src 'none'; script-src 'self'; \
     style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; \
     img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'";

struct AppState {
    upstream: Upstream,
    sessions: SessionStore,
    history: History,
    refresh_seconds: u64,
    session_ttl_seconds: u64,
    secure_cookie: bool,
}

/// Where the request's key came from: a cookie session is dropped when the API server rejects its key.
enum Credential {
    Header(String),
    Session { id: String, api_key: String },
}

impl Credential {
    fn api_key(&self) -> &str {
        match self {
            Credential::Header(key) => key,
            Credential::Session { api_key, .. } => api_key,
        }
    }
}

fn credential(state: &AppState, headers: &HeaderMap) -> Option<Credential> {
    if let Some(key) = presented_key(headers) {
        return Some(Credential::Header(key));
    }
    let id = session_id(headers)?;
    let api_key = state.sessions.api_key(&id)?;
    Some(Credential::Session { id, api_key })
}

pub fn build_app(config: &Config) -> Result<Router, String> {
    let state = Arc::new(AppState {
        upstream: Upstream::new(&config.upstream, Duration::from_secs(config.upstream_timeout_seconds))?,
        sessions: SessionStore::new(Duration::from_secs(config.session_ttl_seconds)),
        history: History::new(
            usize::try_from(config.history_samples).unwrap_or(usize::MAX),
            usize::try_from(config.history_requests).unwrap_or(usize::MAX),
        ),
        refresh_seconds: config.refresh_seconds,
        session_ttl_seconds: config.session_ttl_seconds,
        secure_cookie: config.secure_cookie,
    });
    Ok(Router::new()
        .route("/", get(dashboard))
        .route("/api/metrics", get(api_metrics))
        .route("/api/history", get(api_history))
        .route("/assets/dashboard.css", get(stylesheet))
        .route("/assets/dashboard.js", get(script))
        .route("/login", get(login_page).post(login))
        .route("/logout", post(logout))
        .route("/healthz", get(|| async { "ok\n" }))
        .layer(middleware::map_response(security_headers))
        .with_state(state))
}

async fn security_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    response
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn with_cookie(mut response: Response, cookie: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(cookie) {
        response.headers_mut().append(header::SET_COOKIE, value);
    }
    response
}

fn is_rejection(status: StatusCode) -> bool {
    matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
}

/// Result of reading `/metrics` with a viewer's key. A successful read is also recorded into the chart history.
enum Fetched {
    Ok {
        metrics: ServerMetrics,
        raw: serde_json::Value,
    },
    Rejected(StatusCode),
    Failed(String),
}

async fn fetch(state: &AppState, credential: &Credential) -> Fetched {
    let upstream = match state.upstream.fetch(credential.api_key()).await {
        Ok(upstream) => upstream,
        Err(error) => return Fetched::Failed(error),
    };
    if is_rejection(upstream.status) {
        if let Credential::Session { id, .. } = credential {
            state.sessions.remove(id);
        }
        return Fetched::Rejected(upstream.status);
    }
    if !upstream.status.is_success() {
        return Fetched::Failed(format!("API server returned {}", upstream.status));
    }
    let parsed = serde_json::from_slice::<ServerMetrics>(&upstream.body)
        .and_then(|metrics| serde_json::from_slice::<serde_json::Value>(&upstream.body).map(|raw| (metrics, raw)));
    match parsed {
        Ok((metrics, raw)) => {
            state.history.record(&metrics, unix_now());
            Fetched::Ok { metrics, raw }
        }
        Err(error) => Fetched::Failed(format!("API server sent invalid metrics JSON: {error}")),
    }
}

fn session_rejected(state: &AppState) -> Response {
    with_cookie(
        Redirect::to("/login").into_response(),
        &clear_session_cookie(state.secure_cookie),
    )
}

async fn dashboard(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(credential) = credential(&state, &headers) else {
        return no_store(Redirect::to("/login").into_response());
    };
    let response = match fetch(&state, &credential).await {
        Fetched::Ok { metrics, .. } => Html(render::dashboard(&metrics, state.refresh_seconds)).into_response(),
        Fetched::Rejected(status) => match credential {
            Credential::Session { .. } => session_rejected(&state),
            Credential::Header(_) => (status, "API key rejected by the API server\n").into_response(),
        },
        Fetched::Failed(message) => (StatusCode::BAD_GATEWAY, Html(render::upstream_error(&message))).into_response(),
    };
    no_store(response)
}

fn json_response(status: StatusCode, body: String) -> Response {
    no_store(
        (
            status,
            [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
            body,
        )
            .into_response(),
    )
}

fn json_error(status: StatusCode, message: &str) -> Response {
    json_response(status, serde_json::json!({ "error": message }).to_string())
}

/// The API server's `/metrics` body, passed through unchanged.
async fn api_metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(credential) = credential(&state, &headers) else {
        return json_error(StatusCode::UNAUTHORIZED, "api key required");
    };
    match state.upstream.fetch(credential.api_key()).await {
        Ok(upstream) => {
            if is_rejection(upstream.status) {
                if let Credential::Session { id, .. } = &credential {
                    state.sessions.remove(id);
                }
            } else if upstream.status.is_success() {
                if let Ok(metrics) = serde_json::from_slice::<ServerMetrics>(&upstream.body) {
                    state.history.record(&metrics, unix_now());
                }
            }
            let json = [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))];
            no_store((upstream.status, json, upstream.body).into_response())
        }
        Err(error) => json_error(StatusCode::BAD_GATEWAY, &error),
    }
}

#[derive(Deserialize)]
struct HistoryQuery {
    /// Return only entries with a sequence number above this one.
    #[serde(default)]
    after: u64,
    /// Return at most this many of the newest samples.
    limit: Option<usize>,
}

/// Current metrics plus the chart history the client has not seen yet. Each call is checked upstream first.
async fn api_history(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<HistoryQuery>,
) -> Response {
    let Some(credential) = credential(&state, &headers) else {
        return json_error(StatusCode::UNAUTHORIZED, "api key required");
    };
    match fetch(&state, &credential).await {
        Fetched::Ok { raw, .. } => {
            let mut view = state.history.view(query.after);
            if let Some(limit) = query.limit {
                let excess = view.samples.len().saturating_sub(limit);
                view.samples.drain(..excess);
            }
            let body = serde_json::json!({ "metrics": raw, "history": view }).to_string();
            json_response(StatusCode::OK, body)
        }
        Fetched::Rejected(status) => json_error(status, "API key rejected by the API server"),
        Fetched::Failed(message) => json_error(StatusCode::BAD_GATEWAY, &message),
    }
}

async fn stylesheet() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        render::CSS,
    )
        .into_response()
}

async fn script() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        render::JS,
    )
        .into_response()
}

async fn login_page() -> Response {
    no_store(Html(render::login(None)).into_response())
}

#[derive(Deserialize)]
struct LoginForm {
    api_key: String,
}

async fn login(State(state): State<Arc<AppState>>, Form(form): Form<LoginForm>) -> Response {
    let api_key = form.api_key.trim().to_string();
    if api_key.is_empty() {
        return no_store((StatusCode::UNAUTHORIZED, Html(render::login(Some("Enter an API key.")))).into_response());
    }
    let response = match state.upstream.fetch(&api_key).await {
        Ok(upstream) if upstream.status.is_success() => match state.sessions.create(api_key) {
            Ok(id) => with_cookie(
                Redirect::to("/").into_response(),
                &session_cookie(&id, state.session_ttl_seconds, state.secure_cookie),
            ),
            Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, Html(render::login(Some(&error)))).into_response(),
        },
        Ok(upstream) if is_rejection(upstream.status) => (
            StatusCode::UNAUTHORIZED,
            Html(render::login(Some("The API server rejected this key."))),
        )
            .into_response(),
        Ok(upstream) => {
            let message = format!("API server returned {}", upstream.status);
            (StatusCode::BAD_GATEWAY, Html(render::login(Some(&message)))).into_response()
        }
        Err(error) => (StatusCode::BAD_GATEWAY, Html(render::login(Some(&error)))).into_response(),
    };
    no_store(response)
}

async fn logout(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(id) = session_id(&headers) {
        state.sessions.remove(&id);
    }
    no_store(with_cookie(
        Redirect::to("/login").into_response(),
        &clear_session_cookie(state.secure_cookie),
    ))
}
