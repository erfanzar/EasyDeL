//! HTTP routes. Every page that shows data forwards the caller's key to the API server and lets it decide.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use serde::Deserialize;

use crate::auth::{clear_session_cookie, presented_key, session_cookie, session_id};
use crate::config::Config;
use crate::metrics::ServerMetrics;
use crate::render;
use crate::session::SessionStore;
use crate::upstream::Upstream;

struct AppState {
    upstream: Upstream,
    sessions: SessionStore,
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
        refresh_seconds: config.refresh_seconds,
        session_ttl_seconds: config.session_ttl_seconds,
        secure_cookie: config.secure_cookie,
    });
    Ok(Router::new()
        .route("/", get(dashboard))
        .route("/api/metrics", get(api_metrics))
        .route("/login", get(login_page).post(login))
        .route("/logout", post(logout))
        .route("/healthz", get(|| async { "ok\n" }))
        .with_state(state))
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

async fn dashboard(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(credential) = credential(&state, &headers) else {
        return no_store(Redirect::to("/login").into_response());
    };
    let response = match state.upstream.fetch(credential.api_key()).await {
        Err(error) => (StatusCode::BAD_GATEWAY, Html(render::upstream_error(&error))).into_response(),
        Ok(upstream) if is_rejection(upstream.status) => match &credential {
            Credential::Session { id, .. } => {
                state.sessions.remove(id);
                with_cookie(
                    Redirect::to("/login").into_response(),
                    &clear_session_cookie(state.secure_cookie),
                )
            }
            Credential::Header(_) => (upstream.status, "API key rejected by the API server\n").into_response(),
        },
        Ok(upstream) if !upstream.status.is_success() => {
            let message = format!("API server returned {}", upstream.status);
            (StatusCode::BAD_GATEWAY, Html(render::upstream_error(&message))).into_response()
        }
        Ok(upstream) => match serde_json::from_slice::<ServerMetrics>(&upstream.body) {
            Ok(metrics) => Html(render::dashboard(&metrics, state.refresh_seconds)).into_response(),
            Err(error) => {
                let message = format!("API server sent invalid metrics JSON: {error}");
                (StatusCode::BAD_GATEWAY, Html(render::upstream_error(&message))).into_response()
            }
        },
    };
    no_store(response)
}

async fn api_metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let json = [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))];
    let Some(credential) = credential(&state, &headers) else {
        return no_store((StatusCode::UNAUTHORIZED, json, r#"{"error":"api key required"}"#).into_response());
    };
    let response = match state.upstream.fetch(credential.api_key()).await {
        Ok(upstream) => {
            if let (true, Credential::Session { id, .. }) = (is_rejection(upstream.status), &credential) {
                state.sessions.remove(id);
            }
            (upstream.status, json, upstream.body).into_response()
        }
        Err(error) => {
            let body = serde_json::json!({ "error": error }).to_string();
            (StatusCode::BAD_GATEWAY, json, body).into_response()
        }
    };
    no_store(response)
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
