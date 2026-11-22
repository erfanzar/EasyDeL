//! Server-rendered dashboard for the eSurge OpenAI-compatible API server.
//!
//! The dashboard owns no credentials. Every page forwards the caller's API key to the API server's
//! `GET /metrics`, which decides whether the key is valid; the dashboard only renders the JSON it returns.

pub mod auth;
pub mod config;
pub mod history;
pub mod metrics;
pub mod render;
pub mod routes;
pub mod session;
pub mod upstream;

pub use config::Config;
pub use routes::build_app;

/// Bind `config.listen` and serve until SIGTERM or Ctrl-C.
pub async fn serve(config: Config) -> Result<(), String> {
    let app = build_app(&config)?;
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|error| format!("bind {}: {error}", config.listen))?;
    let local = listener.local_addr().map_err(|error| error.to_string())?;
    println!(
        "esurge-dashboard listening on http://{local} (upstream {})",
        config.upstream
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
}

async fn shutdown_signal() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}
