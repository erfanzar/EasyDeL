//! Command-line configuration.

use std::net::SocketAddr;

use clap::Parser;

/// Longest accepted session lifetime (30 days).
pub const MAX_SESSION_TTL_SECONDS: u64 = 30 * 24 * 3600;

#[derive(Clone, Debug, Parser)]
#[command(
    name = "esurge-dashboard",
    version,
    about = "Dashboard for the eSurge OpenAI-compatible API server"
)]
pub struct Config {
    /// Base URL of the eSurge API server. Only `GET {upstream}/metrics` is read.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub upstream: String,

    /// Address the dashboard listens on.
    #[arg(long, default_value = "127.0.0.1:9090")]
    pub listen: SocketAddr,

    /// Seconds between page refreshes.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub refresh_seconds: u64,

    /// Timeout for one upstream metrics request, in seconds.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..=300))]
    pub upstream_timeout_seconds: u64,

    /// Lifetime of a browser login session, in seconds.
    #[arg(long, default_value_t = 43_200, value_parser = clap::value_parser!(u64).range(1..=MAX_SESSION_TTL_SECONDS))]
    pub session_ttl_seconds: u64,

    /// Metrics snapshots kept in memory for the charts (one per second at most; 43200 is 12 h of continuous polling).
    #[arg(long, default_value_t = 43_200, value_parser = clap::value_parser!(u64).range(1..=1_000_000))]
    pub history_samples: u64,

    /// Completed requests kept in memory for the request log and per-request charts.
    #[arg(long, default_value_t = 5_000, value_parser = clap::value_parser!(u64).range(1..=1_000_000))]
    pub history_requests: u64,

    /// Mark the session cookie `Secure` (set this when the dashboard is served over HTTPS).
    #[arg(long)]
    pub secure_cookie: bool,
}
