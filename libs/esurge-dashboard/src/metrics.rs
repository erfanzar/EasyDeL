//! Typed view of the API server's `GET /metrics` JSON. Missing fields default; unknown fields are ignored.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct ServerMetrics {
    pub status: String,
    pub uptime_seconds: f64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub average_tokens_per_second: f64,
    pub active_requests: u64,
    pub models_loaded: u64,
    pub auth_stats: Option<AuthStats>,
    pub engine_metrics: BTreeMap<String, EngineMetrics>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AuthStats {
    pub total_keys: u64,
    pub active_keys: u64,
    pub total_requests_all_keys: u64,
    pub total_tokens_all_keys: u64,
    pub audit_log_entries: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct EngineMetrics {
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub window_seconds: f64,
    pub aggregate: Aggregate,
    pub recent_requests: Vec<RecentRequest>,
    /// Engine-step prefill/decode split; absent on API servers that predate it.
    pub phase: Option<Phase>,
}

/// Tokens the engine computed per phase: prompt tokens are prefill, everything else is decode.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Phase {
    pub window_seconds: f64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub steps: u64,
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub mixed_steps: u64,
    pub prefill_tokens_total: u64,
    pub decode_tokens_total: u64,
    pub steps_total: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Aggregate {
    pub completed: u64,
    pub failed: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: Option<f64>,
    pub cache_read_tokens: u64,
    pub uncached_input_tokens: u64,
    pub token_cache_hit_rate: Option<f64>,
    pub ttft_p50_s: Option<f64>,
    pub ttft_p99_s: Option<f64>,
    pub latency_p50_s: Option<f64>,
    pub latency_p99_s: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub output_tps: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RecentRequest {
    pub request_id: String,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_s: Option<f64>,
    pub latency_s: Option<f64>,
    pub decode_tps: Option<f64>,
    pub cache_hit: bool,
    pub cache_read_tokens: u64,
    pub uncached_input_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_and_null_fields_default() {
        let parsed: ServerMetrics = serde_json::from_str(
            r#"{"status":"ready","unknown":1,"engine_metrics":{"m":{"aggregate":{"ttft_p50_s":null},
                "recent_requests":[{"request_id":"r1","cache_hit":true}],
                "phase":{"prefill_tps":12.5,"decode_tokens_total":7}},"old":{}}}"#,
        )
        .unwrap();
        let model = &parsed.engine_metrics["m"];
        let phase = model.phase.as_ref().unwrap();
        assert_eq!(phase.prefill_tps, Some(12.5));
        assert_eq!((phase.decode_tokens_total, phase.prefill_tokens_total), (7, 0));
        assert!(parsed.engine_metrics["old"].phase.is_none());
        assert_eq!(parsed.status, "ready");
        assert!(parsed.auth_stats.is_none());
        assert_eq!(model.aggregate.ttft_p50_s, None);
        assert!(model.recent_requests[0].cache_hit);
        assert_eq!(model.recent_requests[0].prompt_tokens, 0);
    }
}
