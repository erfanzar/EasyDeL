//! Rolling history of `/metrics` snapshots and completed requests, kept for the charts.
//!
//! The dashboard holds no API key, so it cannot poll the API server on its own. History grows from the fetches
//! that viewers make: each one is validated upstream with the viewer's own key before anything is recorded or
//! returned. With a browser open the page polls every few seconds, so the charts cover the time it was open.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::{Mutex, MutexGuard};

use serde::Serialize;

use crate::metrics::{Aggregate, Phase, RecentRequest, ServerMetrics};

/// Snapshots closer together than this are recorded once, so several open browsers do not multiply the history.
pub const MIN_SAMPLE_INTERVAL_SECONDS: f64 = 1.0;

/// One `/metrics` snapshot, reduced to the numbers the charts plot.
#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub seq: u64,
    /// Unix time the snapshot was taken, in seconds.
    pub t: f64,
    pub uptime_seconds: f64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_requests: u64,
    pub total_tokens_generated: u64,
    pub key_requests: Option<u64>,
    pub key_tokens: Option<u64>,
    pub models: BTreeMap<String, ModelSample>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelSample {
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub aggregate: Aggregate,
    pub phase: Option<Phase>,
}

/// A completed request, logged once the first time it appears in a model's `recent_requests`.
#[derive(Clone, Debug, Serialize)]
pub struct LoggedRequest {
    pub seq: u64,
    pub model: String,
    /// Unix time of the snapshot in which the request first appeared.
    pub seen_at: f64,
    #[serde(flatten)]
    pub request: RecentRequest,
}

/// Entries newer than a sequence number. Clients pass the returned `seq` back to receive only what is new.
#[derive(Debug, Serialize)]
pub struct HistoryView {
    pub seq: u64,
    /// The client's `after` was ahead of this store (the dashboard restarted): this view is complete, not a delta.
    pub reset: bool,
    pub sample_capacity: usize,
    pub request_capacity: usize,
    pub samples: Vec<Sample>,
    pub requests: Vec<LoggedRequest>,
}

struct Inner {
    last_seq: u64,
    samples: VecDeque<Sample>,
    requests: VecDeque<LoggedRequest>,
    seen: HashSet<(String, String)>,
}

pub struct History {
    sample_capacity: usize,
    request_capacity: usize,
    inner: Mutex<Inner>,
}

impl History {
    pub fn new(sample_capacity: usize, request_capacity: usize) -> Self {
        Self {
            sample_capacity: sample_capacity.max(1),
            request_capacity: request_capacity.max(1),
            inner: Mutex::new(Inner {
                last_seq: 0,
                samples: VecDeque::new(),
                requests: VecDeque::new(),
                seen: HashSet::new(),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record a snapshot taken at unix time `now`, and log every request it shows for the first time.
    pub fn record(&self, metrics: &ServerMetrics, now: f64) {
        let mut guard = self.lock();
        let inner = &mut *guard;
        let due = inner
            .samples
            .back()
            .is_none_or(|last| now - last.t >= MIN_SAMPLE_INTERVAL_SECONDS);
        if due {
            inner.last_seq += 1;
            inner.samples.push_back(Sample {
                seq: inner.last_seq,
                t: now,
                uptime_seconds: metrics.uptime_seconds,
                total_requests: metrics.total_requests,
                successful_requests: metrics.successful_requests,
                failed_requests: metrics.failed_requests,
                active_requests: metrics.active_requests,
                total_tokens_generated: metrics.total_tokens_generated,
                key_requests: metrics.auth_stats.as_ref().map(|auth| auth.total_requests_all_keys),
                key_tokens: metrics.auth_stats.as_ref().map(|auth| auth.total_tokens_all_keys),
                models: metrics
                    .engine_metrics
                    .iter()
                    .map(|(name, engine)| {
                        let sample = ModelSample {
                            running_requests: engine.running_requests,
                            waiting_requests: engine.waiting_requests,
                            aggregate: engine.aggregate.clone(),
                            phase: engine.phase.clone(),
                        };
                        (name.clone(), sample)
                    })
                    .collect(),
            });
            while inner.samples.len() > self.sample_capacity {
                inner.samples.pop_front();
            }
        }
        for (model, engine) in &metrics.engine_metrics {
            // The API server lists recent requests newest first; log them oldest first.
            for request in engine.recent_requests.iter().rev() {
                if request.request_id.is_empty() || !inner.seen.insert((model.clone(), request.request_id.clone())) {
                    continue;
                }
                inner.last_seq += 1;
                inner.requests.push_back(LoggedRequest {
                    seq: inner.last_seq,
                    model: model.clone(),
                    seen_at: now,
                    request: request.clone(),
                });
                while inner.requests.len() > self.request_capacity {
                    if let Some(old) = inner.requests.pop_front() {
                        inner.seen.remove(&(old.model, old.request.request_id));
                    }
                }
            }
        }
    }

    /// Samples and requests with a sequence number above `after`.
    pub fn view(&self, after: u64) -> HistoryView {
        let inner = self.lock();
        let reset = after > inner.last_seq;
        let after = if reset { 0 } else { after };
        HistoryView {
            seq: inner.last_seq,
            reset,
            sample_capacity: self.sample_capacity,
            request_capacity: self.request_capacity,
            samples: inner
                .samples
                .iter()
                .filter(|sample| sample.seq > after)
                .cloned()
                .collect(),
            requests: inner
                .requests
                .iter()
                .filter(|request| request.seq > after)
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(ids: &[&str], total_requests: u64) -> ServerMetrics {
        let recent: Vec<String> = ids
            .iter()
            .map(|id| format!(r#"{{"request_id":"{id}","prompt_tokens":10}}"#))
            .collect();
        serde_json::from_str(&format!(
            r#"{{"status":"ready","total_requests":{total_requests},"auth_stats":{{"total_tokens_all_keys":7}},
                "engine_metrics":{{"m":{{"running_requests":1,"recent_requests":[{}]}}}}}}"#,
            recent.join(",")
        ))
        .unwrap()
    }

    fn ids(view: &HistoryView) -> Vec<&str> {
        view.requests
            .iter()
            .map(|request| request.request.request_id.as_str())
            .collect()
    }

    #[test]
    fn samples_are_throttled_and_capped() {
        let history = History::new(3, 10);
        history.record(&metrics(&[], 1), 100.0);
        history.record(&metrics(&[], 2), 100.5);
        assert_eq!(history.view(0).samples.len(), 1);
        for (offset, t) in [101.0, 102.0, 103.0].into_iter().enumerate() {
            history.record(&metrics(&[], 3 + offset as u64), t);
        }
        let view = history.view(0);
        let times: Vec<f64> = view.samples.iter().map(|sample| sample.t).collect();
        assert_eq!(times, vec![101.0, 102.0, 103.0]);
        assert_eq!(view.samples[2].total_requests, 5);
        assert_eq!(view.samples[0].models["m"].running_requests, 1);
        assert_eq!(view.samples[0].key_tokens, Some(7));
    }

    #[test]
    fn requests_are_logged_once_oldest_first() {
        let history = History::new(10, 10);
        history.record(&metrics(&["b", "a"], 1), 1.0);
        history.record(&metrics(&["c", "b", "a"], 2), 2.0);
        let view = history.view(0);
        assert_eq!(ids(&view), ["a", "b", "c"]);
        assert_eq!(view.requests[2].model, "m");
        assert!((view.requests[2].seen_at - 2.0).abs() < f64::EPSILON);
        assert!(view.requests.windows(2).all(|pair| pair[0].seq < pair[1].seq));
    }

    #[test]
    fn request_log_is_capped() {
        let history = History::new(10, 2);
        history.record(&metrics(&["c", "b", "a"], 1), 1.0);
        assert_eq!(ids(&history.view(0)), ["b", "c"]);
    }

    #[test]
    fn views_are_deltas_after_a_sequence_number() {
        let history = History::new(10, 10);
        history.record(&metrics(&["a"], 1), 1.0);
        let first = history.view(0);
        assert!(!first.reset);
        assert_eq!((first.samples.len(), first.requests.len()), (1, 1));

        history.record(&metrics(&["b", "a"], 2), 2.0);
        let delta = history.view(first.seq);
        assert_eq!(delta.samples.len(), 1);
        assert_eq!(ids(&delta), ["b"]);
        assert!(history.view(delta.seq).samples.is_empty());

        let ahead = history.view(delta.seq + 100);
        assert!(ahead.reset);
        assert_eq!((ahead.samples.len(), ahead.requests.len()), (2, 2));
    }
}
