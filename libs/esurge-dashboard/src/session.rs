//! In-memory browser login sessions: random id -> API key, with expiry.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

struct Entry {
    api_key: String,
    expires_at: Instant,
}

pub struct SessionStore {
    ttl: Duration,
    sessions: Mutex<HashMap<String, Entry>>,
}

impl SessionStore {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Entry>> {
        self.sessions.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Store `api_key` under a fresh 256-bit random id and return the id. Expired sessions are pruned.
    pub fn create(&self, api_key: String) -> Result<String, String> {
        let mut bytes = [0_u8; 32];
        getrandom::fill(&mut bytes).map_err(|error| format!("no randomness for a session id: {error}"))?;
        let id: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        let now = Instant::now();
        let expires_at = now.checked_add(self.ttl).unwrap_or(now);
        let mut sessions = self.lock();
        sessions.retain(|_, entry| entry.expires_at > now);
        sessions.insert(id.clone(), Entry { api_key, expires_at });
        Ok(id)
    }

    /// API key of a live session; an expired session is removed and treated as absent.
    pub fn api_key(&self, id: &str) -> Option<String> {
        let now = Instant::now();
        let mut sessions = self.lock();
        match sessions.get(id) {
            Some(entry) if entry.expires_at > now => Some(entry.api_key.clone()),
            Some(_) => {
                sessions.remove(id);
                None
            }
            None => None,
        }
    }

    pub fn remove(&self, id: &str) {
        self.lock().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_lookup_and_remove() {
        let store = SessionStore::new(Duration::from_secs(60));
        let id = store.create("sk-key".into()).unwrap();
        assert_eq!(id.len(), 64);
        assert_eq!(store.api_key(&id).as_deref(), Some("sk-key"));
        assert_ne!(store.create("sk-key".into()).unwrap(), id);
        store.remove(&id);
        assert_eq!(store.api_key(&id), None);
        assert_eq!(store.api_key("unknown"), None);
    }

    #[test]
    fn expired_sessions_are_absent() {
        let store = SessionStore::new(Duration::ZERO);
        let id = store.create("sk-key".into()).unwrap();
        assert_eq!(store.api_key(&id), None);
    }
}
