//! Client for the API server's `GET /metrics`.

use std::time::Duration;

use axum::body::Bytes;
use axum::http::{header, StatusCode};
use reqwest::{redirect, Client, Url};

pub struct Upstream {
    client: Client,
    metrics_url: Url,
}

pub struct UpstreamResponse {
    pub status: StatusCode,
    pub body: Bytes,
}

impl Upstream {
    /// `base` is the API server's root URL, e.g. `http://10.0.0.1:8000`.
    pub fn new(base: &str, timeout: Duration) -> Result<Self, String> {
        let mut url = Url::parse(base).map_err(|error| format!("invalid upstream URL {base:?}: {error}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!("upstream URL {base:?} must use http or https"));
        }
        if url.query().is_some() || url.fragment().is_some() || !url.username().is_empty() || url.password().is_some() {
            return Err(format!(
                "upstream URL {base:?} must not carry credentials, a query or a fragment"
            ));
        }
        let path = format!("{}/metrics", url.path().trim_end_matches('/'));
        url.set_path(&path);
        let client = Client::builder()
            .timeout(timeout)
            .redirect(redirect::Policy::none())
            .build()
            .map_err(|error| format!("cannot build HTTP client: {error}"))?;
        Ok(Self {
            client,
            metrics_url: url,
        })
    }

    pub fn metrics_url(&self) -> &Url {
        &self.metrics_url
    }

    /// Fetch `/metrics` as `api_key`. `Err` means the server could not be reached or the body not read.
    pub async fn fetch(&self, api_key: &str) -> Result<UpstreamResponse, String> {
        let response = self
            .client
            .get(self.metrics_url.clone())
            .bearer_auth(api_key)
            .header(header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(|error| format!("API server unreachable: {}", error.without_url()))?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|error| format!("reading API server response: {error}"))?;
        Ok(UpstreamResponse { status, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_url_is_derived_from_the_base() {
        let timeout = Duration::from_secs(1);
        assert_eq!(
            Upstream::new("http://10.0.0.1:8000", timeout)
                .unwrap()
                .metrics_url()
                .as_str(),
            "http://10.0.0.1:8000/metrics"
        );
        assert_eq!(
            Upstream::new("http://h:8000/api/", timeout)
                .unwrap()
                .metrics_url()
                .as_str(),
            "http://h:8000/api/metrics"
        );
        assert!(Upstream::new("ftp://h", timeout).is_err());
        assert!(Upstream::new("http://h:8000/?api_key=x", timeout).is_err());
        assert!(Upstream::new("http://user:pw@h:8000", timeout).is_err());
        assert!(Upstream::new("not a url", timeout).is_err());
    }
}
