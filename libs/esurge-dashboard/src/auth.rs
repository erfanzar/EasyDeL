//! Where a request's API key comes from. Keys are read from headers or a login session, never the query string.
//!
//! The dashboard does not validate keys itself: it forwards them to the API server, which compares them.

use axum::http::{header, HeaderMap};
use cookie::{Cookie, SameSite};

pub const SESSION_COOKIE: &str = "esurge_session";

/// API key sent on the request itself: `Authorization: Bearer <key>` (any scheme case) or `X-API-Key: <key>`.
pub fn presented_key(headers: &HeaderMap) -> Option<String> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let (scheme, token) = value.trim().split_once(' ')?;
            let token = token.trim();
            (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then(|| token.to_string())
        });
    bearer.or_else(|| {
        headers
            .get("x-api-key")
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

/// Login-session id from the `Cookie` headers, if any.
pub fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(Cookie::split_parse)
        .filter_map(Result::ok)
        .find(|cookie| cookie.name() == SESSION_COOKIE && !cookie.value().is_empty())
        .map(|cookie| cookie.value().to_string())
}

/// `Set-Cookie` value for a new login session.
pub fn session_cookie(id: &str, ttl_seconds: u64, secure: bool) -> String {
    let max_age = cookie::time::Duration::seconds(i64::try_from(ttl_seconds).unwrap_or(i64::MAX));
    Cookie::build((SESSION_COOKIE, id.to_string()))
        .http_only(true)
        .same_site(SameSite::Strict)
        .path("/")
        .max_age(max_age)
        .secure(secure)
        .build()
        .to_string()
}

/// `Set-Cookie` value that removes the login session cookie.
pub fn clear_session_cookie(secure: bool) -> String {
    session_cookie("", 0, secure)
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(*name, HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn bearer_scheme_and_header_names_are_case_insensitive_but_key_is_exact() {
        assert_eq!(
            presented_key(&headers(&[("authorization", "Bearer sk-AbC")])).as_deref(),
            Some("sk-AbC")
        );
        assert_eq!(
            presented_key(&headers(&[("Authorization", "bearer sk-AbC")])).as_deref(),
            Some("sk-AbC")
        );
        assert_eq!(
            presented_key(&headers(&[("AUTHORIZATION", "BEARER  sk-AbC ")])).as_deref(),
            Some("sk-AbC")
        );
        assert_eq!(
            presented_key(&headers(&[("X-API-Key", "sk-AbC")])).as_deref(),
            Some("sk-AbC")
        );
        assert_eq!(
            presented_key(&headers(&[("x-api-key", " sk-AbC ")])).as_deref(),
            Some("sk-AbC")
        );
    }

    #[test]
    fn missing_empty_or_foreign_credentials_yield_none() {
        assert_eq!(presented_key(&HeaderMap::new()), None);
        assert_eq!(
            presented_key(&headers(&[("authorization", "Basic dXNlcjpwYXNz")])),
            None
        );
        assert_eq!(presented_key(&headers(&[("authorization", "Bearer ")])), None);
        assert_eq!(presented_key(&headers(&[("x-api-key", "  ")])), None);
    }

    #[test]
    fn session_cookie_is_found_among_other_cookies() {
        let map = headers(&[("cookie", "theme=dark"), ("cookie", "a=1; esurge_session=abc123; b=2")]);
        assert_eq!(session_id(&map).as_deref(), Some("abc123"));
        assert_eq!(session_id(&headers(&[("cookie", "esurge_session=")])), None);
        assert_eq!(session_id(&headers(&[("cookie", "other=abc")])), None);
    }

    #[test]
    fn cookies_are_http_only_strict_and_clearable() {
        let set = session_cookie("abc", 60, false);
        assert!(set.starts_with("esurge_session=abc"));
        assert!(set.contains("HttpOnly") && set.contains("SameSite=Strict") && set.contains("Max-Age=60"));
        assert!(!set.contains("Secure"));
        assert!(session_cookie("abc", 60, true).contains("Secure"));
        assert!(clear_session_cookie(false).contains("Max-Age=0"));
    }
}
