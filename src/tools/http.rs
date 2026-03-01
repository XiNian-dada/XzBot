use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::{Client, Proxy};

use crate::config::NetworkConfig;

pub fn build_client(timeout_ms: u64, network: &NetworkConfig, cookies: bool) -> Result<Client> {
    let mut builder = Client::builder().timeout(Duration::from_millis(timeout_ms));
    if cookies {
        builder = builder.cookie_store(true);
    }
    if network.proxy_enabled {
        let proxy = Proxy::all(network.proxy_url.trim()).context("failed to parse proxy url")?;
        builder = builder.proxy(proxy);
    }
    builder
        .build()
        .context("failed to build HTTP client with proxy")
}
