use anyhow::{bail, Context, Result};
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;

const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";
const BROWSER_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const PREFERRED_WEATHER_DOMAIN: &str = "tianqi.2345.com";
const MAX_WEATHER_REF_CHARS: usize = 1200;

pub async fn get_weather(client: &Client, location: &str, debug: bool) -> Result<String> {
    let location = location.trim();
    if location.is_empty() {
        bail!("location is empty");
    }

    if debug {
        println!("[DEBUG] tool.get_weather location={location}");
    }

    let geo = geocode_location(client, location).await?;
    if debug {
        println!(
            "[DEBUG] weather.geocode name={} lat={} lon={} country={}",
            geo.name, geo.latitude, geo.longitude, geo.country
        );
    }

    let weather = fetch_current_weather(client, geo.latitude, geo.longitude).await?;
    let mut out = format!(
        "天气查询结果\nquery: {location}\nlocation: {} ({}, {})\ncountry: {}\ntimezone: {}\nlocal_time: {}\ncondition: {}\ntemperature: {:.1}°C\nfeels_like: {:.1}°C\nhumidity: {}%\nprecipitation: {:.1} mm\nwind: {:.1} km/h, direction {}°",
        geo.name,
        geo.latitude,
        geo.longitude,
        geo.country,
        weather.timezone,
        weather.time,
        weather_condition(weather.weather_code),
        weather.temperature,
        weather.apparent_temperature,
        weather.relative_humidity,
        weather.precipitation,
        weather.wind_speed,
        weather.wind_direction
    );

    // Keep API current-conditions as baseline, then append web-based multi-day reference.
    match fetch_weather_reference(client, location, debug).await {
        Ok(reference) => {
            out.push_str("\n\n");
            out.push_str(&reference);
        }
        Err(err) => {
            if debug {
                println!("[DEBUG] weather.reference failed location={} err={}", location, err);
            }
        }
    }

    Ok(out)
}

struct GeocodeResult {
    name: String,
    country: String,
    latitude: f64,
    longitude: f64,
}

async fn geocode_location(client: &Client, location: &str) -> Result<GeocodeResult> {
    let encoded = urlencoding::encode(location);
    let url = format!(
        "https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=zh&format=json"
    );
    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("weather geocoding request failed: {url}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read geocoding response body")?;
    if !status.is_success() {
        bail!("weather geocoding endpoint returned {status}: {body}");
    }

    let value: Value =
        serde_json::from_str(&body).context("failed to parse weather geocoding response JSON")?;
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("geocoding response missing results"))?;
    let first = results
        .first()
        .ok_or_else(|| anyhow::anyhow!("location not found: {location}"))?;

    let name = first
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(location)
        .to_string();
    let country = first
        .get("country")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let latitude = first
        .get("latitude")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("geocoding response missing latitude"))?;
    let longitude = first
        .get("longitude")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("geocoding response missing longitude"))?;

    Ok(GeocodeResult {
        name,
        country,
        latitude,
        longitude,
    })
}

struct CurrentWeather {
    timezone: String,
    time: String,
    weather_code: i64,
    temperature: f64,
    apparent_temperature: f64,
    relative_humidity: i64,
    precipitation: f64,
    wind_speed: f64,
    wind_direction: i64,
}

async fn fetch_current_weather(
    client: &Client,
    latitude: f64,
    longitude: f64,
) -> Result<CurrentWeather> {
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m&timezone=auto"
    );
    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("weather forecast request failed: {url}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read weather forecast response body")?;
    if !status.is_success() {
        bail!("weather forecast endpoint returned {status}: {body}");
    }

    let value: Value =
        serde_json::from_str(&body).context("failed to parse weather forecast response JSON")?;
    let current = value
        .get("current")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("weather response missing current"))?;

    let timezone = value
        .get("timezone")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let time = current
        .get("time")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let weather_code = current
        .get("weather_code")
        .and_then(Value::as_i64)
        .unwrap_or(-1);
    let temperature = current
        .get("temperature_2m")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("weather response missing temperature_2m"))?;
    let apparent_temperature = current
        .get("apparent_temperature")
        .and_then(Value::as_f64)
        .unwrap_or(temperature);
    let relative_humidity = current
        .get("relative_humidity_2m")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let precipitation = current
        .get("precipitation")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let wind_speed = current
        .get("wind_speed_10m")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let wind_direction = current
        .get("wind_direction_10m")
        .and_then(Value::as_i64)
        .unwrap_or(0);

    Ok(CurrentWeather {
        timezone,
        time,
        weather_code,
        temperature,
        apparent_temperature,
        relative_humidity,
        precipitation,
        wind_speed,
        wind_direction,
    })
}

fn weather_condition(code: i64) -> &'static str {
    match code {
        0 => "晴朗",
        1 | 2 => "晴间多云",
        3 => "阴天",
        45 | 48 => "雾",
        51 | 53 | 55 => "毛毛雨",
        56 | 57 => "冻毛毛雨",
        61 | 63 | 65 => "雨",
        66 | 67 => "冻雨",
        71 | 73 | 75 => "雪",
        77 => "雪粒",
        80 | 81 | 82 => "阵雨",
        85 | 86 => "阵雪",
        95 => "雷暴",
        96 | 99 => "雷暴伴冰雹",
        _ => "未知天气",
    }
}

#[derive(Clone)]
struct WeatherCandidate {
    title: String,
    url: String,
}

async fn fetch_weather_reference(client: &Client, location: &str, debug: bool) -> Result<String> {
    let query = format!("{location} 天气 15天 30天");
    let encoded = urlencoding::encode(&query);
    let search_url = format!("https://cn.bing.com/search?q={encoded}&setlang=zh-Hans&cc=CN");

    let response = client
        .get(&search_url)
        .header("User-Agent", BROWSER_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .header("Referer", "https://cn.bing.com/")
        .send()
        .await
        .with_context(|| format!("weather web search request failed: {search_url}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read weather web search response body")?;
    if !status.is_success() {
        bail!("weather web search returned {status}");
    }

    let candidates = parse_bing_weather_candidates(&body)?;
    if candidates.is_empty() {
        bail!("weather web search has no candidates");
    }

    // Prefer tianqi.2345.com for detailed multi-day forecasts; fallback to next best hit.
    let picked = pick_weather_candidate(candidates).ok_or_else(|| {
        anyhow::anyhow!("weather web search candidates have no valid result url")
    })?;
    let preferred = is_preferred_weather_domain(&picked.url);
    let (page_title, page_text) = fetch_weather_page_summary(client, &picked.url).await?;

    if debug {
        println!(
            "[DEBUG] weather.reference picked preferred={} url={}",
            preferred, picked.url
        );
    }

    let source = if preferred {
        format!("source: {}", PREFERRED_WEATHER_DOMAIN)
    } else {
        let host = host_of(&picked.url).unwrap_or_else(|| "unknown".to_string());
        format!("source: {} (fallback)", host)
    };

    Ok(format!(
        "多日天气参考\nquery: {query}\n{source}\nurl: {}\nresult_title: {}\npage_title: {}\ncontent: {}",
        picked.url,
        picked.title,
        page_title,
        page_text
    ))
}

fn parse_bing_weather_candidates(html: &str) -> Result<Vec<WeatherCandidate>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("li.b_algo")
        .map_err(|err| anyhow::anyhow!("failed to parse selector li.b_algo: {err}"))?;
    let title_sel =
        Selector::parse("h2 a").map_err(|err| anyhow::anyhow!("failed to parse selector h2 a: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(12) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };
        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .unwrap_or("")
            .replace("&amp;", "&")
            .replace("&#38;", "&");
        let href = normalize_weather_result_url(&raw_href);
        if title.is_empty() || href.is_empty() {
            continue;
        }
        out.push(WeatherCandidate { title, url: href });
    }

    Ok(out)
}

fn normalize_weather_result_url(url: &str) -> String {
    let trimmed = url.trim();
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return String::new();
    }
    let Ok(parsed) = reqwest::Url::parse(trimmed) else {
        return String::new();
    };
    let Some(host) = parsed.host_str() else {
        return String::new();
    };
    let lower_host = host.to_ascii_lowercase();
    if lower_host.ends_with("bing.com") || lower_host == "go.microsoft.com" {
        return String::new();
    }
    trimmed.to_string()
}

fn pick_weather_candidate(candidates: Vec<WeatherCandidate>) -> Option<WeatherCandidate> {
    if let Some(hit) = candidates
        .iter()
        .find(|c| is_preferred_weather_domain(&c.url))
        .cloned()
    {
        return Some(hit);
    }
    candidates.into_iter().next()
}

fn is_preferred_weather_domain(url: &str) -> bool {
    let Some(host) = host_of(url) else {
        return false;
    };
    host == PREFERRED_WEATHER_DOMAIN || host.ends_with(&format!(".{PREFERRED_WEATHER_DOMAIN}"))
}

fn host_of(url: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|v| v.to_ascii_lowercase()))
}

async fn fetch_weather_page_summary(client: &Client, url: &str) -> Result<(String, String)> {
    let response = client
        .get(url)
        .header("User-Agent", BROWSER_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .send()
        .await
        .with_context(|| format!("weather reference fetch failed: {url}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read weather reference body")?;
    if !status.is_success() {
        bail!("weather reference page returned {status}");
    }

    let (title, text) = extract_html_title_text(&body)?;
    Ok((title, text))
}

fn extract_html_title_text(html: &str) -> Result<(String, String)> {
    let doc = Html::parse_document(html);
    let body_sel =
        Selector::parse("body").map_err(|err| anyhow::anyhow!("failed to parse selector body: {err}"))?;
    let title_sel =
        Selector::parse("title").map_err(|err| anyhow::anyhow!("failed to parse selector title: {err}"))?;

    let title = doc
        .select(&title_sel)
        .next()
        .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "(no title)".to_string());

    let body_text = doc
        .select(&body_sel)
        .next()
        .map(|n| n.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();

    let normalized = normalize_whitespace(&body_text);
    let truncated = truncate_text(&normalized, MAX_WEATHER_REF_CHARS);
    Ok((title, truncated))
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "...(truncated)"
}
