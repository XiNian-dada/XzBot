use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde_json::Value;

const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";

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
    Ok(format!(
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
    ))
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
