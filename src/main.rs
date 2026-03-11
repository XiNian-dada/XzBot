//! 二进制入口：仅负责声明模块并启动应用运行时。

mod app;
mod bot;
mod config;
mod llm;
mod logger;
mod onebot;
mod plugins;
mod post_api;
mod store;
mod token_stats;
mod tools;

#[tokio::main]
async fn main() {
    if let Err(err) = app::run().await {
        logger::error_err("application runtime failed", &err);
        std::process::exit(1);
    }
}
