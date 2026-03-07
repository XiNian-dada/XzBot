//! 工具模块入口：聚合搜索、天气、系统信息等函数调用能力。

/// Shared HTTP client construction helpers.
pub mod http;
/// Host and process runtime inspection tools.
pub mod system;
/// Weather query tool and multi-day reference extraction.
pub mod weather;
/// Web search and URL fetch tools.
pub mod web;
