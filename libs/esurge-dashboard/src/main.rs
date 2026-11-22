use std::process::ExitCode;

use clap::Parser;
use esurge_dashboard::Config;

#[tokio::main]
async fn main() -> ExitCode {
    match esurge_dashboard::serve(Config::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("esurge-dashboard: {error}");
            ExitCode::FAILURE
        }
    }
}
