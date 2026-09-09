use std::process::ExitCode;

use md::cli::{self, Command};

fn usage() {
    eprintln!("usage: md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01]");
    eprintln!("              [--eq-steps 2000] [--steps 10000] [--sample-every 50]");
    eprintln!("              [--seed 2026] [--force cells|naive] [--ramp-to T] [--out artifacts]");
    eprintln!("       md check artifacts");
    eprintln!("       md video artifacts --out run.mp4");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("{}", md::greeting());
        return ExitCode::SUCCESS;
    }
    let command = match cli::parse(&args) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("error: {error}");
            usage();
            return ExitCode::from(2);
        }
    };
    let result = match command {
        Command::Run(cfg) => md::ops::run_sim(&cfg),
        Command::Check(cfg) => md::ops::check(&cfg).map(|report| {
            print!("{}", md::ops::format_report(&report, &cfg));
            if report.pass { String::new() } else { "__FAIL__".into() }
        }),
        Command::Video(cfg) => md::ops::make_video(&cfg),
    };
    match result {
        Ok(message) if message == "__FAIL__" => ExitCode::FAILURE,
        Ok(message) => {
            if !message.is_empty() {
                println!("{message}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
