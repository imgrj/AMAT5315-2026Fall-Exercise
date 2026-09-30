#[test]
fn test_cli_execution() {
    if let Ok(cwd) = std::env::var("SEISMIC_CWD") {
        if !cwd.trim().is_empty() {
            let _ = std::env::set_current_dir(std::path::Path::new(&cwd));
        }
    }

    let raw = std::env::var("SEISMIC_ARGS").unwrap_or_default();
    if raw.trim().is_empty() {
        return;
    }

    let mut args = vec!["seismic".to_string()];
    let mut cur = String::new();
    let mut in_quotes = false;
    for c in raw.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ' ' | '\t' if !in_quotes => {
                if !cur.is_empty() {
                    args.push(cur.clone());
                    cur.clear();
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        args.push(cur);
    }

    seismic::runner::run_cli(&args).expect("run_cli execution failed");
}
