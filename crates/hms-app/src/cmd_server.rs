//! TCP command server: exposes the editor command executor (`App::run_script`) to external
//! processes — the `hms-mcp` bridge (an MCP server that drives the editor), plus any scripting
//! client. Runs on a background thread; because `App` state is not `Send`, each request is handed
//! to the UI thread over a channel and executed there (drained in `App::update`).
//!
//! Wire protocol (BOTH directions), framed so multi-line scripts pass cleanly:
//!   `<decimal-byte-count>\n<utf8 body>`
//! A request body is script text (one or more command lines); the response body is the executor's
//! output log. A connection is keep-alive: it may carry many request/response pairs.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::mpsc::{Receiver, Sender};

/// A command awaiting execution on the UI thread. `reply` carries the output log back.
pub struct CmdRequest {
    pub text: String,
    pub reply: Sender<String>,
}

/// Bind the command server and return the receiver the UI thread drains each frame.
/// Port: env `HMS_CMD_PORT` (default 47800); set to `0` to disable. Binds 127.0.0.1 only.
pub fn spawn() -> Option<Receiver<CmdRequest>> {
    let port: u16 = std::env::var("HMS_CMD_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(47800);
    if port == 0 {
        return None;
    }
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            log::warn!("cmd server: bind 127.0.0.1:{port} failed ({e}); scripting-over-TCP disabled");
            return None;
        }
    };
    log::info!("cmd server listening on 127.0.0.1:{port} (override with HMS_CMD_PORT, 0=off)");
    let (tx, rx) = std::sync::mpsc::channel::<CmdRequest>();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(_) => continue,
            };
            let tx = tx.clone();
            std::thread::spawn(move || {
                let _ = handle_conn(stream, tx);
            });
        }
    });
    Some(rx)
}

fn read_frame<R: BufRead>(r: &mut R) -> std::io::Result<Option<String>> {
    let mut header = String::new();
    if r.read_line(&mut header)? == 0 {
        return Ok(None); // clean EOF
    }
    let n: usize = header
        .trim()
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad frame length"))?;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn write_frame<W: Write>(w: &mut W, body: &str) -> std::io::Result<()> {
    write!(w, "{}\n", body.len())?;
    w.write_all(body.as_bytes())?;
    w.flush()
}

fn handle_conn(stream: std::net::TcpStream, tx: Sender<CmdRequest>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    loop {
        let text = match read_frame(&mut reader)? {
            Some(t) => t,
            None => return Ok(()), // client closed
        };
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<String>();
        if tx.send(CmdRequest { text, reply: reply_tx }).is_err() {
            let _ = write_frame(&mut writer, "ERROR: editor shutting down");
            return Ok(());
        }
        // Wait for the UI thread to execute it; timeout guards a hung/busy window.
        let out = match reply_rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(o) => o,
            Err(_) => "ERROR: editor did not respond within 30s (is the window busy loading?)".into(),
        };
        write_frame(&mut writer, &out)?;
    }
}
