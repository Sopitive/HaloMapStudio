//! hms-mcp — a Model Context Protocol (stdio) server that lets an LLM agent drive the Halo Map
//! Studio forge editor. It is a thin bridge:
//!
//!   LLM client  ⇄  (JSON-RPC over stdio)  ⇄  hms-mcp  ⇄  (TCP frame)  ⇄  hms-app cmd_server  ⇄  App::run_script
//!
//! Running hms-mcp in the editor's own process is impossible (egui/eframe own stdio and the App
//! state is not Send), so the executor lives in hms-app behind a localhost TCP server and this
//! binary forwards tool calls to it. The whole editor command grammar is reachable through the
//! `forge_script` tool; `list_palette`/`list_objects` are discovery conveniences; `place_object`
//! is a structured shortcut.
//!
//! Config: env `HMS_CMD_PORT` (default 47800) or `HMS_MCP_TARGET=host:port` selects the editor.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let reader = stdin.lock();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                // Parse error with no id we can echo — emit a generic JSON-RPC error.
                send(&mut out, &json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                }));
                continue;
            }
        };
        // Notifications carry no id and get no response.
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        match method {
            "initialize" => {
                let client_ver = msg
                    .get("params")
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(PROTOCOL_VERSION)
                    .to_string();
                reply(&mut out, id, json!({
                    "protocolVersion": client_ver,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "hms-mcp", "version": env!("CARGO_PKG_VERSION") }
                }));
            }
            "notifications/initialized" | "initialized" => { /* no response */ }
            "ping" => reply(&mut out, id, json!({})),
            "tools/list" => reply(&mut out, id, json!({ "tools": tool_defs() })),
            "tools/call" => {
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                match handle_tool_call(&params) {
                    Ok(text) => reply(&mut out, id, tool_result(&text, false)),
                    Err(text) => reply(&mut out, id, tool_result(&text, true)),
                }
            }
            "" => { /* response/unknown with no method — ignore */ }
            other => {
                if id.is_some() {
                    reply_err(&mut out, id, -32601, &format!("method not found: {other}"));
                }
            }
        }
    }
}

/// Tool catalog advertised to the agent.
fn tool_defs() -> Value {
    let grammar = "\
One command per line ('#'/'//' = comment):\n\
  place <name|#idx> [at X Y Z | rel <datum> DX DY DZ | onface <datum> <dir> | camera]\n\
  move <datum> DX DY DZ | moveto <datum> X Y Z | rotate <datum> <x|y|z> DEG\n\
  set <datum> <field> <value>   fields: team color label spawnseq scale cached_type respawn shape\n\
                                shape = none|sphere|cylinder|box\n\
  select <datum|all|none> | delete [datum] | camera <to|lookat X Y Z|spawn|frame>\n\
  loadmap <name|path> | loadvariant <path> | screenshot <path|dir> [WxH] | wait\n\
  mapid [current|<mapname>|variant <path>]\n\
  list palette [f] | list maps [f] | list variants [current|<id>|map <name>|all] [f]\n\
  list objects [type <t>] [name <s>] [label <s>] | list types | get <datum> | echo <text>\n\
  foreach <variants…|maps…|objects…> as $v … end   ($v=item, $stem=file stem, $idx=index)\n\
Direction tokens: +x -x +y -y +z -z (aliases north/south/east/west/up/down).\n\
Datums are hex like 0xD8000001 (from 'list objects'). World units are Halo units (~0.1 m).\n\
NOTE: this MCP server drives the LIVE editor window (map loads stream over frames, so a\n\
loadmap→screenshot chain may capture mid-load). For deterministic batch capture, run the\n\
headless CLI instead: hms-app --script <file>  (loads synchronously; foreach batches variants).";
    json!([
        {
            "name": "forge_script",
            "description": format!("Run one or more Halo Map Studio forge editor commands. Returns a per-line output log. The whole run is a single undo step.\n\nGrammar:\n{grammar}\n\nTypical flow: 'list palette block' to find an object, then 'place <name> at X Y Z', then 'list objects' to get its datum, then 'set <datum> color 1' etc."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "script": { "type": "string", "description": "Newline-separated commands." }
                },
                "required": ["script"]
            }
        },
        {
            "name": "list_palette",
            "description": "List forge palette objects (index + name) that can be placed. Optional case-insensitive name filter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": { "type": "string", "description": "Substring to match object names (optional)." }
                }
            }
        },
        {
            "name": "list_objects",
            "description": "List objects currently in the scene with their datum id and position. Use the datum with move/rotate/set/delete.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "place_object",
            "description": "Place a palette object by name (substring) or #index at an absolute world position. Returns the new object's datum.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Palette object name substring, or '#<index>'." },
                    "x": { "type": "number" }, "y": { "type": "number" }, "z": { "type": "number" }
                },
                "required": ["name", "x", "y", "z"]
            }
        }
    ])
}

/// Turn a tool call into a script string, forward it to the editor, return the output log.
fn handle_tool_call(params: &Value) -> Result<String, String> {
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);
    let script = match name {
        "forge_script" => args
            .get("script")
            .and_then(|s| s.as_str())
            .ok_or("forge_script requires a 'script' string")?
            .to_string(),
        "list_palette" => {
            let f = args.get("filter").and_then(|s| s.as_str()).unwrap_or("");
            format!("list palette {f}").trim().to_string()
        }
        "list_objects" => "list objects".to_string(),
        "place_object" => {
            let obj = args.get("name").and_then(|s| s.as_str()).ok_or("place_object requires 'name'")?;
            let x = num(&args, "x")?;
            let y = num(&args, "y")?;
            let z = num(&args, "z")?;
            format!("place {obj} at {x} {y} {z}")
        }
        other => return Err(format!("unknown tool '{other}'")),
    };
    send_to_editor(&script)
}

fn num(args: &Value, key: &str) -> Result<f64, String> {
    args.get(key).and_then(|v| v.as_f64()).ok_or_else(|| format!("'{key}' must be a number"))
}

/// Connect to the editor's TCP command server, send one framed request, read the framed reply.
fn send_to_editor(script: &str) -> Result<String, String> {
    let target = std::env::var("HMS_MCP_TARGET").unwrap_or_else(|_| {
        let port = std::env::var("HMS_CMD_PORT").unwrap_or_else(|_| "47800".into());
        format!("127.0.0.1:{port}")
    });
    let mut stream = TcpStream::connect(&target)
        .map_err(|e| format!("cannot reach HMS editor at {target}: {e}. Is hms-app running?"))?;
    // Request frame: "<len>\n<body>"
    let req = format!("{}\n{}", script.len(), script);
    stream
        .write_all(req.as_bytes())
        .and_then(|_| stream.flush())
        .map_err(|e| format!("send failed: {e}"))?;
    // Response frame: "<len>\n<body>"
    let mut reader = BufReader::new(stream);
    let mut header = String::new();
    reader.read_line(&mut header).map_err(|e| format!("recv failed: {e}"))?;
    let n: usize = header.trim().parse().map_err(|_| "bad response frame from editor".to_string())?;
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf).map_err(|e| format!("recv body failed: {e}"))?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn tool_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [ { "type": "text", "text": text } ],
        "isError": is_error
    })
}

fn reply(out: &mut impl Write, id: Option<Value>, result: Value) {
    let id = id.unwrap_or(Value::Null);
    send(out, &json!({ "jsonrpc": "2.0", "id": id, "result": result }));
}

fn reply_err(out: &mut impl Write, id: Option<Value>, code: i64, message: &str) {
    let id = id.unwrap_or(Value::Null);
    send(out, &json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }));
}

fn send(out: &mut impl Write, v: &Value) {
    let s = v.to_string();
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}
