# hms-mcp — LLM-driven forge editing for Halo Map Studio

An [MCP](https://modelcontextprotocol.io) (Model Context Protocol) stdio server that lets an LLM
agent drive the HMS forge editor: place/move/rotate objects, set labels / spawn sequences / scale /
shape, control the camera, and place objects relative to (or snapped against) other objects.

## Architecture

```
LLM client ⇄ (JSON-RPC / stdio) ⇄ hms-mcp ⇄ (TCP frame) ⇄ hms-app cmd_server ⇄ App::run_script
```

The editor's command executor lives inside `hms-app` (`src/script.rs` + `App::execute_command`),
reachable both from the in-app **⌨ Script** panel and over a localhost TCP command server
(`hms-app/src/cmd_server.rs`). `hms-mcp` is a thin bridge that forwards MCP tool calls to it —
the editor process keeps stdio for itself and `App` state never has to be `Send`.

## Running

1. Start the editor: `hms-app.exe` (the command server binds `127.0.0.1:47800` on startup;
   override with `HMS_CMD_PORT`, disable with `HMS_CMD_PORT=0`).
2. Register `hms-mcp.exe` with your MCP client.

Claude Desktop / Claude Code `mcpServers` entry:

```json
{
  "mcpServers": {
    "hms-forge": {
      "command": "C:\\path\\to\\halo-map-studio-rust\\target\\release\\hms-mcp.exe",
      "env": { "HMS_CMD_PORT": "47800" }
    }
  }
}
```

`HMS_MCP_TARGET=host:port` overrides the whole address if the editor is elsewhere.

## Tools

| Tool | Purpose |
|---|---|
| `forge_script` | Run one or more editor commands (the full grammar). One undo step per call. |
| `list_palette` | List placeable palette objects (index + name), optional name filter. |
| `list_objects` | List scene objects with datum id + position. |
| `place_object` | Place a palette object by name/#index at an absolute world position. |

### Command grammar (`forge_script`)

```
place <name|#idx> [at X Y Z | rel <datum> DX DY DZ | onface <datum> <dir> | camera]
move <datum> DX DY DZ | moveto <datum> X Y Z | rotate <datum> <x|y|z> DEG
set <datum> <field> <value>    fields: team color label spawnseq scale cached_type respawn shape
                               shape = none|sphere|cylinder|box
select <datum|all|none> | delete [datum] | camera <to|lookat X Y Z|spawn|frame>
loadmap <name|path> | loadvariant <path> | screenshot <path|dir> [WxH] | wait
mapid [current|<mapname>|variant <path>]
list palette [f] | list maps [f] | list variants [current|<id>|map <name>|all] [f]
list objects [type <t>] [name <s>] [label <s>] | list types | get <datum> | echo <text>
foreach <variants…|maps…|objects…> as $v … end     # $v=item, $stem=file stem, $idx=index
```

## Headless batch / automation (`hms-app --script`)

The MCP server drives the **live** editor window — map loads stream over frames, so a
`loadmap → screenshot` chain there may capture mid-load. For deterministic, unattended batch
work (loads are synchronous), run the same grammar headlessly:

```
hms-app --script build.hscr        # run a script file
hms-app --exec "list maps"         # run one inline command
```

`HMS_MVAR_DIRS=<;-separated dirs>` tells the query layer where your `.mvar` library lives
(so `list variants` / `foreach variants` can find them). `HMS_SHOT_SIZE=WxH` sets the default
render size. Example — screenshot every variant of the current map:

```
# shots.hscr   →   hms-app --script shots.hscr
foreach variants all as $v
  loadvariant $v
  camera frame
  screenshot out/$stem.png 1280x720
end
```

This replaces the old `HMS_BATCH` environment-variable mode: batching is now just a script.

Directions: `+x -x +y -y +z -z` (aliases `north/south/east/west/up/down`). Datums are hex
(`0xD8000001`) from `list objects`. Lines starting with `#` or `//` are comments.

### Typical agent flow

```
list palette block          → discover object names/indices
place block 1x1x1 at 0 0 0  → returns a datum, e.g. 0xD8000001
list objects                → confirm datum + position
place block 1x1x1 onface 0xD8000001 +x   → snap a second block to the first's +x face
set 0xD8000002 color 1      → tint it
```
