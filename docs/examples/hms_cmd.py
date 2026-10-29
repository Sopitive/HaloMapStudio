#!/usr/bin/env python3
"""Run film verbs against a RUNNING Halo Map Studio (its TCP command server) and print the result.

    python3 -I hms_cmd.py /path/to/match.mov "film objects label=zone_a"
    python3 -I hms_cmd.py /path/to/match.mov "film inside label=zone_a" > inside.csv

Opens the film, starts `film index`, waits until the recording is decoded, then runs each
command given and prints its output. Port: HMS_CMD_PORT (default 47800). Paths with spaces are
not supported by the script grammar (no quoting) - copy the film to a simple path first.
"""
import os, socket, sys, time

HOST, PORT = "127.0.0.1", int(os.environ.get("HMS_CMD_PORT", "47800"))

def run(sock, script):
    body = script.encode()
    sock.sendall(b"%d\n" % len(body) + body)          # request: <byte count>\n<script>
    header = b""
    while not header.endswith(b"\n"):
        header += sock.recv(1)
    want, data = int(header), b""
    while len(data) < want:
        data += sock.recv(want - len(data))
    return data.decode()                               # reply: <byte count>\n<log>

def main(film, *cmds):
    with socket.create_connection((HOST, PORT)) as s:
        out = run(s, f"film {film}")
        if "no such file" in out or out.startswith("ERROR"):
            sys.exit(out)
        run(s, "film index")                           # returns at once; decoding runs on a worker
        for _ in range(600):                           # wait up to ~5 minutes
            if "not decoded yet" not in run(s, "film seats"):
                break
            time.sleep(0.5)
        else:
            sys.exit("the recording did not finish decoding")
        for c in cmds:
            sys.stdout.write(run(s, c))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
