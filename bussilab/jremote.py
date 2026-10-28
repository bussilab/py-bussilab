"""
Module implementing tools for remote jupyter connections.

This module contains some utilities that are mostly designed
to be used as command line tools. The interface of the functions defined in
this module is subject to changes. One should instead use the subcommands
`jrun` and `jremote` in the [cli_documentation](cli_documentation.html)
submodule.

Discovery uses the remote Python environment selected by `--python-exec`.
Connections are forwarded through a local loopback SSH listener; keep the
terminal open while using Jupyter. Discovery supports modern Jupyter Server
and older notebook/Lab installations.

## Changelog

- **v0.0.53**
  - Use JSON server discovery with legacy fallbacks and deduplicate results.
    Missing Lab installations no longer prevent notebook discovery.
  - Accept loopback HTTP(S) URLs with custom paths, queries and authentication
    settings. Validate server selection and handle empty results and EOF.
  - Wait for the SSH listener before opening the browser, report forwarding
    failures, and terminate and reap the tunnel on exit.
  - Report server and browser launch failures; support browser command options.
- **v0.0.42**
  - Handle the JupyterLab 3.6 server-list output format.
- **v0.0.41**
  - Add JupyterLab launching and discovery alongside notebook servers.
  - Shorten long Screen socket names using a hash suffix.
- **v0.0.38**
  - Improve handling of invalid interactive server selections.
- **v0.0.36**
  - Add an explicit server URL option to bypass server discovery.
- **v0.0.35**
  - Parse Python and Screen command options with `shlex.split()`.
- **v0.0.34**
  - Add Screen logging and preserve `LD_LIBRARY_PATH` by default.
- **v0.0.32**
  - Add optional `LD_LIBRARY_PATH` forwarding to Screen sessions.
- **v0.0.2**
  - Introduce Jupyter launching in Screen and remote connections through SSH.
"""
import socket
import os
from contextlib import closing
import sys
import subprocess
import re
import platform
import time
import shlex
import hashlib
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

def find_free_port():
    """Returns the number of a free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

_max_sockname=70
_min_sockname=50

def _adjust_sockname(sockname,port):
    pwd=os.getcwd()
    pwd=re.sub("[^-A-Za-z0-9.]",":",pwd)
    sockname=re.sub(r"\(path\)",pwd,sockname)
    sockname=re.sub(r"\(port\)",str(port),sockname)
    if len(sockname)>_max_sockname:
      m=hashlib.blake2b(digest_size=(_max_sockname-_min_sockname-1)//2)
      m.update(bytes(sockname[_min_sockname:],'utf-8'))
      sockname=sockname[:_min_sockname] + "-" + m.hexdigest()
    return sockname

def run_server(dry_run: bool = False,
               port: int = 0,
               screen_cmd: str = "screen",
               screen_log: str = "",
               no_screen: bool = False,
               keep_ld_library_path: bool = True,
               python_exec: str = "",
               sockname: str = "(path):(port):jupyter",
               lab: bool = False,
               detach: bool = False):
    """Runs a jupyter server inside a screen command.

       This function is only designed to be used as a command line
       tool.
    """
    if not 0 <= port <= 65535:
        raise ValueError("Server port must be between 0 and 65535")
    python_args = shlex.split(python_exec) if python_exec else [sys.executable]
    if python_exec == "":
        python_exec = sys.executable
    print("python_exec:", python_exec)
    if port == 0:
        port = find_free_port()
    print("port:", port)
    cmd = []
    if not no_screen:
        cmd.extend(shlex.split(screen_cmd)) # allows screen_cmd to contain space separated options
        if detach:
            cmd.append("-d")
        if screen_log != "":
            cmd.append("-L")
            if screen_log != "screenlog.0":
              cmd.append("-Logfile")
              cmd.append(screen_log)
        cmd.append("-m")
        cmd.append("-h")
        cmd.append("100000") # scrollback
        cmd.append("-S")
        cmd.append(_adjust_sockname(sockname,port))
        if keep_ld_library_path and 'LD_LIBRARY_PATH' in os.environ:
            cmd.append("/usr/bin/env")
            cmd.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
    cmd.extend(python_args) # explicit commands may contain space separated options
    cmd.append("-m")
    if lab:
        cmd.append("jupyterlab")
    else:
        cmd.append("jupyter")
        cmd.append("notebook")
    cmd.append("--no-browser")
    cmd.append("--port=" + str(port))
    print("cmd:", cmd)
    if dry_run:
        return
    subprocess.run(cmd, check=True)

def _server_url(url):
    """Validate an HTTP(S) loopback URL without restricting paths or tokens."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or
            parsed.hostname not in ("localhost", "127.0.0.1", "::1") or
            parsed.username is not None or parsed.password is not None):
        raise ValueError("Expected an HTTP(S) Jupyter URL on a loopback host")
    # Accessing port also validates its syntax and range.
    if parsed.port == 0:
        raise ValueError("Jupyter port must be positive")
    return parsed


def _list_servers(server, python_exec):
    """Read modern and classic registries, falling back to legacy text output."""
    urls = []
    successful = False
    modern_available = False

    def discover(module, json_output):
        nonlocal successful
        cmd = "( " + python_exec + " -m " + module
        if json_output:
            cmd += " --json"
        cmd += " )"
        result = subprocess.run(["ssh", "--", server, cmd],
                                capture_output=True, text=True, check=False)
        if result.returncode == 255:
            raise subprocess.CalledProcessError(255, result.args,
                                                output=result.stdout, stderr=result.stderr)
        if result.returncode:
            return False
        successful = True
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            line = re.sub(r"^\[JupyterServerListApp\] ", "", line).strip()
            if json_output:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # Ignore shell startup messages and Jupyter logs.
                if not isinstance(record, dict) or not record.get("url"):
                    continue
                url = record["url"]
                token = record.get("token")
                if token:
                    parsed = urlsplit(url)
                    query = parse_qsl(parsed.query, keep_blank_values=True)
                    if not any(key == "token" for key, _ in query):
                        url = urlunsplit(parsed._replace(
                            query=urlencode(query + [("token", token)])))
            elif line.startswith(("http://", "https://")):
                url = line.split()[0]
            else:
                continue
            if url not in urls:
                urls.append(url)
        return True

    modern_available = discover("jupyter server list", True)
    if not modern_available:
        discover("jupyter server list", False)
    if not discover("jupyter notebook list", True):
        discover("jupyter notebook list", False)
    if not modern_available:
        discover("jupyterlab list", False)
    if not successful:
        raise RuntimeError("Could not list Jupyter servers; check the remote Python environment")
    return urls


def _wait_for_tunnel(ssh, port, timeout=30, host="127.0.0.1"):
    """Wait for the local listener, checking for early SSH failure."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = ssh.poll()
        if status is not None:
            raise RuntimeError("SSH tunnel exited with status " + str(status))
        try:
            with socket.create_connection((host, port), timeout=0.2):
                if ssh.poll() is not None:
                    raise RuntimeError("SSH tunnel exited during startup")
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Timed out waiting for the SSH tunnel")


def remote(server: str,
           python_exec: str = "python",
           dry_run: bool = False,
           list_only: bool = False,
           server_url: str = "",
           port: int = 0,
           index: int = 0,
           open_cmd: str = ""):
    """Discover remote Jupyter servers and open one through an SSH tunnel.

    `python_exec` is a remote shell command and can include module-loading
    commands. Dry runs can perform discovery over SSH, but do not open a
    browser or start a tunnel. Combine with `list_only` to avoid SSH entirely.
    """
    if not 0 <= port <= 65535:
        raise ValueError("Local port must be between 0 and 65535")
    print("server:", server)
    if dry_run and list_only:
        print("Remote Python command:", python_exec)
        return
    if server_url:
        url = server_url
        if list_only:
            print(url)
            return
    else:
        urls = _list_servers(server, python_exec)
        for i, url in enumerate(urls, 1):
            print(str(i) + ") " + url)
        if list_only:
            return
        if not urls:
            raise RuntimeError("No running Jupyter servers found on " + server)
        if index == 0 and len(urls) == 1:
            index = 1
        while index == 0:
            sys.stdout.write("Choose a notebook or interrupt (^c):")
            sys.stdout.flush()
            choice = sys.stdin.readline()
            if not choice:
                raise EOFError("No Jupyter server selected")
            try:
                selected = int(choice)
            except ValueError:
                print("Enter a server number.")
                continue
            if not 1 <= selected <= len(urls):
                print("Server number out of range.")
                continue
            index = selected
        if not 1 <= index <= len(urls):
            raise ValueError("Server index must be between 1 and " + str(len(urls)))
        url = urls[index - 1]

    parsed = _server_url(url)
    server_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port == 0:
        port = find_free_port()
    destination = "[::1]" if parsed.hostname == "::1" else parsed.hostname
    bind_host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    ssh_cmd = ["ssh", "-o", "ExitOnForwardFailure=yes", "-N", "-L",
               "{}:{}:{}:{}".format(bind_host, port, destination, server_port), "--", server]
    # Preserve the hostname as well as the path, query and fragment, including
    # for HTTPS certificates issued for localhost or a particular loopback IP.
    local_url = urlunsplit(parsed._replace(netloc=destination + ":" + str(port)))
    if open_cmd:
        xopen_cmd = shlex.split(open_cmd)
    else:
        plat = platform.system()
        if plat == "Darwin":
            xopen_cmd = ["open"]
        elif plat == "Linux":
            xopen_cmd = ["xdg-open"]
        else:
            raise RuntimeError("Unknown platform " + plat + "; specify --open-cmd")
    xopen_cmd.append(local_url)
    print("open_cmd:", shlex.join(xopen_cmd))
    print("ssh:", shlex.join(ssh_cmd))
    if dry_run:
        return
    ssh = subprocess.Popen(ssh_cmd)
    try:
        print("Don't kill this terminal or you will be disconnected!")
        if parsed.hostname == "::1":
            _wait_for_tunnel(ssh, port, host="::1")
        else:
            _wait_for_tunnel(ssh, port)
        subprocess.run(xopen_cmd, check=True)
        print("Use CTRL+c to kill the connection")
        status = ssh.wait()
        if status:
            raise subprocess.CalledProcessError(status, ssh_cmd)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if ssh.poll() is None:
            ssh.terminate()
        try:
            ssh.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ssh.kill()
            ssh.wait()
