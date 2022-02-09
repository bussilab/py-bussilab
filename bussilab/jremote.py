"""
Module implementing tools for remote jupyter connections.

This module contains some utilities that are mostly designed
to be used as command line tools. The interface of the functions defined in
this module is subject to changes. One should instead use the subcommands
`jrun` and `jremote` in the [cli_documentation](cli_documentation.html)
submodule.
"""
import socket
import os
from contextlib import closing
import sys
import subprocess
import re
import platform
import time

def find_free_port():
    """Returns the number of a free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def _adjust_sockname(sockname,port):
    pwd=os.getcwd()
    pwd=re.sub("[^-A-Za-z0-9.]",":",pwd)
    sockname=re.sub(r"\(path\)",pwd,sockname)
    sockname=re.sub(r"\(port\)",str(port),sockname)
    return sockname

def run_server(dry_run: bool = False,
               port: int = 0,
               screen_cmd: str = "screen",
               screen_log: str = "",
               no_screen: bool = False,
               keep_ld_library_path: bool = True,
               python_exec: str = "",
               sockname: str = "(path):jupyter-server:(port)",
               detach: bool = False):
    """Runs a jupyter server inside a screen command.

       This function is only designed to be used as a command line
       tool.
    """
    if python_exec == "":
        python_exec = sys.executable
    print("python_exec:", python_exec)
    if port == 0:
        port = find_free_port()
    print("port:", port)
    cmd = []
    if not no_screen:
        cmd.extend(screen_cmd.split()) # allows screen_cmd to contain space separated options
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
            cmd.append("env")
            cmd.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
    cmd.extend(python_exec.split()) # allows python_exec to contain space separated options
    cmd.append("-m")
    cmd.append("jupyter")
    cmd.append("notebook")
    cmd.append("--no-browser")
    cmd.append("--port=" + str(port))
    print("cmd:", cmd)
    if dry_run:
        return
    subprocess.call(cmd)

def remote(server: str,
           python_exec: str = "python",
           dry_run: bool = False,
           list_only: bool = False,
           port: int = 0,
           index: int = 0,
           open_cmd: str = ""):

    cmd = python_exec+" -m jupyter notebook list"

    print("server:", server)
    print("cmd:", cmd)

    if dry_run and list_only:
        return

    ll = []
    ll_localhost = []
    args = ['ssh', server, cmd]

    for l in subprocess.run(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            check=True).stdout.split('\n'):
        if re.match("^http", l):
            ll.append(re.sub("localhost", server, l))
            ll_localhost.append(l)

    for i in range(len(ll)):
        print(str(i+1)+") "+ll[i])

    if list_only:
        return

    if len(ll_localhost) == 1:
        index = 1

    if index == 0:
        sys.stdout.write("Choose a notebook or interrupt (^c):")
        sys.stdout.flush()
        index = int(sys.stdin.readline())

    url = ll_localhost[index-1].split()[0]

    print("Chosen url:", url)

    if not re.match(r"^http://localhost:[0-9]*/\?token=[0-9a-f]*$", url):
        raise Exception("URL "+url+" looks incorrectly formatted")

    server_port = re.sub("^http://localhost:", "", url)
    server_port = re.sub("/.*$", "", server_port)

    if port == 0:
        port = find_free_port()

    ssh_cmd = ["ssh", "-NL", str(port) + ":localhost:" + str(server_port), server]

    xopen_cmd = [] # list rather than string

    if open_cmd == "":
        plat = platform.system()
        if plat == "Darwin":
            xopen_cmd.append("open")
        elif plat == "Linux":
            xopen_cmd.append("xdg-open")
        else:
            raise Exception("Unknown platform " + plat)
    else:
        xopen_cmd = [open_cmd]

    xopen_cmd.append(re.sub(":" + str(server_port), ":" + str(port), url))
    print("open_cmd:", ' '.join(xopen_cmd))
    print("ssh:", ' '.join(ssh_cmd))
    if dry_run:
        print(xopen_cmd)
    else:
        ssh = subprocess.Popen(ssh_cmd)
        try:
            print("Don't kill this terminal or you will be disconnected!")
            time.sleep(1)
            subprocess.call(xopen_cmd)
            print("Use CTRL+c to kill the connection")
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            try:
                ssh.communicate()
            except KeyboardInterrupt:
                print("Interrupted")
