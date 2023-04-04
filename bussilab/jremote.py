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
import shlex
import hashlib

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
    cmd.extend(shlex.split(python_exec)) # allows python_exec to contain space separated options
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
    subprocess.call(cmd)

def remote(server: str,
           python_exec: str = "python",
           dry_run: bool = False,
           list_only: bool = False,
           server_url: str ="",
           port: int = 0,
           index: int = 0,
           open_cmd: str = ""):

    cmd =  "( " + python_exec + " -m jupyter notebook list 2> /dev/null ) ;"
    cmd += "echo x ;"
    cmd += "( " + python_exec + " -m jupyterlab list 2>&1 ) ;" # needed for jupyter-lab list 3.6

    print("server:", server)
    print("cmd:", cmd)

    if dry_run and list_only:
        return

    if server_url == "":
        ll = []
        ll_localhost = []
        args = ['ssh', server, cmd]
        ll_type = []
        found_lab=False

        for l in subprocess.run(args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                                check=True).stdout.split('\n'):
            l=re.sub(r"^\[JupyterServerListApp\] ","",l) # needed for newer jupyter-lab list 3.6
            if re.match("^http", l):
                ll.append(re.sub("localhost", server, l))
                ll_localhost.append(l)
                if found_lab:
                    ll_type.append("(L)")
                else:
                    ll_type.append("(N)")

            if l == "x":
                found_lab = True

        for i in range(len(ll)):
            print(str(i+1)+") "+ll[i]+" "+ll_type[i])

        if list_only:
            return

        if len(ll_localhost) == 1:
            index = 1

        if index == 0:
            while True:
                sys.stdout.write("Choose a notebook or interrupt (^c):")
                sys.stdout.flush()
                iii = sys.stdin.readline()
                try:
                    index = int(iii)
                except ValueError:
                    print("Cannot parse " + iii.strip() + ", try again.")
                if index>=1 and index<=len(ll_localhost):
                    print(index)
                    break
                print("Index " + str(index) + " out of range, try again.")

        url = ll_localhost[index-1].split()[0]
    else:
        url = server_url

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
