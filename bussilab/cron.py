"""
Run independent Python and Bash steps periodically, optionally inside Screen.

A running scheduler reloads its YAML configuration at scheduled events. Read or
validation failures are logged and the last valid configuration stays active;
startup requires a valid configuration. A changed period reschedules the next
event. Individual step errors, nonzero exits and timeouts are logged with the
event and step index, and remaining steps still run. Child output is inherited
by cron, including its Screen log when enabled. Detached Screen jobs retain
their existing lifecycle.

The explicit `selfupdate` step upgrades bussilab and may update dependencies in
the running environment before restarting. Use it only for environments
deliberately maintained in place. For shared or version-pinned environments,
build and validate a new environment, then restart the scheduler there.

## Changelog

- **v0.0.53**
  - Retain the last valid configuration after reload failures, with explicit
    diagnostics; validate configuration structure and positive periods.
  - Continue independent steps after errors, nonzero exits and timeouts,
    logging the failing step and traceback.
  - Fix fractional-second scheduling arithmetic while retaining local-week
    alignment and existing skip/delay expressions.
- **v0.0.42**
  - Use integer event/period division when evaluating skip/delay schedules.
- **v0.0.41**
  - Read the period from configuration when not explicitly overridden, and
    reload it during operation. Support a mapping with ``period`` and ``steps``
    alongside the original list form; rename ``jobs`` to ``steps``.
  - Resume at the next step after reboot/self-update, preserving the scheduled
    event and remaining iteration count. Add quick-start resume options.
  - Account for fractional seconds when waiting for scheduled events; improve
    event/step logging and continue after unsupported Screen reboot requests.
  - Shorten long Screen socket names with a hash suffix.
- **v0.0.35**
  - Add self-update and reboot steps using replacement Screen windows,
    preserving detached mode and checking for Screen 4.1 or newer.
  - Align scheduling to the local week and allow expressions in skip/delay.
  - Parse Screen/Python command strings with ``shlex.split()`` and respect
    iteration limits before waiting for another event.
- **v0.0.34**
  - Add Bash steps, skip/delay scheduling and ``unique`` Screen session checks.
  - Derive default socket names from configuration paths and preserve
    ``LD_LIBRARY_PATH`` by default when launching through Screen.
- **v0.0.33**
  - Add optional ``LD_LIBRARY_PATH`` forwarding to Screen sessions.
- **v0.0.11**
  - Add ``max_times`` to limit scheduled iterations and introduce cron tests.
- **v0.0.2**
  - Introduce periodic Python scripts from YAML configuration, optional Screen
    sessions, quick start, subprocess timeouts and configuration reloads.
"""

import os
import time
import sys
import datetime
import subprocess
from typing import Optional
import yaml
import re
import shlex
import json
import tempfile
from . import coretools
from . import pip
import hashlib
import math
import traceback

_screenrcfile = """
  hardstatus string 'cron server. ctrl-A ctrl-D to detach, ctrl-C to kill  %{= Kd} %{= Kd}%-w%{= Kr}[%{= KW}%n %t%{= Kr}]%{= Kd}%+w %-= %{KG} %H%{KW}|%{KY}%101`%{KW}|%D %M %d %Y%{= Kc} %C%A%{-}'
  hardstatus alwayslastline
  vbell on
  altscreen on
  defscrollback 100000
"""

def _screen_version(screen_cmd):
    cmd=shlex.split(screen_cmd)
    cmd.append("-v")
    screen_ver=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.split()[2].split(".")
    return int(screen_ver[0]),int(screen_ver[1])

def _time_to_next_event(period: int):
    now_full=time.time()
    fraction = now_full - math.floor(now_full)
    now=time.localtime(now_full)
    seconds=now.tm_wday*24*3600+now.tm_hour*3600 + now.tm_min*60 + now.tm_sec
    seconds_=seconds
    seconds = seconds%period
    # also returns predicted time for next event
    return period-seconds-fraction,seconds_+period-seconds

def _now():
    return '{}'.format(datetime.datetime.now()) + ":"

_max_sockname=70
_min_sockname=50

def _adjust_sockname(sockname,cron_file):
    path_to_cron_file=""
    if len(cron_file) > 0:
        path_to_cron_file=cron_file
    else:
        path_to_cron_file=str(coretools.config_path())
    path_to_cron_file=os.path.abspath(path_to_cron_file)
    path_to_cron_file=re.sub("[^-A-Za-z0-9.]",":",path_to_cron_file)
    sockname=re.sub(r"\(path\)",path_to_cron_file,sockname)
    if len(sockname)>_max_sockname:
      m=hashlib.blake2b(digest_size=(_max_sockname-_min_sockname-1)//2)
      m.update(bytes(sockname[_min_sockname:],'utf-8'))
      sockname=sockname[:_min_sockname] + "-" + m.hexdigest()
    return sockname

def _read_config(cron_file: str):
    if len(cron_file) > 0:
        with open(cron_file) as rc:
            config=yaml.load(rc,Loader=yaml.BaseLoader)
    else:
        config=coretools.config()
    if not isinstance(config, dict) or "cron" not in config:
        raise ValueError("configuration must contain a cron section")
    section = config["cron"]
    if isinstance(section, list):
        return {"steps": section}
    if not isinstance(section, dict):
        raise ValueError("cron must be a mapping or a list of steps")
    return section


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(name + " must be a positive integer")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(name + " must be a positive integer") from exc
    if result <= 0:
        raise ValueError(name + " must be a positive integer")
    return result


def _load_config(cron_file, period):
    config = dict(_read_config(cron_file))
    if not isinstance(config.get("steps"), list):
        raise ValueError("cron.steps must be a list")
    config["period"] = _positive_integer(
        config.get("period", 3600) if period is None else period, "period")
    return config


def _reload_config(cron_file, period, previous=None):
    try:
        return _load_config(cron_file, period)
    except Exception as exc:
        source = cron_file or str(coretools.config_path())
        action = ("keeping the last valid configuration" if previous is not None
                  else "cannot start without a valid configuration")
        print(_now(), "ERROR reading cron configuration {!r}: {}: {}; {}".format(
            source, type(exc).__name__, exc, action), flush=True)
        if previous is None:
            raise
        return previous


def _find_period(cron_file: str, period):
    return _load_config(cron_file, period)["period"]


def _run_step(step, period, event, counter, index):
    if isinstance(step, str):
        c = {"type": "python", "script": step}
    elif isinstance(step, dict):
        c = dict(step)
        c.setdefault("type", "python")
    else:
        raise ValueError("step must be a script string or a mapping")

    if "delay" in c and "skip" not in c:
        raise ValueError("delay can only be used with skip")
    if "skip" in c:
        # Preserve arithmetic expressions supported by existing cron files.
        skip = eval(c["skip"])
        delay = eval(c["delay"]) if "delay" in c else 0
        if (event // period) % skip != delay % skip:
            return None

    kind = c["type"]
    if kind == "python":
        args = [sys.executable, "-c"]
    elif kind == "bash":
        args = ["bash", "--noprofile", "--norc", "-c"]
    elif kind in ("selfupdate", "reboot"):
        if kind == "selfupdate":
            timeout = _time_to_next_event(period)[0] / 2
            pip.upgrade_self(timeout=timeout)
        return _reboot(iterations=counter, skip_steps=index+1, event=event)
    else:
        raise ValueError("Unknown type " + str(kind))

    args.append(c["script"])
    timeout = _time_to_next_event(period)[0] / 2
    print(_now(), "step {} with timeout {:.3f}".format(index, timeout), flush=True)
    # Keep inherited output and existing Screen/detached-job behavior.
    subprocess.run(args, timeout=timeout, check=True)
    return None


def _run(cron_file: str,
         period: int,
         event: int,
         counter: int,
         skip_steps: int = 0,
         *, config=None):
    print(_now(), "Running now (event={})".format(event), flush=True)
    if config is None:
        config = _reload_config(cron_file, period)
    for index in range(skip_steps, len(config["steps"])):
        try:
            result = _run_step(config["steps"][index], period, event, counter, index)
            if isinstance(result, _reboot_now):
                return result
        except Exception as exc:
            print(_now(), "ERROR event={} step={}: {}: {}; continuing to next step".format(
                event, index, type(exc).__name__, exc), flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    return None

class _reboot_now():
    pass

def _reboot(*,
           iterations=0,
           skip_steps=0,
           event=0):
    if "BUSSILAB_CRON_SCREEN_ARGS" in os.environ:
        print(_now(),"environment:",os.environ["BUSSILAB_CRON_SCREEN_ARGS"])
        env = json.loads(os.environ["BUSSILAB_CRON_SCREEN_ARGS"])
        args = env["arguments"]
        args["no_screen"]=False # reboots should be done with screen, which is switched off by default
        args["quick_start"]=True
        args["quick_start_skip_steps"]=skip_steps
        args["quick_start_event"]=event
        args["window"]=True # open in a new window
        # fix number of times counting iterations already done
        if "max_times" in args:
            if args["max_times"] is not None:
                args["max_times"]-=iterations


        screen_ver=_screen_version(args["screen_cmd"])
        if screen_ver < (4,1):
            print("screen version",screen_ver,"does not support reboot")
            return

        # move this process to another window
        cmd=shlex.split(args["screen_cmd"])
        cmd.extend(["-X","number","20"]) # some high number, to make space for the new window
        ret=subprocess.call(cmd)
        if ret!=0:
            raise RuntimeError("error changing window number")

        # run a new screen in the first window
        cron(**args)

        # trigger a clean exit
        return _reboot_now()
    else:
        raise RuntimeError("selfupdate only allowed in screen instances")

def cron(*,
         quick_start: bool = False,
         quick_start_skip_steps: int = 0,
         quick_start_event: int = 0,
         cron_file: str = "",
         screen_cmd: str = "screen",
         screen_log: str = "",
         no_screen: bool = True,
         keep_ld_library_path: bool = True,
         sockname: str = "(path):cron",
         python_exec: str = "",
         detach: bool = False,
         period: Optional[int] = None,
         max_times: Optional[int] = None,
         unique: bool = False,
         window: bool = False
         ):
    if not quick_start and quick_start_skip_steps>0:
        raise RuntimeError("quick_start_skip_steps can only be used with quick_start")
    if not quick_start and quick_start_event>0:
        raise RuntimeError("quick_start_event can only be used with quick_start")
    if no_screen:
        if "BUSSILAB_CRON_SCREEN_ARGS" in os.environ:
            env = json.loads(os.environ["BUSSILAB_CRON_SCREEN_ARGS"])
            if "rcfile" in env:
               os.unlink(env["rcfile"])
        if unique:
            raise RuntimeError("unique can only be used in screen mode")
        if window:
            raise RuntimeError("window can only be used in screen mode")
        print(_now(),"start")
        if max_times is not None:
            print(_now(),"remaining iterations:",max_times)
        config = _reload_config(cron_file, period)
        counter=0
        if quick_start:
            print(_now(),"quick starting")
            if quick_start_skip_steps>0:
              print(_now(),"skipping",quick_start_skip_steps,"steps")
            r=_run(cron_file,config["period"],quick_start_event,counter,
                   quick_start_skip_steps,config=config)
            if isinstance(r,_reboot_now):
                print("exit now")
                return
            counter += 1
            if max_times is not None:
                if counter >= max_times:
                    print("maximum iterations done")
                    return
        while True:
            if max_times is not None:
                if counter >= max_times:
                    return
            scheduled_period = config["period"]
            s=_time_to_next_event(scheduled_period)
            print(_now(),"Waiting " +"{:.3f}".format(s[0])+ " seconds for next scheduled event")
            time.sleep(s[0])
            config = _reload_config(cron_file, period, config)
            if config["period"] != scheduled_period:
                print(_now(), "period changed; rescheduling with period", config["period"],
                      flush=True)
                continue
            r=_run(cron_file,config["period"],s[1],counter,config=config)
            if isinstance(r,_reboot_now):
                return
            counter += 1
    else:

        # this dictionary is passed as an environment variable
        # it has two purposes:
        # - passing the arguments to further calls in case there is a reboot
        # - passing the path to the temporary screenrc file to be removed
        # the latter is only added when running a new screen socket (not window)
        env={ "arguments":{
              "python_exec" : python_exec,
              "screen_cmd"  : screen_cmd,
              "period"      : period,
              "cron_file"   : cron_file,
              "detach"      : detach,
              "max_times"   : max_times
            }}

        if python_exec == "":
            python_exec = sys.executable
        print(_now(),"python_exec:", python_exec)

        cmd = []
        cmd.extend(shlex.split(screen_cmd)) # allows screen_cmd to contain space separated options

        # in window mode, we do not launch a new socket.
        # these arguments are thus ignored
        if not window:
            sockname = _adjust_sockname(sockname,cron_file)

            # check if another process is already running
            if unique:
               cmd1=cmd.copy() # do not modity cmd
               cmd1.append("-ls")
               for l in subprocess.run(cmd1, # do not check errors here since screen -ls fails on MacOS
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True).stdout.split('\n'):
                   if "." + sockname +"\t" in l:
                       print("Another screen with socket name " + sockname + " is already present")
                       return

            # run in detaced mode
            if detach:
                cmd.append("-d")

            # write screen log
            if screen_log != "":
                cmd.append("-L")
                if screen_log != "screenlog.0":
                  cmd.append("-Logfile")
                  cmd.append(screen_log)

            # force a new screen
            cmd.append("-m")

            # create a named socket
            cmd.append("-S")
            cmd.append(sockname)

            # create a temporary screenrc file
            rcfile=tempfile.NamedTemporaryFile("w+t",delete=False)
            print(_screenrcfile,file=rcfile)
            rcfile.flush()

            # pass the temporary screenrc file as an argument
            cmd.append("-c")
            cmd.append(rcfile.name)

            # store the path so that the file can be cancelled later
            env["rcfile"]=rcfile.name

        cmd.append("-t")
        cmd.append("running")
        cmd.append("/usr/bin/env")
        cmd.append("BUSSILAB_CRON_SCREEN_ARGS=" + json.dumps(env))
        if keep_ld_library_path and 'LD_LIBRARY_PATH' in os.environ:
            cmd.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
        cmd.extend(shlex.split(python_exec)) # allows python_exec to contain space separated options
        cmd.append("-m")
        cmd.append("bussilab")
        cmd.append("cron")
        cmd.append("--no-screen")
        if period is not None:
          cmd.append("--period")
          cmd.append(str(period))
        if len(cron_file)>0:
            cmd.append("--cron-file")
            cmd.append(cron_file)
        if quick_start:
            cmd.append("--quick-start")
            if quick_start_skip_steps>0:
              cmd.append("--quick-start-skip-steps")
              cmd.append(str(quick_start_skip_steps))
            if quick_start_event>0:
              cmd.append("--quick-start-event")
              cmd.append(str(quick_start_event))
        if max_times is not None:
            cmd.append("--max-times")
            cmd.append(str(max_times))

        print(_now(),"cmd:",cmd)

        try:
            ret=subprocess.call(cmd)
            if ret != 0:
                msg = "An error occurred."
                if screen_log !="" and screen_log != "screenlog.0":
                    msg += " Notice that some screen versions do not support a logfile with a name different from screenlog.0"
                raise RuntimeError(msg)
        except OSError:
            raise RuntimeError("Execution of failed. Perhaps '" + screen_cmd + "' command is not available on your system.")
