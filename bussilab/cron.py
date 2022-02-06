import os
import time
import sys
import datetime
import subprocess
from typing import Optional
import yaml
import re
from . import coretools

def _time_to_next_event(period: int):
    now=time.localtime()
    seconds=now.tm_hour*3600 + now.tm_min*60 + now.tm_sec
    seconds = seconds%period
    return period-seconds

def _now():
    return '{}'.format(datetime.datetime.now()) + ":"

def _adjust_sockname(sockname,cron_file):
    path_to_cron_file=""
    if len(cron_file) > 0:
        path_to_cron_file=cron_file
    else:
        path_to_cron_file=str(coretools.config_path())
    path_to_cron_file=os.path.abspath(path_to_cron_file)
    path_to_cron_file=re.sub("[^-A-Za-z0-9.]",":",path_to_cron_file)
    sockname=re.sub("\(path to cron file\)",path_to_cron_file,sockname)
    return sockname

def _run(cron_file: str,
         period: int):
    try:
        print(_now(),"Running now")
        if len(cron_file) > 0:
            with open(cron_file) as rc:
                config=yaml.load(rc,Loader=yaml.BaseLoader)
        else:
            config=coretools.config()
        if "cron" in config:
            for i in range(len(config["cron"])):
               args = [sys.executable, "-c", config["cron"][i]]
               timeout=_time_to_next_event(period)//2+1
               print(_now(),"cmd " + str(i) +" with timeout " + str(timeout))
               subprocess.run(args, timeout=timeout)
    except Exception as e:
        print(e)
        print()
        print(_now(),"Wait for the next scheduled event")

def cron(*,
         quick_start: bool = False,
         cron_file: str = "",
         screen_cmd: str = "screen",
         screen_log: str = "",
         no_screen: bool = True,
         keep_ld_library_path: bool = True,
         sockname: str = "(path to cron file):cron",
         python_exec: str = "",
         detach: bool = False,
         period: int = 3600,
         max_times: Optional[int] = None
         ):
    if no_screen:
        print(_now(),"start")
        counter=0
        if quick_start:
            _run(cron_file,period)
            counter += 1
            if max_times is not None:
                if counter >= max_times:
                    return
        while True:
            s=_time_to_next_event(period)
            print(_now(),"Waiting " +str(s)+ " seconds for next scheduled event")
            time.sleep(s)
            _run(cron_file,period)
            counter += 1
            if max_times is not None:
                if counter >= max_times:
                    return
    else:
        if python_exec == "":
            python_exec = sys.executable
        print("python_exec:", python_exec)
        cmd = []
        cmd.extend(screen_cmd.split()) # allows screen_cmd to contain space separated options
        if detach:
            cmd.append("-d")
        if screen_log != "":
            cmd.append("-L")
            if screen_log != "screenlog.0":
              cmd.append("-Logfile")
              cmd.append(screen_log)
        cmd.append("-m")
        cmd.append("-S")
        cmd.append(_adjust_sockname(sockname,cron_file))
        if keep_ld_library_path and 'LD_LIBRARY_PATH' in os.environ:
            cmd.append("env")
            cmd.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
        cmd.extend(python_exec.split()) # allows python_exec to contain space separated options
        cmd.append("-m")
        cmd.append("bussilab")
        cmd.append("cron")
        cmd.append("--no-screen")
        cmd.append("--period")
        cmd.append(str(period))
        if len(cron_file)>0:
            cmd.append("--cron-file")
            cmd.append(cron_file)
        if quick_start:
            cmd.append("--quick-start")
        if max_times is not None:
            cmd.append("--max-times")
            cmd.append(str(max_times))
        print("cmd:",cmd)
        
        try:
            ret=subprocess.call(cmd)
            if ret != 0:
                msg = "An error occurred."
                if screen_log !="" and screen_log != "screenlog.0":
                    msg += " Notice that some screen versions do not support a logfile with a name different from screenlog.0"
                raise RuntimeError(msg)
        except OSError:
            raise RuntimeError("Execution of failed. Perhaps '" + screen_cmd + "' command is not available on your system.")
