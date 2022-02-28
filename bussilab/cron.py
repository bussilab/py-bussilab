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

_screenrcfile = """
  hardstatus string 'cron server do not kill %{= Kd} %{= Kd}%-w%{= Kr}[%{= KW}%n %t%{= Kr}]%{= Kd}%+w %-= %{KG} %H%{KW}|%{KY}%101`%{KW}|%D %M %d %Y%{= Kc} %C%A%{-}'
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
    now=time.localtime()
    seconds=now.tm_wday*24*3600+now.tm_hour*3600 + now.tm_min*60 + now.tm_sec
    seconds_=seconds
    seconds = seconds%period
    # also returns predicted time for next event
    return period-seconds,seconds_+period-seconds

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
    sockname=re.sub(r"\(path\)",path_to_cron_file,sockname)
    return sockname

def _run(cron_file: str,
         period: int,
         event: int,
         counter: int):
    try:
        print(_now(),"Running now")
        if len(cron_file) > 0:
            with open(cron_file) as rc:
                config=yaml.load(rc,Loader=yaml.BaseLoader)
        else:
            config=coretools.config()
        if "cron" in config:
            for i in range(len(config["cron"])):
               if isinstance(config["cron"][i],str):
                   config["cron"][i]={
                       "type": "python",
                       "script": config["cron"][i]
                   }
               c=config["cron"][i]
               if not "type" in c:
                   c["type"]="python"

               if "delay" in c and not "skip" in c:
                   raise RuntimeError("delay can only be used with skip")

               if "skip" in c:
                   skip = eval(c["skip"])
                   if "delay" in c:
                       delay=eval(c["delay"])
                   else:
                       delay=0
                   if (event/period)%skip != delay%skip:
                       continue

               if c["type"] == "python":
                   args = [sys.executable, "-c"]
               elif c["type"] == "bash":
                   args = ["bash", "--noprofile", "--norc", "-c"]
               elif c["type"] == "selfupdate":
                   timeout=_time_to_next_event(period)[0]//2+1
                   pip.upgrade_self(timeout=timeout)
                   return _reboot(iterations=counter+1) # +1 is to add the current iteration to the count
               elif c["type"] == "reboot":
                   return _reboot(iterations=counter+1) # +1 is to add the current iteration to the count
               else:
                   raise RuntimeError("Unknown type " + config["cron"][i]["type"])

               args.append(config["cron"][i]["script"])
               timeout=_time_to_next_event(period)[0]//2+1
               print(_now(),"cmd " + str(i) +" with timeout " + str(timeout))
               subprocess.run(args, timeout=timeout)
    except Exception as e:
        print(e)
        print()
        print(_now(),"Wait for the next scheduled event")

class _reboot_now():
    pass

def _reboot(*,
           iterations=0):
    if "BUSSILAB_CRON_SCREEN_ARGS" in os.environ:
        print(os.environ["BUSSILAB_CRON_SCREEN_ARGS"])
        env = json.loads(os.environ["BUSSILAB_CRON_SCREEN_ARGS"])
        args = env["arguments"]
        args["no_screen"]=False # reboots should be done with screen, which is switched off by default
        args["quick_start"]=False # disable quick start
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
         cron_file: str = "",
         screen_cmd: str = "screen",
         screen_log: str = "",
         no_screen: bool = True,
         keep_ld_library_path: bool = True,
         sockname: str = "(path):cron",
         python_exec: str = "",
         detach: bool = False,
         period: int = 3600,
         max_times: Optional[int] = None,
         unique: bool = False,
         window: bool = False
         ):
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
        counter=0
        if quick_start:
            r=_run(cron_file,period,0,counter)
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
            s=_time_to_next_event(period)
            print(_now(),"Waiting " +str(s[0])+ " seconds for next scheduled event")
            time.sleep(s[0])
            r=_run(cron_file,period,s[1],counter)
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
        print("python_exec:", python_exec)

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

        cmd.append("/usr/bin/env")
        cmd.append("BUSSILAB_CRON_SCREEN_ARGS=" + json.dumps(env))
        if keep_ld_library_path and 'LD_LIBRARY_PATH' in os.environ:
            cmd.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
        cmd.extend(shlex.split(python_exec)) # allows python_exec to contain space separated options
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
