import os
import time
import sys
import datetime
import subprocess
from typing import Optional
import yaml
from . import coretools

def _time_to_next_event(period: int):
    now=time.localtime()
    seconds=now.tm_hour*3600 + now.tm_min*60 + now.tm_sec
    seconds = seconds%period
    return period-seconds

def _now():
    return '{}'.format(datetime.datetime.now()) + ":"

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
         no_screen: bool = True,
         sockname: str = "cron",
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
        cmd = screen_cmd
        if detach:
            cmd += " -d"
        cmd += " -m -S " + sockname
        cmd +=" "+ python_exec + " -m bussilab cron --no-screen"
        cmd +=" --period " + str(period)
        if len(cron_file)>0:
            cmd += " --cron-file " + cron_file
        if quick_start:
            cmd += " --quick-start"
        if max_times is not None:
            cmd += " --max-times " + str(max_times)
        print(cmd)
        os.system(cmd)
