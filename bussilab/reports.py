import subprocess
import re
import secrets
from typing import Optional, List

from . import coretools

def _parse_cpu(line: List[str]):
    #  us, user    : time running un-niced user processes
    #  sy, system  : time running kernel processes
    #  ni, nice    : time running niced user processes
    #  id, idle    : time spent in the kernel idle handler
    #  wa, IO-wait : time waiting for I/O completion
    #  hi : time spent servicing hardware interrupts
    #  si : time spent servicing software interrupts
    #  st : time stolen from this vm by the hypervisor
    field = ["us", "sy", "ni", "id", "wa", "hi", "si", "st"]
    result = {}
    for i in range(1,len(line)):
        for f in field:
            if re.match("^" + f + ".*",line[i]):
                result[f] = float(line[i-1])
    return result

def workstations(wks: Optional[List] = None, short: bool = True):
    msg = ""

    if not wks:
        c = coretools.config()
        wks = c["workstations"]

    if not wks:
        raise ValueError("cannot build wks list")

    for w in wks:
        if isinstance(w, str):
            name = w
            url = w
            disk = "/scratch"
            tmpdisk = "/var"
            nvidia = 'True'
        elif isinstance(w, dict):
            url = w['url']
            try:
                name = w['name']
            except KeyError:
                name = url
            try:
                disk = w['disk']
            except KeyError:
                disk = "/scratch"
            try:
                tmpdisk = w['tmpdisk']
            except KeyError:
                tmpdisk = "/var"
            try:
                nvidia = w['nvidia']
            except KeyError:
                nvidia = 'True'
        else:
            raise TypeError()

        # this is required to allow discarding possible initial login messages
        token=secrets.token_hex()
        cmd = "echo " + token + ";"
        cmd += "top -n 1 -b | head -n 3 | tail -n 1;"
        cmd += "df -h " + disk + " | tail -n 1;"
        cmd += "df -h " + tmpdisk + " | tail -n 1;"

        if nvidia != "False":
            cmd += "nvidia-smi  | grep Default"

        args = ['ssh',url, cmd]
        msg += name
        try:
            out = subprocess.run(args,
               stdout = subprocess.PIPE,
               stderr = subprocess.PIPE,
               universal_newlines=True,
               timeout = 60,
               check = True).stdout.split('\n')
            for i in range(len(out)):
                if out[i]==token:
                   break
            out=out[i+1:]
            cpu_fields = _parse_cpu(out[0].split())
            msg += " CPU"
            if cpu_fields["id"] > 75:
                msg += " :sleeping:"
            elif cpu_fields["id"] > 50:
                msg += " :walking:"
            else:
                msg += " :running:"
            if cpu_fields["id"] + cpu_fields["us"] + cpu_fields["ni"] < 80:
                msg += "(:warning: id+us+ni<80)"
            if nvidia != 'False':
                msg += " GPU"
                for j in range(3,len(out)-1):
                    gpu_fields = out[j].split()
                    gpu_usage = 0.0
                    for i in range(1,len(gpu_fields)):
                        if re.match("^Default",gpu_fields[i]):
                            gpu_usage = float(gpu_fields[i-1].strip("%"))
                    if gpu_usage < 25:
                        msg += " :sleeping:"
                    elif gpu_usage < 50:
                        msg += " :walking:"
                    else:
                        msg += " :running:"
            msg += " disk"
            # scratch
            disk_fields = out[1].split()
            disk_occupation = int(disk_fields[-2].strip("%"))
            if disk_occupation < 80:
                msg += " :smile:"
            elif disk_occupation < 90:
                msg += " :neutral_face:"
            elif disk_occupation < 99:
                msg += " :worried:  ({}%)".format(disk_occupation)
            else:
                msg += " :scream: ({}%)".format(disk_occupation)
            # var
            disk_fields = out[2].split()
            disk_occupation = int(disk_fields[-2].strip("%"))
            if disk_occupation > 70:
                msg += " (:warning: /var {}%)".format(disk_occupation)
            msg += "\n"
            if not short:
                msg+= "\n".join(out)+"\n\n"
        except subprocess.TimeoutExpired:
            msg += " :skull_and_crossbones:"
            msg += "\n"
            msg += "Timeout"
            msg += "\n\n"
        except Exception:
            msg += " :skull_and_crossbones:"
            msg += "\n"
            msg += "Error connecting"
            msg += "\n\n"
    return msg

