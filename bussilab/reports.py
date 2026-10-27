import subprocess
import re
import secrets
import shlex
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
    # top may omit whitespace after commas and use decimal commas.
    text = " ".join(line)
    result = {
        key: float(value.replace(",", "."))
        for value, key in re.findall(
            r"(\d+(?:[.,]\d+)?)\s*(us|sy|ni|id|wa|hi|si|st)\b", text)
    }
    missing = {"id", "us", "ni"} - result.keys()
    if missing:
        raise ValueError("missing CPU fields: " + ", ".join(sorted(missing)))
    return result

def workstations(wks: Optional[List] = None, short: bool = True):
    """Report CPU, disk and automatically detected NVIDIA GPU usage over SSH.

    The legacy ``nvidia`` workstation setting is ignored. Machines without
    ``nvidia-smi`` omit GPU status; a failed GPU query produces a warning.
    """
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
        else:
            raise TypeError()

        # this is required to allow discarding possible initial login messages
        token=secrets.token_hex()
        cmd = "export LC_ALL=C; echo " + token + ";"
        cmd += "top -n 1 -b | head -n 3 | tail -n 1;"
        cmd += "df -Ph " + shlex.quote(disk) + " | tail -n 1;"
        cmd += "df -Ph " + shlex.quote(tmpdisk) + " | tail -n 1;"
        # GPU probing is optional; its failure must not discard CPU/disk data.
        cmd += ("if command -v nvidia-smi >/dev/null 2>&1; then "
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                " || echo __BUSSILAB_GPU_ERROR__; fi")

        args = ['ssh',url, cmd]
        msg += name
        try:
            out = subprocess.run(args,
               stdout = subprocess.PIPE,
               stderr = subprocess.PIPE,
               universal_newlines=True,
               timeout = 60,
               check = True).stdout.split('\n')
            try:
                start = out.index(token)
            except ValueError as exc:
                raise ValueError("missing report start marker") from exc
            out = out[start+1:]
            if len(out) < 3:
                raise ValueError("incomplete CPU/disk output")
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
            gpu_lines = [line.strip() for line in out[3:] if line.strip()]
            if gpu_lines:
                msg += " GPU"
                for line in gpu_lines:
                    if line == "__BUSSILAB_GPU_ERROR__" or line == "N/A" or line == "[N/A]":
                        msg += " :warning:"
                        continue
                    gpu_usage = float(line)
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
        except (subprocess.CalledProcessError, OSError) as exc:
            msg += " :skull_and_crossbones:"
            msg += "\n"
            msg += "Error connecting: " + str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                msg += "\n" + exc.stderr.strip()
            msg += "\n\n"
        except (ValueError, IndexError, KeyError) as exc:
            msg += " :skull_and_crossbones:\n"
            msg += "Error parsing report: " + str(exc) + "\n\n"
    return msg

