"""
Workstation status reports collected over SSH.

:func:`workstations` returns a compact Markdown table for posting with
``bussilab.notify.notify(report, type="markdown")``. Workstations can be passed
explicitly or read from the ``workstations`` configuration list. CPU and GPU
utilization and disk occupancy are shown as percentages, with connection
status circles and warning symbols. Set ``short=False`` to append raw output.

## Changelog

- **v0.0.53**
  - Replaced emoji-based status summaries with a Markdown table containing
    Workstation, CPU, GPU, Disk and Temp columns. Added numeric percentages,
    green/red connection indicators and warning symbols beside affected values.
    Raw output now follows the table in fenced code blocks.
  - Detect ``nvidia-smi`` in the remote shell's PATH automatically, ignoring
    the legacy ``nvidia`` setting. Query GPU utilization as CSV; missing tools
    omit GPU usage, while failed queries retain CPU/disk results and show
    unavailable GPU utilization.
  - Handle CPU fields without intervening spaces and decimal-comma values;
    request the C locale remotely. Quote disk paths and handle spaces in
    mount points. Distinguish connection failures from parsing errors.
  - Added automated tests using mocked SSH calls and local fake utilities;
    no SSH connection is required by the tests.
- **v0.0.41**
  - Increased the SSH timeout from 20 to 60 seconds.
- **v0.0.34**
  - Added a random output marker to discard login banners before parsing.
- **v0.0.33**
  - Default workstation names to their SSH address and initialize the temp
    path for string entries. Include percentages in disk-space warnings.
- **v0.0.31**
  - Added configurable temp-disk monitoring, defaulting to ``/var``, with a
    warning above 70% occupancy.
- **v0.0.2**
  - Introduced SSH workstation reports with CPU, NVIDIA GPU and disk status,
    configuration-based host lists, timeouts and optional raw output.
"""

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

def _markdown_cell(value: str):
    """Keep arbitrary names, paths and diagnostics inside one table cell."""
    value = " ".join(value.splitlines())
    return re.sub(r"([\\`*_|\[\]<>])", r"\\\1", value)


def _disk_usage(line: str):
    match = re.search(r"(?:^|\s)(\d+)%\s", line)
    if match is None:
        raise ValueError("missing disk usage percentage")
    return int(match.group(1))


def workstations(wks: Optional[List] = None, short: bool = True):
    """Return a Markdown table of workstation CPU, GPU and disk usage.

    Send the result with ``notify(message=report, type="markdown")``.
    A green circle means SSH returned data; a red circle marks a connection
    failure or timeout. Collection errors appear in the affected metric cells.
    CPU usage is 100 minus idle; GPU percentages follow device order.
    Warning values carry a warning symbol: CPU id+us+ni below 80%, disk usage at least
    90%, temp usage above 70%, and unavailable GPU utilization.
    Disk and Temp show the used percentage of their configured paths. An em dash means
    no GPU was detected. With ``short=False``, raw output follows the table
    in fenced code blocks.

    The legacy ``nvidia`` workstation setting is ignored. Machines without
    ``nvidia-smi`` omit GPU status; a failed GPU query produces a warning.
    """
    rows = ["| Workstation | CPU | GPU | Disk | Temp |",
            "| --- | --- | --- | --- | --- |"]
    details = []

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
        cpu = gpu = disk_status = tmp_status = "—"
        status = "🔴"
        warning_columns = set()
        try:
            out = subprocess.run(args,
               stdout = subprocess.PIPE,
               stderr = subprocess.PIPE,
               universal_newlines=True,
               timeout = 60,
               check = True).stdout.split('\n')
            status = "🟢"
            try:
                start = out.index(token)
            except ValueError as exc:
                raise ValueError("missing report start marker") from exc
            out = out[start+1:]
            if len(out) < 3:
                raise ValueError("incomplete CPU/disk output")
            cpu_fields = _parse_cpu(out[0].split())
            idle = cpu_fields["id"]
            cpu = "{:g}%".format(100 - idle)
            if idle + cpu_fields["us"] + cpu_fields["ni"] < 80:
                warning_columns.add(1)
            gpu_lines = [line.strip() for line in out[3:] if line.strip()]
            gpu_status = []
            for line in gpu_lines:
                if line in ("__BUSSILAB_GPU_ERROR__", "N/A", "[N/A]"):
                    gpu_status.append("unavailable")
                    continue
                usage = float(line)
                gpu_status.append("{:g}%".format(usage))
            gpu = ", ".join(gpu_status) if gpu_status else "—"
            # The percentage precedes the mount point, which may contain spaces.
            disk_occupation = _disk_usage(out[1])
            disk_status = "{}%".format(disk_occupation)
            if disk_occupation >= 90:
                warning_columns.add(3)
            tmp_occupation = _disk_usage(out[2])
            tmp_status = "{}%".format(tmp_occupation)
            if tmp_occupation > 70:
                warning_columns.add(4)
            if not short:
                raw = "\n".join(out).rstrip()
                # A longer fence also handles backticks in remote output.
                fence = "`" * max(3, max((len(m.group()) + 1 for m in
                                          re.finditer(r"`+", raw)), default=0))
                details.append("### " + _markdown_cell(name) + "\n\n" +
                               fence + "text\n" + raw + "\n" + fence)
        except subprocess.TimeoutExpired:
            cpu = "Timeout"
        except (subprocess.CalledProcessError, OSError) as exc:
            cpu = "Error connecting: " + str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                cpu += " " + exc.stderr.strip()
        except (ValueError, IndexError, KeyError) as exc:
            # Mark the metric being parsed while preserving earlier results.
            if cpu == "—":
                cpu = "Error parsing report: " + str(exc)
            elif gpu == "—" and gpu_lines:
                gpu = "Error parsing report: " + str(exc)
            elif disk_status == "—":
                disk_status = "Error parsing report: " + str(exc)
            else:
                tmp_status = "Error parsing report: " + str(exc)
        cells = [_markdown_cell(cell) for cell in
                 (status + " " + name, cpu, gpu, disk_status, tmp_status)]
        for column in warning_columns:
            # Add warning symbols after escaping remote text.
            cells[column] = re.sub(r"(\d+(?:\.\d+)?%)$", r"\1 ⚠️", cells[column])
        cells[2] = cells[2].replace("unavailable", "unavailable ⚠️")
        rows.append("| " + " | ".join(cells) + " |")
    report = "\n".join(rows) + "\n"
    if details:
        report += "\n" + "\n\n".join(details) + "\n"
    return report
