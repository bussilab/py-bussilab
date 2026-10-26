"""
Tools to implement the command line interface
=============================================

This module defines the command-line parser and its subcommands. The same
commands can be run from Python with `cli()`:

```python
from bussilab.cli import cli
cli("-h")
cli("wham -b bias") # provide the command line as a string
cli(["wham", "-b", "bias"]) # alternatively use a list
```

For command usage and options, see
[cli_documentation](cli_documentation.html) or run `bussilab COMMAND -h`.
The `command`, `arg`, `group`, and `endgroup` decorators register additional
commands. Commands registered after this module is imported become available
on the next `cli()` call.

The underlying modules also provide Python functions that work directly with
data structures. For example, `bussilab.wham.wham(bias)` accepts a NumPy array,
while the `wham` command reads a file.

## Changelog

- **v0.0.52**
  - Added `notify --no-unfurl`, `--markdown-file`, and support for multiple
    paths with `--file`.
  - Made invalid command-line arguments exit with argparse's status code
    instead of printing a Python traceback; `cli()` still raises `TypeError`
    for parser errors by default when called from Python.
  - Preserved two-dimensional WHAM input from single-row and single-column
    files. Added clear errors for `--use-frame-weight` without a bias column
    and `--traj-weight` without values.
  - Made commands registered after import available to subsequent `cli()`
    calls by rebuilding the cached parser when needed.
  - Expanded command usage notes and clarified the generated CLI documentation.
- **v0.0.49**
  - Corrected numeric parsing of `wham --threshold` and the Jupyter `--port`
    options; fixed help and output for `required` and other command examples.
- **v0.0.42**
  - Added notification replies, reactions, quiet mode, footer control, and
    Screen log attachments with an optional line limit.
- **v0.0.41**
  - Added `jrun --lab` and improved Jupyter Screen socket names and usage notes.
  - Added cron quick-start options for resuming steps and event numbers, and
    allowed the period to come from the cron configuration.
- **v0.0.35**
  - Used `shlex.split()` for string arguments passed to `cli()`, preserving
    quoted values.
- **v0.0.34**
  - Added `cron --unique`, Jupyter Screen log support, and improved default
    Screen socket names.
- **v0.0.32**
  - Added file uploads to `notify`.
- **v0.0.12**
  - Added the `required` command for pip, conda, and MacPorts dependencies.
- **v0.0.2**
  - Introduced the command registration helpers, Python `cli()` entry point,
    generated command documentation, and the initial subcommands.

"""

# Minimal imports here.

# Support argument parsing:
import argparse

# Support deprecation warnings:
import warnings

# Support split
import shlex

# Support static type checks:
from typing import Optional, Union, List, Callable, Dict, Tuple

# Do not import heavy modules here since they could slow down the command line interface
# It's fine instead to load them lazily in functions where they are needed.

class _Argument():
    """Special class returned by the decorators present in this module."""
    def __init__(self, func: Union[Callable, "_Argument"]):
        if isinstance(func, _Argument):
            self.func: Callable = func.func
            self.calls: List[Callable] = func.calls
        else:
            self.func = func
            self.calls = []
    def __call__(self, *args, **kwargs):
        msg = ("Functions that are registered for the command line are not directly callable.\n"
               "Use cli() instead.")
        raise NotImplementedError(msg)

# list of registered commands:
_commands: List[Tuple[str, str, str, _Argument, Dict]] = []


class _ExtendedParser():
    """Internal class used to attach subparsers help to parser."""
    def __init__(self,
                 parser: argparse.ArgumentParser,
                 subparsers_help: Dict[str, str],
                 subparsers_doc: Dict[str, str]
                ):
        self.parser = parser
        self.subparsers_help = subparsers_help
        self.subparsers_doc = subparsers_doc

_eparser: Optional[_ExtendedParser] = None

def _create_eparser(prog: Optional[str] = None) -> _ExtendedParser:
    """Create a `parser` object."""
    parser = argparse.ArgumentParser(prog=prog, description="command line tool")
    parser.add_argument("--version", action='store_true', dest="_version")
    subparsers = parser.add_subparsers(title="Subcommands")
    subparsers_help = {}
    subparsers_doc = {}
    for a in _commands:
        (name, h, description, func, kwargs) = a
        p = [subparsers.add_parser(name, help=h, description=description, **kwargs)]
        f = p[0]
        for call in reversed(func.calls):
            p = call(p)
        if len(p) > 1:
            msg = "Non matching group/endgroup"
            raise TypeError(msg)
        p[0].set_defaults(_func=func.func)
        subparsers_help[name] = f.format_help()
        if func.func.__doc__ is not None:
            subparsers_doc[name] = func.func.__doc__
        else:
            subparsers_doc[name] = ""
    return _ExtendedParser(parser, subparsers_help, subparsers_doc)

def command(name: str, help: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator that registers a function as a subcommand.

       Write this decorator **above** the other decorators
       `bussilab.cli.arg`, `bussilab.cli.group`, and `bussilab.cli.endgroup`.
       Newly registered commands are available on the next `cli()` call.

       Parameters
       ----------
       name : str
           Name of the subcommand (will be used on the command line)

       help : str
           Short help message for the subcommand (one line).

       description : str, optional
           Longer description. If not provided, it is set to a copy of `help`.

       kwargs
           Other parameters are passed as is to the `add_parser` function of `argparse`.

       Examples
       --------

       Simple command line tool that accepts a single `--out` argument followed by a string
       and call the function `do_something` with that string as an argument.

       ```python
       from bussilab.cli import command, arg

       @command("subcommand")
       @arg("--out")
       def myfunc(out):
           do_something(out)
       ```
    """
    if description is None:
        description = help
    def add_command(func):
        global _eparser
        func = _Argument(func)
        _commands.append((name, help, description, func, kwargs))
        _eparser = None
        return func
    return add_command

def arg(*name, **kwargs):
    """ Decorator that adds an argument to a command line tool.

        Parameters are passed to the `parser.add_argument()` function.
        Write it **below** the `bussilab.cli.command` decorator.
    """
    def call(p):
        p[-1].add_argument(*name, **kwargs)
        return p
    def add_argument(func):
        func = _Argument(func)
        func.calls.append(call)
        return func
    return add_argument

def group(title: Union[str, Callable, None] = None,
          description: Optional[str] = None,
          exclusive: Optional[bool] = None,
          required: Optional[bool] = None):
    """ Decorator that adds a group of arguments for a command line tool.

        Write it **below** the `bussilab.cli.command` decorator.
        It should be followed by a number of `bussilab.cli.arg` decorators and by a closing
        `bussilab.cli.endgroup` decorator.

        Parameters
        ----------
        title : str
            The name of the group. Can only be used for non exclusive groups.

        description : str
            A description of the group. Can only be used for non exclusive groups.

        exclusive : bool
            If True, the arguments belonging to this group are mutually exclusive

        required : bool
            If True, one of the arguments at least should be passed.
            Can only be used for exclusive groups.

        Examples
        --------

        This is a simple command line tool that accepts three arguments (`-a`, `-b`, or `-c`),
        mutually exclusive. When ran, it will just print booleans showing if these arguments
        were passed.

        ```python
        from bussilab.cli import command, group, arg, endgroup

        @command("doit")
        @group(exclusive=True)
        @arg("-a", action='store_true')
        @arg("-b", action='store_true')
        @arg("-c", action='store_true')
        @endgroup
        def check(a, b, c):
           print(a, b, c)
        ```
    """
    noarg = None
    # note: _Argument is callable as well
    if description is None and exclusive is None and required is None and callable(title):
        noarg = title
        title = None
    if exclusive is None:
        exclusive = False
    if not exclusive and required is not None:
        msg = "required can only be used for exclusive groups"
        raise TypeError(msg)
    if exclusive and title is not None:
        msg = "title can only be used for non exclusive groups"
        raise TypeError(msg)
    if exclusive and description is not None:
        msg = "description can only be used for non exclusive groups"
        raise TypeError(msg)
    if exclusive and required is None:
        required = False  # default for add_mutually_exclusive_group
    def call(p):
        if exclusive:
            p.append(p[-1].add_mutually_exclusive_group(required=required))
        else:
            p.append(p[-1].add_argument_group(title=title, description=description))
        return p
    def begin_group(func: Union[Callable, _Argument]):
        func = _Argument(func)
        func.calls.append(call)
        return func
    if noarg:
        return begin_group(noarg)
    return begin_group

def endgroup(f: Optional[Callable] = None):
    """ Decorator that ends a group of arguments for a command line tool.

        See `bussilab.cli.group`.
    """
    def call(p):
        if len(p) < 2:
            msg = "Non matching group/endgroup"
            raise TypeError(msg)
        return p[:-1]
    def end_group(func):
        func = _Argument(func)
        func.calls.append(call)
        return func
    if f:
        return end_group(f)
    return end_group

def cli(arguments: Union[str, List[str]] = "",
        *,
        prog: Optional[str] = "",  # None is used to imply sys.argv[0]
        use_argcomplete: bool = False,
        throws_on_parser_errors: bool = True
       ) -> Optional[int]:
    """Executes a command line tool from python.

       Run a registered subcommand from Python using the same arguments as the
       `bussilab` executable. A string is split with `shlex.split()`; a list
       preserves each argument exactly.

       Parameters
       ----------
       arguments : str or list
           Command line arguments. If a string is passed, it is first split using
           shlex.split()

       prog : str or None
           Name used in help and error messages. The default uses an empty name;
           None uses `sys.argv[0]`. **Mostly for internal use**.

       use_argcomplete : bool
           If True, the `autocomplete` function of `argcomplete` module is called on the parser,
           so as to allow autocompletion in the command line tool.
           If `argcomplete` module is not installed, nothing is done and no failure is reported.
           **Mostly for internal use**.

       throws_on_parser_errors : bool
           If True, convert nonzero parser exits to `TypeError`. If False,
           return the parser's exit code. The executable uses False so invalid
           arguments exit with argparse's status code. **Mostly for internal use**.

       Returns
       -------
       None or int
           Returns None after a successful command or a help request. With
           `throws_on_parser_errors=False`, returns the parser's nonzero exit
           code for invalid arguments. Exceptions raised by a command propagate.
    """

    global _eparser

    # allow passing a single string
    if isinstance(arguments, str):
        arguments = shlex.split(arguments)

    func = None

    if prog == "":
        if _eparser is None:
            _eparser = _create_eparser("")
        eparser = _eparser
    else:
        eparser = _create_eparser(prog)

    if use_argcomplete:
        # optional feature:
        try:
            import argcomplete
            argcomplete.autocomplete(eparser.parser)
        except ImportError:
            pass

    # Parse options
    # In order to avoid python to crash when cli() is called from python
    # with wrong arguments, it is necessary to intercept the exception.
    try:
        args = vars(eparser.parser.parse_args(arguments))
    except SystemExit as e:
        if e.code != 0:
            if throws_on_parser_errors:
                raise TypeError("Error parsing command line arguments," +
                                " return code is " + str(e.code))
            else:
                return e.code
        return None

    if '_func' in args:
        func = args["_func"]
        del args["_func"]

    main_args = {}
    for i in args.keys():
        if i[0] == "_":
            main_args[i[1:]] = args[i]

    for i in main_args:
        del args["_"+i]

    if "version" in main_args and main_args["version"]:
        from . import __version__
        print(__version__)
        return None

    remove = []
    for i in args.keys():
        if args[i] is None:
            remove.append(i)

    for i in remove:
        del args[i]

    if func:
        func(**args)
    else:
        eparser.parser.print_usage()

    return None

def _main_(prog: Optional[str] = None):

    import sys

    # process command line arguments:
    # prog==None implies argv[0]
    sys.exit(cli(prog=prog, arguments=sys.argv[1:], use_argcomplete=True,
                 throws_on_parser_errors=False))

################################################################################

# BELOW ARE ALL THE COMMAND LINE TOOLS

# These functions are made not available from python by their decorators.
# Only to be used from cli() or from the command line.

@command("list", help="List available python modules.")
def _list_submodules():
    """Print the available Python submodules and their short descriptions."""
    from . import list_submodules, describe_submodule
    print("List of available python modules:")
    print()
    l = list_submodules()
    c = max(map(len, l))
    for m in list_submodules():
        print(m, " "*(c-len(m)) + ": " + describe_submodule(m))

@command("check",
         help="Check installed features",
         description="Check installed features")
@group(exclusive=True)
@arg("--import", action='store_true',
     dest="check_import", help="check if all the submodules can be imported")
@endgroup
def _check(check_import: bool = False):
    """Import every submodule with `--import` to check installed dependencies."""
    if check_import:
        from . import import_submodules
        print("Importing all submodules:")
        import_submodules()
        print("All submodules were imported successfully, dependencies seem to be in place")
    else:
        print("nothing to check")

@command("wham", help="Perform binless WHAM",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("-b", "--bias", help="File containing bias potential", required=True)
@arg("-o", "--out", help="Output file with log weights")
@arg("--use-frame-weight", action="store_true",
     help="read frame weights from the last bias-file column")
@arg("--traj-weight", nargs="+", type=float,
     help="relative weights of the bias columns")
@arg("-T", "--temperature", type=float, default=1.0,
     help="system temperature in energy units", dest="T")
@arg("-m", "--maxiter", type=int, default=1000, help="maximum number of iterations")
@arg("-t", "--threshold", type=float, default=1e-40, help="threshold for convergence")
@arg("-v", "--verbose", action="store_true")
def _wham(**args):
    """Read a whitespace-delimited bias matrix and run binless WHAM.

       Each row is a frame and each column is a Hamiltonian's bias potential.
       With `--use-frame-weight`, the last column supplies frame weights and
       at least one preceding bias column is required. Print log weights to
       standard output, or write them to `--out` with four decimal places.
       Iteration count, error, and log partition functions are always printed
       to standard output. See `bussilab.wham` for the Python API.
    """
    from . import wham
    import numpy as np
    # load bias file:
    args["bias"] = np.loadtxt(args["bias"], ndmin=2)

    # with --use-frame-weight, last column in bias is interpreted as frame weight
    if "use_frame_weight" in args:
        if args["use_frame_weight"]:
            if args["bias"].shape[1] < 2:
                raise ValueError("--use-frame-weight requires at least one bias column and one weight column")
            args["frame_weight"] = args["bias"][:, -1]
            args["bias"] = args["bias"][:, :-1]
        del args["use_frame_weight"]

    if "out" in args:
        outfile = args["out"]
        del args["out"]
    else:
        outfile = None

    ret = wham.wham(**args)

    print("# nit:", ret.nit)
    print("# eps:", ret.eps)
    print("# logZ:", np.array_str(ret.logZ, max_line_width=np.inf))

    if outfile is None:
        for i in ret.logW:
            print(i)
    else:
        with open(outfile, "w") as f:
            for i in ret.logW:
                print("%8.4f"%i, file=f)

@command("jrun", help="Run jupyter server",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("-d", "--dry-run", action='store_true',
     help="show command instead of executing it")
@arg("--lab", help="use jupyterlab", action="store_true")
@arg("--port", help="set port", default=0, type=int)
@arg("--screen-cmd", help="screen command", default="screen")
@arg("--screen-log", help="screen logfile (no logfile by default)", default="")
@arg("--python-exec", help="python executable", default="")
@arg("-S", "--sockname", help="screen sockname", default="(path):(port):jupyter")
@arg("--no-screen", help="do not run screen", action="store_true")
@arg("--keep-ld-library-path", help="(ignored, this is the default now)", action="store_true")
@arg("--detach", help="detach screen", action="store_true")
def _jrun(**kargs):
    """
       This is a tool to run a jupyter server within a screen command.
       The typical usage would be
       ```bash
       cd /path/to/your/notebook/dir
       bussilab jrun
       ```
       A free port is identified first (can be overridden with the `--port`
       option) and a jupyter server is then run  inside a `screen` instance.
       You will thus have to type `CTRL+aCTRL+d` in order to detach the screen
       letting it run in the background.

       Alternatively, you can immediately detach the screen with
       ```bash
       cd /path/to/your/notebook/dir
       bussilab jrun --detach
       ```

       Notice that, since the server is run inside a screen instance, in order
       to visualize python outputs that has been sent directly to the terminal you
       should connect to the screen instance later. By default, a socket name
       containing the path where the server is running is used, with `/` replaced
       by `:`. It should thus be easy to use `screen -ls` to find the proper screen
       instance.
    """
    from . import jremote
    if kargs["keep_ld_library_path"]:
        warnings.warn("--keep-ld-library-path is the default now - this flag might be removed in the future", FutureWarning)
    kargs["keep_ld_library_path"]=True
    jremote.run_server(**kargs)

@command("jremote", help="Run jupyter client",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("server", help="server URL (e.g. giorgione.phys.sissa.it)")
@arg("-d", "--dry-run", action='store_true',
     help="show command instead of executing it")
@arg("-l", "--list-only", action='store_true',
     help="only report a list of servers")
@arg("--port", help="set port", default=0, type=int)
@arg("-i", "--index", type=int, help="choose server, by default interactive choice", default=0)
@arg("--python-exec",
     help="remote python executable (e.g. module load python3 python-home; python)",
     default="python")
@arg("--server-url", help="URL on server (default: choose interactively)", default="")
@arg("--open-cmd", help="open command (detected automatically by default)", default="")
def _jremote(**kargs):
    """
       This is a tool to connect to a remote running jupyter server.
       The typical usage would be
       ```bash
       bussilab jremote server.url
       ```

       A list of jupyter servers running on the selected machines will be
       shown, and one of them can be picked typing its progressive number.
       In case there is a single server running, it will be opened by default.

       Notice that if the name of the python executable on the server is
       different from plain `python` you can override it with
       ``--python-exec``. You can also run other scripts before, for instance
       loading relevant modules:
       ```bash
       bussilab jremote giorgione.phys.sissa.it \
           --python-exec "module load python3 python-home; python3"
       ```

       If you recurrently connect to the same workstation, it is convenient
       to write a small script like this one, call it `jremote` and put it in
       your path:
       ```bash
       export PYTHONPATH=/path/to/bussilab/source
       python -m bussilab jremote giorgione.phys.sissa.it --python-exec "module load python3 python-home ; python3"
       ```
       Replace `python` with the name of your Python interpreter if needed.

    """
    from . import jremote
    jremote.remote(**kargs)

@command("pip_upgrade_all", help="Upgrade all packages with pip",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("--user", action='store_true',
     help="install/upgrade in user location")
def _pip_upgrade_all(**kargs):
    """
       This is a tool to upgrade all your packages with pip.
       It is a convenient way to upgrade all the packages without
       the need to list them explicitly.

       Warning: this uses pip, so it might not work as expected
       if you are working in conda.

       The typical usage would be
       ```bash
       bussilab pip_upgrade_all
       ```
       If you installed packages in your home you should use
       ```bash
       bussilab pip_upgrade_all --user
       ```
    """
    from . import pip
    pip.upgrade_all(**kargs)

@command("notify", help="Send a notification to Slack",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("-m", "--message", help="message")
@arg("--markdown-file", help="path to a standard Markdown message")
@group(exclusive=True)
@arg("-c","--channel", help="channel (check ~/.bussilabrc by default)")
@arg("-u","--update", help="url of the message to be updated")
@arg("-d","--delete", help="url of the message to be deleted")
@arg("-r","--reply", help="url of the message to be replied")
@arg("-R","--reply-broadcast", help="url of the message to be broadcast-replied")
@arg("-X","--react", help="react to a message")
@endgroup
@arg("-f","--file", nargs="+",
     help="path(s) to files to be uploaded (incompatible with -u and -d)")
@arg("-t","--title", help="title of the message")
@arg("--no-footer", help="ignore footer", action="store_true")
@arg("--no-unfurl", help="disable link and media previews", action="store_true")
@arg("--screenlog", help="screenlog file")
@arg("--screenlog-maxlines", help="maximum number of lines in screenlog (0 means all)", default=0, type=int)
@arg("--type", help="'plain_text' or 'mrkdwn' (use --markdown-file for standard Markdown)", default='mrkdwn')
@arg("--token", help="token (check ~/.bussilabrc by default)")
@arg("-q","--quiet", help="quiet (do not write output)", action="store_true")
def _notify(**kargs):
    """
       This is a tool to send a notification to Slack.
       See the documentation of `bussilab.notify`.
    """
    from . import notify
    quiet=kargs["quiet"]
    del kargs["quiet"]
    if "markdown_file" in kargs:
       kargs["type"]="markdown"
    if kargs["no_footer"]:
       kargs["footer"]=False
    del kargs["no_footer"]
    if kargs["no_unfurl"]:
       kargs["unfurl"]=False
    del kargs["no_unfurl"]
    msg=notify.notify(**kargs)
    if not quiet:
      print(msg)

@command("cron", help="Run cron",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@arg("--quick-start", help="run immediately", action="store_true")
@arg("--quick-start-skip-steps", help="skip steps on quick start", default=None, type=int)
@arg("--quick-start-event", help="event number for quick start", default=None, type=int)
@arg("--cron-file", help="path to cron file")
@arg("--screen-cmd", help="screen command", default="screen")
@arg("--screen-log", help="screen logfile (no logfile by default)", default="")
@arg("--no-screen", help="do not run screen", action="store_true")
@arg("--keep-ld-library-path", help="(ignored, this is the default now)", action="store_true")
@arg("-S", "--sockname", help="screen sockname", default="(path):cron")
@arg("--python-exec", help="python executable", default="")
@arg("--detach", help="detach screen", action="store_true")
@arg("--unique", help="allow only one screen with this socket", action="store_true")
@arg("--window", help="run a new window within the same screen", action="store_true")
@arg("--period", help="period (seconds) default read from cron file or set to 3600", default=None, type=int)
@arg("--max-times", help="maximum number of calls", default=None, type=int)
def _cron(**kargs):
    """Run scheduled steps from a YAML cron configuration.

       Use `--cron-file` to select a file; otherwise the package configuration
       is used. The `cron` section can be a list of steps or a mapping with
       `steps` and an optional `period` in seconds. By default, the command
       starts in a GNU Screen session;
       `--no-screen` runs it in the current terminal. `--quick-start` runs the
       first iteration immediately, and `--max-times` limits iterations.
    """
    from . import cron
    if kargs["keep_ld_library_path"]:
        warnings.warn("--keep-ld-library-path is the default now - this flag might be removed in the future", FutureWarning)
    kargs["keep_ld_library_path"]=True
    cron.cron(**kargs)

@command("required", help="print requirements",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
@group(exclusive=True)
@arg("--macports", help="macports syntax", action="store_true")
@arg("--conda", help="conda syntax", action="store_true")
@endgroup
@arg("--pyver", help="pyversion (e.g. 38), for macports only", default="")
def _required(macports: bool = False, conda: bool = False, pyver: str = ""):
    """Print package requirements for pip, conda, or MacPorts."""
    if conda:
        from . import required_conda
        print(" ".join(required_conda()))
    elif macports:
        from . import required_macports
        print(required_macports(pyver))
    else:
        from . import required_pip
        print(required_pip())


# Here we create a general eparser to be used from cli with prog=""
_eparser = _create_eparser("")
