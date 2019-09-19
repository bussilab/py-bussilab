"""
Tools to implement the command line interface
=============================================

This module is used internally to implement the command line interface.
In addition, it can be used to run the commands that are available
in the command line interface without the need to leave python:

```python
from bussilab.cli import cli
cli("-h")
cli("wham -b bias") # provide the command line as a string
cli(["wham", "-b", "bias"]) # alternatively use a list
```

The documentation of all the commands can be found in the
[cli_documentation](cli_documentation.html) submodule.

Notice however that these commands typically have alternative python
implementations that allow you to work directly on data structures
and are thus more flexible.
For instance, `bussilab.wham.wham(bias)`, where `bias` is a numpy array,
is often more convenient than `bussilab.cli.cli("wham -b bias")`,
where `bias` is a file.

"""

# Minimal imports here.

# Support argument parsing:
import argparse

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
                 subparsers_help: Dict[str, str]
                ):
        self.parser = parser
        self.subparsers_help = subparsers_help

def _create_eparser(prog: Optional[str] = None) -> _ExtendedParser:
    """Create a `parser` object."""
    parser = argparse.ArgumentParser(prog=prog, description="command line tool")
    parser.add_argument("--version", action='store_true', dest="_version")
    subparsers = parser.add_subparsers(title="Subcommands")
    subparsers_help = {}
    for a in _commands:
        (name, help, description, func, kwargs) = a
        p = [subparsers.add_parser(name, help=help, description=description, **kwargs)]
        f = p[0]
        for call in reversed(func.calls):
            p = call(p)
        if len(p) > 1:
            msg = "Non matching group/endgroup"
            raise TypeError(msg)
        p[0].set_defaults(_func=func.func)
        subparsers_help[name] = f.format_help()
    return _ExtendedParser(parser, subparsers_help)

def command(name: str, help: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator that registers a function as a subcommand.

       This decorator should be written **before** the other decorators
       `bussilab.cli.arg`, `bussilab.cli.group`, and `bussilab.cli.endgroup`.

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
        func = _Argument(func)
        _commands.append((name, help, description, func, kwargs))
        return func
    return add_command

def arg(*name, **kwargs):
    """ Decorator that adds an argument to a command line tool.

        Parameters are passed to the `parser.add_argument()` function.
        It should be written **after** the `bussilab.cli.command` decorator.
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

        It should be written **after** the `bussilab.cli.command` decorator.
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
        required = False # default for add_mutually_exclusive_group
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
        return p[1:]
    def end_group(func):
        func = _Argument(func)
        func.calls.append(call)
        return func
    if f:
        return end_group(f)
    return end_group

def cli(arguments: Union[str, List[str]] = "",
        *,
        prog: str = "",
        use_argcomplete: bool = False
       ) -> Optional[int]:
    """Executes a command line tool from python.

       This is the main function of this module and allows to launch all the subcommands available
       in the command line interface directly from python.

       Parameters
       ----------
       arguments : str or list
           Command line arguments. If a string is passed, it is first split using
           spaces as separators.

       prog : str
           Name of the calling program. It is used to build help texts. **Mostly for internal use**.

       use_argcomplete : bool
           If True, the `autocomplete` function of `argcomplete` module is called on the parser,
           so as to allow autocompletion in the command line tool.
           If `argcomplete` module is not installed, nothing is done and no failure is reported.
           **Mostly for internal use**.

       Returns
       -------
       None or int
           If an error happens while parsing, it returns the corresponding error code.
           Otherwise it returns None.
           If an error happens while executing the requested command, an exception is thrown.
    """

    # allow passing a single string
    if isinstance(arguments, str):
        arguments = arguments.split()

    func = None

    if prog == "":
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

################################################################################

# BELOW ARE ALL THE COMMAND LINE TOOLS

# These functions are made not available from python by their decorators.
# Only to be used from cli() or from the command line.

@command("list", help="List available python modules.")
def _list_submodules():
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
@arg("-o", "--out", help="Output file with weights")
@arg("--use-frame-weight", action="store_true")
@arg("--traj-weight", nargs="*", type=float)
@arg("-T", "--temperature", type=float, default=1.0,
     help="system temperature in energy units", dest="T")
@arg("-m", "--maxiter", type=int, default=1000, help="maximum number of iterations")
@arg("-t", "--threshold", type=int, default=1e-40, help="threshold for convergence")
@arg("-v", "--verbose", action="store_true")
def _wham(**args):
    from . import wham
    import numpy as np
    # load bias file:
    args["bias"] = np.loadtxt(args["bias"])

    # with --use-frame-weight, last column in bias is interpreted as frame weight
    if "use_frame_weight" in args:
        if args["use_frame_weight"]:
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

# Here we create a general eparser to be used from cli with prog=""
_eparser = _create_eparser("")
