"""
Documentation for command line tools
====================================

This module only contains the documentation of the subcommands used in the
command line interface.
This documentation is generated automatically and is equivalent to the one
that you can see by using the `-h` option with each subcommand.

_subparsers_help_
"""

from .cli import _create_eparser

# Then we process the docstring in order to include the generated documentation
def _process_doc(doc):
    import re
    eparser = _create_eparser("bussilab")
    _subparsers_help_ = ""
    for name in eparser.subparsers_help.keys():
        _subparsers_help_ = (
            _subparsers_help_+"## "+name+"\n```text\n"+eparser.subparsers_help[name]+"```\n"
        )
    doc = re.sub("_subparsers_help_", _subparsers_help_, doc)
    return doc

__doc__ = _process_doc(__doc__)
