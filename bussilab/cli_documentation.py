"""
Documentation for command line tools
====================================

This page is generated from the registered commands when this module is
imported. Each section includes the corresponding `bussilab COMMAND -h`
output, followed by any longer usage notes from the command's docstring.
For calling commands from Python or registering new commands, see
[`bussilab.cli`](cli.html).

_subparsers_help_
"""

import inspect
from .cli import _create_eparser

# Then we process the docstring in order to include the generated documentation
def _process_doc(doc):
    eparser = _create_eparser("bussilab")
    _subparsers_help_ = ""
    for name in eparser.subparsers_help.keys():
        _subparsers_help_ = (
            _subparsers_help_+"## "+name+"\n```text\n"+eparser.subparsers_help[name]+"```\n"
        )
        _subparsers_help_ = (
            _subparsers_help_+"\n" +
            inspect.cleandoc(eparser.subparsers_doc[name])+"\n\n"
        )
    doc = doc.replace("_subparsers_help_", _subparsers_help_)
    return doc

__doc__ = _process_doc(__doc__)
