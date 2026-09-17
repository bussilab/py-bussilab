import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _setup_keyword(name):
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setup"):
            for keyword in node.keywords:
                if keyword.arg == name:
                    return ast.literal_eval(keyword.value)
    raise AssertionError("setup() has no {!r} keyword".format(name))


def _launcher_versions():
    tree = ast.parse((ROOT / "bin" / "bussilab").read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "PYTHON_VERSIONS"
                        for target in node.targets)):
            return ast.literal_eval(node.value).split()
    raise AssertionError("bin/bussilab has no PYTHON_VERSIONS assignment")


def test_supported_python_versions_are_consistent():
    prefix = "Programming Language :: Python :: "
    setup_versions = [
        classifier[len(prefix):]
        for classifier in _setup_keyword("classifiers")
        if classifier.startswith(prefix)
    ]

    assert _launcher_versions() == setup_versions
    assert _setup_keyword("python_requires") == ">=" + setup_versions[0]
