import ast
import json
from pathlib import Path
import re


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
    latest_major, latest_minor = map(int, setup_versions[-1].split("."))
    next_version = "{}.{}".format(latest_major, latest_minor + 1)
    assert _setup_keyword("python_requires") == (
        ">=" + setup_versions[0] + ",<" + next_version
    )

    workflow = (ROOT / ".github/workflows/python-package.yml").read_text()
    matrix = re.search(r"^\s*python-version: (\[[^\n]+\])$", workflow, re.M)
    assert matrix is not None
    assert json.loads(matrix.group(1)) == setup_versions

    config = (ROOT / "conda/conda_build_config.yaml").read_text()
    conda_versions = re.findall(r'^\s+- "(\d+\.\d+)"$', config, re.M)
    assert conda_versions == list(reversed(setup_versions))

    recipe = (ROOT / "conda/meta.yaml.in").read_text()
    python_spec = "- python >=" + setup_versions[0] + ",<" + next_version
    assert recipe.count(python_spec) == 2
