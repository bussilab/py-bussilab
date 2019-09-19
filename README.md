README

# Examples

python -m venv examples_venv
source examples_venv/bin/activate
pip install jupyter jupyter_contrib_nbextensions matplotlib numpy argcomplete
source sourceme.sh

rm -fr docdir/ && ~/Library/Python/3.7/bin/pdoc3 -o ^Ccdir bussilab --html

https://www.codacy.com/blog/which-python-static-analysis-tools-should-i-use/
