import ast
import re
from bussilab import required_conda
from bussilab import __version__

def description():
    from bussilab import __doc__ as doc
    return doc.partition('\n')[0]

with open('conda/meta.yaml.in') as f:
    recipe=f.read()

recipe=re.sub("__VERSION__",__version__,recipe)

match=re.search("( *)(__REQUIRED__)",recipe)

requirements=""

for r in ast.literal_eval(required_conda()):
    requirements+=match.group(1)+"- " + r+"\n"

recipe=re.sub("( *)(__REQUIRED__)\n",requirements,recipe)

recipe=re.sub("__SUMMARY__",description(),recipe)
recipe=re.sub("__DESCRIPTION__",description(),recipe)

with open('conda/meta.yaml',"w") as f:
    f.write(recipe)
