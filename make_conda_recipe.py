import ast
import re
from bussilab import required_conda
from bussilab import __version__

with open('conda/meta.yaml.in') as f:
    recipe=f.read()

recipe=re.sub("__VERSION__",__version__,recipe)

match=re.search("( *)(__REQUIRED__)",recipe)

requirements=""

for r in ast.literal_eval(required_conda()):
    requirements+=match.group(1)+"- " + r+"\n"

recipe=re.sub("( *)(__REQUIRED__)\n",requirements,recipe)

with open('conda/meta.yaml',"w") as f:
    f.write(recipe)
