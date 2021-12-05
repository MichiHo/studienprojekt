import os

import ade_utils as utils

ade_index = utils.adeindex.load()

out = os.path.expanduser("~/ssh_transfer/classnames.csv")
with open(out,"w") as file:
    for i,name in enumerate(ade_index['objectnames']):
        file.write(f"{i:4}; {name}")
