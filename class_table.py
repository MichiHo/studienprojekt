"""Create a small html-file containing the classnames, scenes and their colors."""
import argparse

from utils import *


parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(overwrite=False)
parser.add_argument('--output-path', type=path_arg, default="palette.html", help='the path of the output html file. (default: "palette.html")')
parser.add_argument('--conf-path', type=path_arg, default=None, help='path of conf.json, if it deviates from the default.')
parser.add_argument('--overwrite',dest="overwrite", action="store_true", help='whether to overwrite the output file if it is present')
args = parser.parse_args()

if args.conf_path is not None:
    conf = GeneralConfig(args.conf_path)

if os.path.exists(args.output_path) and not args.overwrite:
    if input(f"Output path {args.output_path} exists. Overwrite? [y/n]").lower() not in ['y']:
        exit()

with open(args.output_path,"w") as f:
    f.write("<html><table>")
    f.write("<tr><th>Name</th><th>Scene</th><th>Color</th></tr>")
    for cl in conf.classes:
        f.write(f"""\t<tr>
        <td>{cl.name}</td>
        <td>{cl.scene}</td>
        <td style='color:rgb({cl.color[0]},{cl.color[1]},{cl.color[2]})'>
            ██
        </td>
        </tr>""")
    # f.write("</ul><ul>")
    # for i,cl in c.all_classes.items():
    #     f.write(f"""\t<li>{i:2}
    #     <span style='color:rgb({cl.color[0]},{cl.color[1]},{cl.color[2]})'>
    #         {cl.name}
    #     </span>
    #     ({cl.scene})</li>""")
    f.write("</table></html>")
