import conf

#import ade20k.utils as utils

#a = utils.adeindex.load()
#c = utils.configuration.load(a, "ade20k/filters.json")

with open("palette.html","w") as f:
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