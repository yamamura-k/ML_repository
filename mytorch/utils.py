from genericpath import exists
import os
import subprocess
def _dot_var(v, verbose=False):
    name = ""if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f"{v.shape} {v.dtype}"
    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'
def _dot_func(f):
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f"{id(x)} -> {id(f)}\n"
    for y in f.outputs:
        txt += f"{id(f)} -> {id(y())}\n"
    return txt

def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen = set()
    def add_func(f):
        if f in seen:
            return
        seen.add(f)
        funcs.append(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return "digraph g{\n" + txt + "}"

def plot_dot_graph(output, verbose=True, save_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)
    tmp_dir = os.path.join(os.path.expanduser("~"), ".mytorch")
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")
    with open(graph_path, "w") as f:
        f.write(dot_graph)
    extension = os.path.splitext(save_file)[1][1:]
    cmd = f"dot {graph_path} -T {extension} -o {save_file}"
    subprocess.run(cmd, shell=True)