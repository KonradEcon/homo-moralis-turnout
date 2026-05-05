import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple
from ex_ante_functions import *
import scipy.optimize as sco
import warnings

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["axes.xmargin"] = 0
mpl.rcParams["axes.ymargin"] = 0
mpl.rcParams["axes.axisbelow"] = True
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.framealpha"] = 0.75
mpl.rcParams["legend.fancybox"] = True
mpl.rcParams["legend.facecolor"] = "white"
mpl.rcParams["legend.edgecolor"] = "0.7"

viridis_map = mpl.colormaps.get_cmap('viridis') # getting some colors
viridis = viridis_map(np.linspace(0,1,7))
consistent_linewidth = 1.5
equilibrium_marker_outer_size = 70
equilibrium_marker_inner_size = 4
equilibrium_marker_zorder = 5
equilibrium_marker_facecolor = (1.0,1.0,1.0,0.5)

def set_axis_bounds(ax, xlim=None, ylim=None):
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_zorder(0)
    for line in ax.lines:
        line.set_clip_on(False)
    for collection in ax.collections:
        if isinstance(collection,(mpl.collections.PathCollection,mpl.collections.LineCollection)):
            collection.set_clip_on(False)
    ax.margins(x=0,y=0)
    ax.autoscale(enable=True, axis="both", tight=True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

def turnout_bounds(a0,av,b0,bv):
    return (min(a0,b0),max(a0+av,b0+bv))

def plot_bullseye(ax,x,y,color,outer_size=equilibrium_marker_outer_size,inner_size=equilibrium_marker_inner_size,zorder=equilibrium_marker_zorder):
    ax.scatter(
        x,
        y,
        facecolors=[equilibrium_marker_facecolor],
        edgecolors=color,
        linewidths=1.4,
        marker="o",
        s=outer_size,
        zorder=zorder,
    )
    ax.scatter(
        x,
        y,
        color=color,
        marker="o",
        s=inner_size,
        zorder=zorder + 0.1,
    )

def bullseye_legend_handle(color,outer_size=equilibrium_marker_outer_size,inner_size=equilibrium_marker_inner_size):
    return (
        mpl.lines.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor=equilibrium_marker_facecolor,
            markeredgecolor=color,
            markeredgewidth=1.4,
            markersize=np.sqrt(outer_size),
        ),
        mpl.lines.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            color=color,
            markersize=np.sqrt(inner_size),
        ),
    )

def deduplicate_points(x,y,dx,dy):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.empty((0,2))

    points = np.column_stack((x[mask],y[mask]))
    dedup_dx = max(dx / 2,1e-12)
    dedup_dy = max(dy / 2,1e-12)
    x0 = np.min(points[:,0])
    y0 = np.min(points[:,1])
    keys = np.column_stack((
        np.rint((points[:,0] - x0) / dedup_dx).astype(np.int64),
        np.rint((points[:,1] - y0) / dedup_dy).astype(np.int64),
    ))
    _, unique_idx = np.unique(keys,axis=0,return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx]

def estimate_curve_steps(x,y,dx_fallback,dy_fallback):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return (max(dx_fallback,1e-12),max(dy_fallback,1e-12))

    x = x[mask]
    y = y[mask]

    unique_x = np.unique(np.sort(x))
    if len(unique_x) > 1:
        x_diffs = np.diff(unique_x)
        x_diffs = x_diffs[x_diffs > 1e-12]
        if len(x_diffs) > 0:
            dx_est = max(dx_fallback,np.median(x_diffs))
        else:
            dx_est = max(dx_fallback,1e-12)
    else:
        dx_est = max(dx_fallback,1e-12)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    y_step_candidates = []
    x_window = 1.5 * dx_est
    for i in range(0,len(x_sorted)-1):
        dx_local = x_sorted[i+1] - x_sorted[i]
        if dx_local > 1e-12 and dx_local <= x_window:
            dy_local = np.abs(y_sorted[i+1] - y_sorted[i])
            if dy_local > 1e-12:
                y_step_candidates.append(dy_local)

    if len(y_step_candidates) > 0:
        dy_est = max(dy_fallback,np.median(y_step_candidates))
    else:
        dy_est = max(dy_fallback,1e-12)

    return (dx_est,dy_est)

def build_local_graph(points,dx,dy):
    adjacency = [set() for _ in range(len(points))]
    if len(points) <= 1:
        return adjacency

    x_tol = 1.75 * dx
    y_tol = 1.75 * dy
    order = np.argsort(points[:,0])
    for pos in range(0,len(order)):
        i = order[pos]
        for next_pos in range(pos+1,len(order)):
            j = order[next_pos]
            if points[j,0] - points[i,0] > x_tol:
                break
            x_scaled = (points[j,0] - points[i,0]) / max(dx,1e-12)
            y_scaled = np.abs(points[j,1] - points[i,1]) / max(dy,1e-12)
            if np.abs(points[j,1] - points[i,1]) <= y_tol and x_scaled**2 + y_scaled**2 <= 7.5:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency

def connected_components(adjacency,points):
    visited = np.zeros(len(adjacency),dtype=np.bool_)
    components = []
    for i in range(0,len(adjacency)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = []
        while len(stack) > 0:
            current = stack.pop()
            component.append(current)
            for nbr in adjacency[current]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        components.append(component)
    components.sort(key=lambda comp: np.min(points[comp,0]))
    return components

def scaled_distance(point_a,point_b,dx,dy):
    return np.sqrt(
        ((point_a[0] - point_b[0]) / max(dx,1e-12))**2
        + ((point_a[1] - point_b[1]) / max(dy,1e-12))**2
    )

def build_component_mst(component,adjacency,points,dx,dy):
    if len(component) <= 1:
        return {node: set() for node in component}

    component_set = set(component)
    tree_adjacency = {node: set() for node in component}
    visited = {component[0]}

    while len(visited) < len(component):
        best_edge = None
        best_weight = np.inf
        for node in visited:
            for nbr in adjacency[node]:
                if nbr not in component_set or nbr in visited:
                    continue
                weight = scaled_distance(points[node],points[nbr],dx,dy)
                if weight < best_weight:
                    best_weight = weight
                    best_edge = (node,nbr)

        if best_edge is None:
            return None

        node,nbr = best_edge
        tree_adjacency[node].add(nbr)
        tree_adjacency[nbr].add(node)
        visited.add(nbr)

    return tree_adjacency

def order_line_component(component,adjacency,points,dx,dy):
    if len(component) == 1:
        return None

    tree_adjacency = build_component_mst(component,adjacency,points,dx,dy)
    if tree_adjacency is None:
        return None

    degrees = {node: len(tree_adjacency[node]) for node in component}
    if any(degrees[node] > 2 for node in component):
        return None

    endpoints = [node for node in component if degrees[node] == 1]
    if len(component) == 2:
        if len(endpoints) != 2:
            return None
    elif len(endpoints) != 2:
        return None

    start = endpoints[0]
    ordered = [start]
    visited = {start}
    prev = None
    current = start

    while True:
        nbrs = [nbr for nbr in tree_adjacency[current] if nbr != prev]
        if len(nbrs) == 0:
            break
        if len(nbrs) > 1:
            return None
        nxt = nbrs[0]
        if nxt in visited:
            return None
        ordered.append(nxt)
        visited.add(nxt)
        prev = current
        current = nxt

    if len(ordered) != len(component):
        return None
    return ordered

def plot_point_graph(ax,x,y,color,dx,dy,warning_label):
    points = deduplicate_points(x,y,dx,dy)
    if len(points) == 0:
        return None

    adjacency = build_local_graph(points,dx,dy)
    components = connected_components(adjacency,points)
    first_line_handle = None
    first_scatter_handle = None
    for i in range(0,len(components)):
        comp_points = points[components[i],:]
        ordered = order_line_component(components[i],adjacency,points,dx,dy)
        if ordered is None:
            scatter_handle = ax.scatter(comp_points[:,0],comp_points[:,1],color=color,s=1)
            if first_scatter_handle is None:
                first_scatter_handle = scatter_handle
            if len(components[i]) > 1:
                warnings.warn(
                    warning_label + ": component " + str(i+1) + " is not line-shaped; plotting as scatter.",
                    UserWarning,
                )
        else:
            line_points = points[ordered,:]
            line_handle, = ax.plot(line_points[:,0],line_points[:,1],color=color,linewidth=consistent_linewidth)
            if first_line_handle is None:
                first_line_handle = line_handle
    if first_line_handle is not None:
        return first_line_handle
    return first_scatter_handle

def equilibrium_curve_label(variable_symbol,total,index):
    if total == 1:
        return "Equilibrium"
    return rf"${variable_symbol}={variable_symbol}_{index+1}$"

def plot_b_equilibria_case(m,kap,rho,the,a0,av,b0,bv,eps,name,theta_label=r"\theta"):
    b = get_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlabel("$b^i$")
    ax.set_ylabel(r"$EU_B^\kappa(a_0,a_0,b,b^i)$")
    viridis_new = viridis_map(np.linspace(0,1,len(b)+1))
    b_vec = np.linspace(b0,b0+bv,500)
    for i in range(0,len(b)):
        print(b[i])
        plot_bullseye(ax,b[i],utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b[i]),viridis_new[i])
        ax.plot(
            b_vec,
            utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b_vec),
            color=viridis_new[i],
            label=equilibrium_curve_label("b",len(b),i),
        )

    fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',' + theta_label + r'=' + str(the) +  r',$',ha="center")
    fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
    if len(b) > 0:
        ax.legend(
            handles=[bullseye_legend_handle("black")],
            labels=[r"Equilibria ($b$)"],
            handler_map={tuple: HandlerTuple(ndivide=1)},
        )
    set_axis_bounds(ax, xlim=(b0,b0+bv))
    plt.savefig("1d_plots/" + name + "_equilibria.png",format="png",dpi=600,bbox_inches="tight")
    plt.close()

# plotting h function
xs=np.linspace(-1, 1, 200)

plt.plot(xs,h(xs,0.1), label='m=0.1',color=viridis[0])
plt.plot(xs,h(xs,2), label='m=2',color=viridis[1],linestyle="dashdot")
plt.plot(xs,h(xs,10), label='m=10',color=viridis[2],linestyle="dotted")
plt.plot(xs,h(xs,100), label='m=100',color=viridis[3],linestyle="--")
plt.legend()
set_axis_bounds(plt.gca(), xlim=(-1.0,1.0))
plt.savefig("1d_plots/arctan_m.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting multiple equilibria for m growing
# defining some baseline parameters
m = 12.0
kap = 0.2
rho = 2.67
the = 2.0
av = 1.25
bv = 0.75
a0 = 0.5
b0 = 0.1
eps = 1e-10

# plotting a three equilibria case

b = get_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel("$b^i$")
ax.set_ylabel(r"$EU_B^\kappa(a_0,a_0,b,b^i)$")
viridis_new = viridis_map(np.linspace(0,1,len(b)+1))
b_vec = np.linspace(b0,b0+bv,500)
for i in range(0,len(b)):
    print(b[i])
    plot_bullseye(ax,b[i],utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b[i]),viridis_new[i])
    ax.plot(
        b_vec,
        utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b_vec),
        color=viridis_new[i],
        label=equilibrium_curve_label("b",len(b),i),
    )

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
if len(b) > 0:
    ax.legend(
        handles=[bullseye_legend_handle("black")],
        labels=[r"Equilibria ($b$)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
set_axis_bounds(ax, xlim=(b0,b0+bv))
plt.savefig("1d_plots/ex_ante_3_eqs.png",format="png",dpi=600,bbox_inches="tight")
plt.close()

# plotting equilibria with m growing

m_axis_log = np.linspace(-0.5,6,20000)
m_arr = np.zeros(0)
b_arr = np.zeros(0)
for i in range(0,len(m_axis_log)):
    m_new = np.exp(np.log(10)*m_axis_log[i])
    eqs = get_equilibria_b(m_new,the,kap,a0,b0,av,rho,bv,eps)
    m_arr = np.append(m_arr,m_new*np.ones(len(eqs)))
    b_arr = np.append(b_arr,eqs)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(m_arr,b0*np.ones(len(m_arr)),label='$b_0$',color=viridis[0],linestyle="--")
ax.plot(m_arr,a0*np.ones(len(m_arr)),label='$a_0$',color=viridis[0],linestyle="dotted")
ax.scatter(m_arr,b_arr,s=1,color=viridis[3])
ax.plot([],[],label='Equilibria',color=viridis[3])
ax.set_xscale('log')
ax.set_xlabel("$m$")
ax.set_ylabel("$b$")

fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
plt.legend()
set_axis_bounds(
    ax,
    xlim=(np.min(m_arr),np.max(m_arr)),
    ylim=(min(np.min(b_arr),a0,b0),max(np.max(b_arr),a0,b0)),
)
plt.savefig("1d_plots/ex_ante_limit.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# two equilibria case

b0 = 0.1
a0 = 0.45
bv = 0.4
av = 0.6
rho = 3.0
kap  =0.4
m = 14.0
the = 2.0

b = get_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel("$b^i$")
ax.set_ylabel(r"$EU_B^\kappa(a_0,a_0,b,b^i)$")
viridis_new = viridis_map(np.linspace(0,1,len(b)+1))
b_vec = np.linspace(b0,b0+bv,500)
for i in range(0,len(b)):
    print(b[i])
    plot_bullseye(ax,b[i],utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b[i]),viridis_new[i])
    ax.plot(
        b_vec,
        utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b_vec),
        color=viridis_new[i],
        label=equilibrium_curve_label("b",len(b),i),
    )

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
if len(b) > 0:
    ax.legend(
        handles=[bullseye_legend_handle("black")],
        labels=[r"Equilibria ($b$)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
set_axis_bounds(ax, xlim=(b0,b0+bv))
plt.savefig("1d_plots/ex_ante_2_eqs.png",format="png",dpi=600,bbox_inches="tight")
plt.close()

# The ex-ante script has a single theta parameter. For this B-winning case (rho * bv > av),
# use theta_B to generate the nonpartisan equilibrium plot.
m = 20.0
kap = 0.35
rho = 10.0
the = 1.8
a0 = 0.85
av = 1.65
b0 = 0.2
bv = 0.7

plot_b_equilibria_case(
    m,
    kap,
    rho,
    the,
    a0,
    av,
    b0,
    bv,
    eps,
    "turnout_higher_with_nonpartisan",
    theta_label=r"\theta_B",
)

# finding and plotting a knife-edge case for two equilibria with kappa = 1

# kappa = 1 and other parameters except bv selected to get a knife-edge case
m = 40.0
kap = 1.0
rho = 2.0
the = 6.0
a0 = 0.55
av = 0.75
b0 = 0.2

# finding the knife-edge case by varying bv
def difference(bv):
    roots = np.real(roots_aux_b(m,the,kap,a0,b0,av,rho,bv))
    return utility_b(m,the,kap,a0,b0,av,rho,bv,roots[2],roots[2]) - utility_b(m,the,kap,a0,b0,av,rho,bv,roots[0],roots[0])

bv_knife = sco.brentq(difference,0.5,1.0)
print(bv_knife)
vec_difference = np.vectorize(difference)
bv_vec = np.linspace(0.5,0.75,500)
plt.plot(bv_vec,vec_difference(bv_vec),color=viridis[0])
plt.scatter(bv_knife,0,color=viridis[0],marker="*",s=15,label=r"$b_v^*$")
plt.xlabel("$b_v$")
plt.ylabel(r"$EU_B^1(a_0,a_0,b_h,b_h)-EU_B^1(a_0,a_0,b_\ell,b_\ell)$")
plt.legend()
set_axis_bounds(plt.gca(), xlim=(bv_vec[0],bv_vec[-1]))
plt.savefig("1d_plots/knife_edge_diff_between_local_maxes.png",format="png",dpi=600,bbox_inches='tight')
plt.close()


# plotting the knife-edge case
fig, ax = plt.subplots(figsize=(5,5))
legend_handles = []
legend_labels = []

def knife_edge_plot_equilibria_b(bv_plot):
    eqs_plot = np.sort(get_equilibria_b(m,the,kap,a0,b0,av,rho,bv_plot,eps))
    if np.isclose(bv_plot,bv_knife):
        raw_roots = roots_aux_b(m,the,kap,a0,b0,av,rho,bv_plot)
        raw_roots = np.real(raw_roots[np.abs(np.imag(raw_roots)) < eps])
        raw_roots = raw_roots[raw_roots > b0]
        raw_roots = raw_roots[raw_roots < b0 + bv_plot]
        raw_roots = np.sort(raw_roots)
        if len(raw_roots) >= 2:
            eqs_plot = np.array([raw_roots[0],raw_roots[-1]])
    return eqs_plot

comparison_bvs = [
    (0.55, viridis[1], "dotted", r"$b_v = 0.55$"),
    (0.65, viridis[2], "dashdot", r"$b_v = 0.65$"),
    (bv_knife, viridis[0], "solid", r"$b_v \approx" + str(round(bv_knife,3)) + r"$"),
]
for bv_plot, color, linestyle, label in comparison_bvs:
    b_vec = np.linspace(b0,b0+bv_plot,500)
    line_handle, = ax.plot(
        b_vec,
        utility_b(m,the,kap,a0,b0,av,rho,bv_plot,b_vec,b_vec),
        color=color,
        linestyle=linestyle,
        label=label,
    )
    legend_handles.append(line_handle)
    legend_labels.append(label)
    eqs_plot = knife_edge_plot_equilibria_b(bv_plot)
    for i in range(0,len(eqs_plot)):
        plot_bullseye(
            ax,
            eqs_plot[i],
            utility_b(m,the,kap,a0,b0,av,rho,bv_plot,eqs_plot[i],eqs_plot[i]),
            color,
        )
b_vec = np.linspace(b0,b0+bv_knife,500)
eqs = knife_edge_plot_equilibria_b(bv_knife)
ax.plot(b_vec,len(b_vec)*[utility_b(m,the,kap,a0,b0,av,rho,bv_knife,eqs[0],eqs[0])],color=viridis[0],linestyle="--")
ax.set_xlabel("$b^i$")
ax.set_ylabel("$EU_B^1(a_0,a_0,b,b^i)$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r'$',ha="center")
ax.legend(
    handles=[bullseye_legend_handle("black")] + legend_handles,
    labels=[r"Equilibria"] + legend_labels,
    handler_map={tuple: HandlerTuple(ndivide=1)},
)
set_axis_bounds(ax, xlim=(b0,b0+max(0.55,0.65,bv_knife)))
plt.savefig("1d_plots/ex_ante_knife_edge_kappa_is_1.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting all equilibria when av starts exceeding rho bv

b0 = 0.4
a0 = 0.5
bv = 0.4
rho = 1.5
kap  =0.4
m = 14.0
the = 2.0

fig, ax = plt.subplots(figsize=(5,5))
av_arr = np.linspace(0.3,0.9,500)
av_a_eqs = np.zeros(0)
av_b_eqs = np.zeros(0)
a_eqs = np.zeros(0)
b_eqs = np.zeros(0)
for i in range(0,len(av_arr)):
    eqs_b = get_equilibria_b(m,the,kap,a0,b0,av_arr[i],rho,bv,eps)
    eqs_a = get_equilibria_a(m,the,kap,a0,b0,av_arr[i],rho,bv,eps)
    av_a_eqs = np.append(av_a_eqs,av_arr[i]*np.ones(len(eqs_a)))
    av_b_eqs = np.append(av_b_eqs,av_arr[i]*np.ones(len(eqs_b)))
    a_eqs = np.append(a_eqs,eqs_a)
    b_eqs = np.append(b_eqs,eqs_b)

ax.plot(av_arr,b0*np.ones(len(av_arr)),label='$b_0$',color=viridis[5],linestyle="--")
ax.plot(av_arr,a0*np.ones(len(av_arr)),label='$a_0$',color=viridis[0],linestyle="--")

ax.scatter(av_a_eqs,a_eqs,s=1,color=viridis[0])
ax.plot([np.nan],[np.nan],label='$a$ in equilibrium',color=viridis[0])
ax.scatter(av_b_eqs,b_eqs,s=1,color=viridis[5])
ax.plot([np.nan],[np.nan],label='$b$ in equilibrium',color=viridis[5])
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) + r',b_0=' + str(b0) +r',b_v=' + str(bv) +  r'$',ha="center")
ax.set_xlabel("$a_v$")
ax.set_ylabel("$a,b$")
plt.legend()
set_axis_bounds(
    ax,
    xlim=(av_arr[0],av_arr[-1]),
    ylim=(min(np.min(a_eqs),np.min(b_eqs),a0,b0),max(np.max(a_eqs),np.max(b_eqs),a0,b0)),
)
plt.savefig("1d_plots/ex_ante_around_lambda_0.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting majority coordination problem with high minority base
b0 = 0.5
bv = 0.5
av = 1.0
rho = 1.5
m = 10.0

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.linspace(0.0,1.0,3),3*[b0],color="black",linestyle="--",label=r'$b_0$')
a0_vec = np.linspace(0.0,1.0,500)
a_vec = np.linspace(a0_vec[0],a0_vec[-1] + av,len(a0_vec))
delta_a0 = a0_vec[1] - a0_vec[0]
delta_a = a_vec[1] - a_vec[0]
j=0
for kap in [0.1,0.5,0.8]:
    a0_forward = []
    eqs_forward = []
    for a0 in a0_vec:
        new_eqs = get_equilibria_a(m,the,kap,a0,b0,av,rho,bv,eps)
        for eq in new_eqs:
            a0_forward.append(a0)
            eqs_forward.append(eq)

    a0_reverse,eqs_reverse = find_equilibria_a_veca(
        m,
        the,
        kap,
        b0,
        av,
        rho,
        bv,
        a0_vec[0],
        a0_vec[-1],
        a_vec,
        eps,
    )

    a0_plot = np.concatenate((np.array(a0_forward),a0_reverse))
    eqs_plot = np.concatenate((np.array(eqs_forward),eqs_reverse))
    dx_plot,dy_plot = estimate_curve_steps(a0_plot,eqs_plot,delta_a0,delta_a)
    handle = plot_point_graph(
        ax,
        a0_plot,
        eqs_plot,
        viridis[3*j],
        dx_plot,
        dy_plot,
        "ex_ante_majority_coordination_problem kappa=" + str(kap),
    )
    if handle is None:
        ax.plot([np.nan],[np.nan],label=r'$\kappa = ' + str(kap) + r'$',color=viridis[3*j])
    else:
        handle.set_label(r'$\kappa = ' + str(kap) + r'$')
    j+=1

ax.set_xlabel("$a_0$")
ax.set_ylabel("a")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_v=' + str(av) + r',b_0=' + str(b0) +r',b_v=' + str(bv) +  r'$',ha="center")
plt.legend()
set_axis_bounds(ax, xlim=(a0_vec[0],a0_vec[-1]))
plt.savefig("1d_plots/ex_ante_majority_coordination_problem.png",format="png",dpi=600,bbox_inches='tight')
