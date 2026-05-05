from ex_post_funs_simple import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as sco
import warnings
from matplotlib.legend_handler import HandlerTuple

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
viridis = viridis_map(np.linspace(0,1,8))
color_a = "black"
color_b = viridis[6]
consistent_color_a = viridis[1]
consistent_linewidth = 1.5
equilibrium_marker_outer_size = 70
equilibrium_marker_inner_size = 4
deviation_equilibrium_marker_outer_size = equilibrium_marker_outer_size
deviation_equilibrium_marker_inner_size = equilibrium_marker_inner_size
equilibrium_marker_zorder = 5
equilibrium_marker_color = "black"
equilibrium_marker_facecolor = (1.0,1.0,1.0,0.5)
equilibrium_legend_markersize = np.sqrt(equilibrium_marker_outer_size)
equilibrium_legend_inner_markersize = np.sqrt(equilibrium_marker_inner_size)
residual_scatter_size = 4

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

def format_numeric_tick(value):
    rounded_value = np.round(value,2)
    label = f"{rounded_value:.2f}".rstrip("0").rstrip(".")
    if label == "-0":
        label = "0"
    return label

def set_boundary_ticks(ax,a0,av,b0,bv,num_ticks=6):
    x_ticks = np.linspace(a0,a0+av,num_ticks)
    y_ticks = np.linspace(b0,b0+bv,num_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([format_numeric_tick(value) for value in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([format_numeric_tick(value) for value in y_ticks])
    ax.tick_params(axis="x",top=False,labeltop=False,bottom=True,labelbottom=True)
    ax.tick_params(axis="y",right=False,labelright=False,left=True,labelleft=True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

def set_axes_bounds(axes, xlim=None, ylim=None):
    for current_ax in np.ravel(np.atleast_1d(axes)):
        set_axis_bounds(current_ax, xlim=xlim, ylim=ylim)

def turnout_bounds(a0,av,b0,bv):
    return (min(a0,b0),max(a0+av,b0+bv))

def equilibrium_marker_label(total):
    if total == 1:
        return "Equilibrium"
    return "Equilibria"

def equilibrium_pair_label(total,index):
    if total == 1:
        return "Equilibrium"
    return r"$(a,b)=(a_" + str(index+1) + r",b_" + str(index+1) + r")$"

def plot_bullseye(ax,x,y,color,outer_size,inner_size,zorder,label="_nolegend_"):
    ax.scatter(
        x,
        y,
        facecolors=[equilibrium_marker_facecolor],
        edgecolors=color,
        linewidths=1.4,
        marker="o",
        s=outer_size,
        label=label,
        zorder=zorder,
    )
    ax.scatter(
        x,
        y,
        color=color,
        marker="o",
        s=inner_size,
        label="_nolegend_",
        zorder=zorder + 0.1,
    )

def bullseye_legend_handle(color,outer_size,inner_size):
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

def finite_points(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.empty((0,2))
    return np.column_stack((x[mask],y[mask]))

def choose_annotation_position(x0,y0,occupied_points,placed_positions,xlim,ylim):
    x_span = max(xlim[1] - xlim[0],1e-12)
    y_span = max(ylim[1] - ylim[0],1e-12)
    margin_x = 0.025*x_span
    margin_y = 0.025*y_span
    offset_x = 0.018*x_span
    offset_y = 0.018*y_span
    candidate_offsets = [
        (1.0,1.0),
        (-1.0,1.0),
        (1.0,-1.0),
        (-1.0,-1.0),
        (1.3,0.4),
        (-1.3,0.4),
        (1.3,-0.4),
        (-1.3,-0.4),
        (0.4,1.3),
        (-0.4,1.3),
        (0.4,-1.3),
        (-0.4,-1.3),
    ]
    best_candidate = None
    best_score = -np.inf

    for sx,sy in candidate_offsets:
        x_text = x0 + sx*offset_x
        y_text = y0 + sy*offset_y
        if x_text < xlim[0] + margin_x or x_text > xlim[1] - margin_x:
            continue
        if y_text < ylim[0] + margin_y or y_text > ylim[1] - margin_y:
            continue

        if len(occupied_points) > 0:
            dx_occ = (occupied_points[:,0] - x_text) / x_span
            dy_occ = (occupied_points[:,1] - y_text) / y_span
            occupied_score = np.min(dx_occ**2 + dy_occ**2)
        else:
            occupied_score = np.inf

        if len(placed_positions) > 0:
            placed_points = np.asarray(placed_positions)
            dx_lab = (placed_points[:,0] - x_text) / x_span
            dy_lab = (placed_points[:,1] - y_text) / y_span
            placed_score = np.min(dx_lab**2 + dy_lab**2)
        else:
            placed_score = np.inf

        edge_score = min(
            (x_text - xlim[0]) / x_span,
            (xlim[1] - x_text) / x_span,
            (y_text - ylim[0]) / y_span,
            (ylim[1] - y_text) / y_span,
        )
        score = min(occupied_score,placed_score) + 0.25*edge_score
        if score > best_score:
            ha = "left" if sx > 0.1 else ("right" if sx < -0.1 else "center")
            va = "bottom" if sy > 0.1 else ("top" if sy < -0.1 else "center")
            best_candidate = (x_text,y_text,ha,va)
            best_score = score

    if best_candidate is None:
        x_text = min(max(x0 + offset_x,xlim[0] + margin_x),xlim[1] - margin_x)
        y_text = min(max(y0 + offset_y,ylim[0] + margin_y),ylim[1] - margin_y)
        return (x_text,y_text,"left","bottom")
    return best_candidate

def annotate_equilibria(ax,a,b,occupied_points,xlim,ylim):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) <= 1:
        return

    placed_positions = []
    eq_points = finite_points(a,b)
    if len(eq_points) > 0:
        if len(occupied_points) > 0:
            occupied_points = np.vstack((occupied_points,eq_points))
        else:
            occupied_points = eq_points

    for i in range(0,len(a)):
        x_text,y_text,ha,va = choose_annotation_position(
            a[i],
            b[i],
            occupied_points,
            placed_positions,
            xlim,
            ylim,
        )
        ax.text(
            x_text,
            y_text,
            str(i+1),
            color="black",
            ha=ha,
            va=va,
            bbox=dict(
                facecolor="white",
                edgecolor="0.5",
                linewidth=0.6,
                alpha=0.8,
                boxstyle="round,pad=0.15",
            ),
            zorder=6,
        )
        placed_positions.append((x_text,y_text))

def deduplicate_points(x,y,dx,dy):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.empty((0,2))

    points = np.column_stack((x[mask],y[mask]))
    dedup_dx = max(dx/2,1e-12)
    dedup_dy = max(dy/2,1e-12)
    x0 = np.min(points[:,0])
    y0 = np.min(points[:,1])
    keys = np.column_stack((
        np.rint((points[:,0]-x0)/dedup_dx).astype(np.int64),
        np.rint((points[:,1]-y0)/dedup_dy).astype(np.int64),
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
    x_window = 1.5*dx_est
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

    x_tol = 1.75*dx
    y_tol = 1.75*dy
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

    degrees = {}
    for node in component:
        degrees[node] = len(tree_adjacency[node])

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

def decompose_component_paths(component,adjacency,points,dx,dy):
    if len(component) == 0:
        return [], []

    tree_adjacency = build_component_mst(component,adjacency,points,dx,dy)
    if tree_adjacency is None:
        return [], component

    degrees = {node: len(tree_adjacency[node]) for node in component}
    ordered = order_line_component(component,adjacency,points,dx,dy)
    if ordered is not None:
        return [ordered], []

    special_nodes = [node for node in component if degrees[node] != 2]
    if len(special_nodes) == 0:
        return [], component

    paths = []
    used_edges = set()
    for start in special_nodes:
        for nbr in tree_adjacency[start]:
            edge = tuple(sorted((start,nbr)))
            if edge in used_edges:
                continue

            path = [start,nbr]
            used_edges.add(edge)
            prev = start
            current = nbr

            while degrees[current] == 2:
                next_nodes = [node for node in tree_adjacency[current] if node != prev]
                if len(next_nodes) != 1:
                    break
                nxt = next_nodes[0]
                edge = tuple(sorted((current,nxt)))
                if edge in used_edges:
                    break
                path.append(nxt)
                used_edges.add(edge)
                prev = current
                current = nxt

            paths.append(path)

    residual_nodes = [node for node in component if degrees[node] > 2]
    covered_nodes = set()
    for path in paths:
        covered_nodes.update(path)
    for node in component:
        if node not in covered_nodes:
            residual_nodes.append(node)

    residual_nodes = sorted(set(residual_nodes))
    return paths,residual_nodes

def plot_point_graph(ax,x,y,color,dx,dy,warning_label,residual_scatter_only=False):
    points = deduplicate_points(x,y,dx,dy)
    if len(points) == 0:
        return None

    adjacency = build_local_graph(points,dx,dy)
    components = connected_components(adjacency,points)
    first_line_handle = None
    first_scatter_handle = None
    for i in range(0,len(components)):
        comp_points = points[components[i],:]
        if residual_scatter_only:
            paths,residual_nodes = decompose_component_paths(components[i],adjacency,points,dx,dy)
            if len(paths) == 0:
                scatter_handle = ax.scatter(comp_points[:,0],comp_points[:,1],color=color,s=1)
                if first_scatter_handle is None:
                    first_scatter_handle = scatter_handle
                if len(components[i]) > 1:
                    warnings.warn(
                        warning_label + ": component " + str(i+1) + " has no line part; plotting as scatter.",
                        UserWarning,
                    )
                continue

            for path in paths:
                if len(path) >= 2:
                    line_points = points[path,:]
                    line_handle, = ax.plot(line_points[:,0],line_points[:,1],color=color,linewidth=consistent_linewidth)
                    if first_line_handle is None:
                        first_line_handle = line_handle

            if len(residual_nodes) > 0:
                residual_points = points[residual_nodes,:]
                scatter_handle = ax.scatter(residual_points[:,0],residual_points[:,1],color=color,s=residual_scatter_size)
                if first_scatter_handle is None:
                    first_scatter_handle = scatter_handle
                warnings.warn(
                    warning_label + ": component " + str(i+1) + " has residual non-line points; plotting those as scatter.",
                    UserWarning,
                )
            continue

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

def do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,name,approx=False,add=None,approx2=False,base_plot_name=None):
    b_vec = np.linspace(b0,b0+bv,N)
    a_vec = np.linspace(a0,a0+av,N)

    # Combine both scan directions so near-vertical / near-horizontal branches stay visible.
    x_forward,y_forward = find_group_br_a_vecb(m,thea,kap,a0,av,b_vec,eps)
    x_reverse,y_reverse = find_group_br_a_veca(m,thea,kap,a0,av,b0,bv,a_vec,eps)
    x = np.concatenate((x_forward,x_reverse))
    y = np.concatenate((y_forward,y_reverse))
    x[x==0] = np.nan
    y[y==0] = np.nan
    z_forward,w_forward = find_group_br_b_veca(m,theb,kap,rho,b0,bv,a_vec,eps)
    z_reverse,w_reverse = find_group_br_b_vecb(m,theb,kap,rho,b0,bv,a0,av,b_vec,eps)
    z = np.concatenate((z_forward,z_reverse))
    w = np.concatenate((w_forward,w_reverse))
    z[z==0] = np.nan
    w[w==0] = np.nan

    a,b = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)

    if not add is None:
        a = np.append(a,add[0])
        b = np.append(b,add[1])

    fig, ax = plt.subplots(figsize=(5,5))
    delta_a = a_vec[1] - a_vec[0]
    delta_b = b_vec[1] - b_vec[0]
    a_consistent_handle = plot_point_graph(ax,x,y,consistent_color_a,delta_a,delta_b,name + " A-consistent",residual_scatter_only=True)
    b_consistent_handle = plot_point_graph(ax,z,w,color_b,delta_a,delta_b,name + " B-consistent",residual_scatter_only=True)
    diagonal_handle, = ax.plot(
        np.linspace(max(a0,b0),min(b0+bv,a0+av),N),
        np.linspace(max(a0,b0),min(b0+bv,a0+av),N),
        color="black",
        label="b=a",
        linestyle="--",
        linewidth=consistent_linewidth,
    )
    if a_consistent_handle is not None:
        a_consistent_handle.set_label("$A$-consistent")
    if b_consistent_handle is not None:
        b_consistent_handle.set_label("$B$-consistent")

    equilibrium_handle = None
    for i in range(0,len(a)):
        
        stability_test_b_minus = [b[i],b[i]]
        stability_test_b_plus = [b[i],b[i]]
        stability_test_a_minus = [a[i],a[i] - 1e-3]
        stability_test_a_plus = [a[i],a[i] + 1e-3]

        for k in range(0,100):
            cand_br = find_group_br_b(m,theb,kap,rho,b0,bv,stability_test_a_plus[-1],eps)
            cand_br[cand_br < b0] = b0
            cand_br[cand_br > b0+bv] = b0+bv
            if len(cand_br) > 0:
                stability_test_b_plus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_b_plus[-1]))])
                stability_test_a_plus.append(stability_test_a_plus[-1])
            else:
                break

            cand_br = find_group_br_a(m,thea,kap,a0,av,stability_test_b_plus[-1],eps)
            cand_br[cand_br < a0] = a0
            cand_br[cand_br > a0+av] = a0+av
            if len(cand_br)> 0:
                stability_test_a_plus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_a_plus[-1]))])
                stability_test_b_plus.append(stability_test_b_plus[-1])
            else:
                break

        for k in range(0,100):
            cand_br = find_group_br_b(m,theb,kap,rho,b0,bv,stability_test_a_minus[-1],eps)
            cand_br[cand_br < b0] = b0
            cand_br[cand_br > b0+bv] = b0+bv
            # print(cand_br)
            if len(cand_br) > 0:
                stability_test_b_minus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_b_minus[-1]))])
                stability_test_a_minus.append(stability_test_a_minus[-1])
            else:
                break

            cand_br = find_group_br_a(m,thea,kap,a0,av,stability_test_b_minus[-1],eps)
            cand_br[cand_br < a0] = a0
            cand_br[cand_br > a0+av] = a0+av
            if len(cand_br) > 0:
                stability_test_a_minus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_a_minus[-1]))])
                stability_test_b_minus.append(stability_test_b_minus[-1])
            else:
                break
            
        plotline_a_minus = [stability_test_a_minus[0],stability_test_a_minus[-1]]
        plotline_b_minus = [stability_test_b_minus[0],stability_test_b_minus[-1]]
        plotline_a_plus = [stability_test_a_plus[0],stability_test_a_plus[-1]]
        plotline_b_plus = [stability_test_b_plus[0],stability_test_b_plus[-1]]

        # if np.abs(plotline_a_minus[-1]-a[i]) < 1e-3 + 1e-4 and np.abs(plotline_b_minus[-1]-b[i]) < 1e-3+ 1e-4 and np.abs(plotline_a_plus[-1]-a[i]) < 1e-3+ 1e-4 and np.abs(plotline_b_plus[-1]-b[i]) < 1e-3+ 1e-4:
    
            
        plot_bullseye(
            ax,
            a[i],
            b[i],
            equilibrium_marker_color,
            equilibrium_marker_outer_size,
            equilibrium_marker_inner_size,
            equilibrium_marker_zorder,
        )
        # else:
            # ax.scatter(a[i],b[i],color="red",marker="*",s=15)



        # ax.plot(plotline_a_minus,plotline_b_minus,color="red",linestyle="-")
        # ax.plot(plotline_a_plus,plotline_b_plus,color="yellow",linestyle="-")
        print(len(stability_test_a_minus))
        print(len(stability_test_a_plus))
        print("equilibrium ", a[i],b[i])
        print("stab_a_plus ",stability_test_a_plus[-1],stability_test_b_plus[-1])
        print("stab_a_minus",stability_test_a_minus[-1],stability_test_b_minus[-1])
        
        print(name)
        print("a=" + str(a[i]))
        print("b=" + str(b[i]))
    if len(a) > 0:
        equilibrium_handle = bullseye_legend_handle(
            equilibrium_marker_color,
            equilibrium_marker_outer_size,
            equilibrium_marker_inner_size,
        )
    # uncomment to plot the ratio condition:
    # quadsol = a0 /2 + np.sqrt(a0**2/ 4 + (av**2/(rho*bv**2))*(b_vec**2 - b0*b_vec))
    # plt.plot(quadsol,b_vec,color="green",label=r"$\frac{b}{\rho a} = \frac{C_A'(a)}{C_B'(b)}$")
    ax.set_xlabel("$a$")
    if approx2:
        fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A\approx' + str(round(thea,2)) + r',\theta_B=' + str(theb) + r',$',ha="center")
    else:
        fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
    if approx:
        fig.text(0.5,-0.05,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=\frac{b_0 a_v}{a_0}\approx' + str(round(bv,2)) + r'$',ha="center")
    else:
        fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
    ax.set_ylabel("$b$")
    set_boundary_ticks(ax,a0,av,b0,bv)
    legend_handles = []
    legend_labels = []
    for handle in [a_consistent_handle,b_consistent_handle,diagonal_handle]:
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(handle.get_label())
    if equilibrium_handle is not None:
        legend_handles.append(equilibrium_handle)
        legend_labels.append(equilibrium_marker_label(len(a)))
    ax.legend(handles=legend_handles,labels=legend_labels,handler_map={tuple: HandlerTuple(ndivide=1)})
    set_axis_bounds(ax, xlim=(a0,a0+av), ylim=(b0,b0+bv))
    output_name = name if base_plot_name is None else base_plot_name
    plt.savefig("1d_plots/" + output_name + ".png",format="png",dpi=600,bbox_inches="tight")
    plt.close()
    if len(a) > 0:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel("$a^i$")
        ax.set_ylabel(r"$EU_A^\kappa(a,a^i)$")
        viridis_map_new = mpl.colormaps.get_cmap('viridis')
        viridis_new = viridis_map_new(np.linspace(0,1,len(a)))
        single_equilibrium = len(a) == 1
        if add is None:
            for i in range(0,len(a)):
                plot_bullseye(
                    ax,
                    a[i],
                    utility_a(m,thea,kap,a0,av,a[i],b[i],a[i]),
                    viridis_new[i],
                    deviation_equilibrium_marker_outer_size,
                    deviation_equilibrium_marker_inner_size,
                    equilibrium_marker_zorder,
                )
                plt.plot(
                    a_vec,
                    utility_a(m,thea,kap,a0,av,a[i],b[i],a_vec),
                    color=viridis_new[i],
                    label="_nolegend_" if single_equilibrium else equilibrium_pair_label(len(a),i),
                )
        else:
            for i in range(0,len(a)):
                plot_bullseye(
                    ax,
                    a[i],
                    utility_a(m,thea,kap,a0,av,a[i],b[i],a[i]),
                    viridis_new[0],
                    deviation_equilibrium_marker_outer_size,
                    deviation_equilibrium_marker_inner_size,
                    equilibrium_marker_zorder,
                )
                if i == 0:
                    plt.plot(
                        a_vec,
                        utility_a(m,thea,kap,a0,av,a[i],b[i],a_vec),
                        color=viridis_new[i],
                        label="_nolegend_" if single_equilibrium else equilibrium_pair_label(len(a),i),
                    )
        legend_handles,legend_labels = ax.get_legend_handles_labels()
        if len(a) > 0:
            legend_color = viridis_new[0] if single_equilibrium else "black"
            legend_handles.append(
                bullseye_legend_handle(
                    legend_color,
                    deviation_equilibrium_marker_outer_size,
                    deviation_equilibrium_marker_inner_size,
                )
            )
            legend_labels.append(equilibrium_marker_label(len(a)))
        ax.legend(handles=legend_handles,labels=legend_labels,handler_map={tuple: HandlerTuple(ndivide=1)})
        set_axis_bounds(ax, xlim=(a0,a0+av))
        plt.savefig("1d_plots/" + name + "_deviations_A.png",format="png",dpi=600,bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel("$b^i$")
        ax.set_ylabel(r"$EU_B^\kappa(b,b^i)$")
        for i in range(0,len(a)):
            plot_bullseye(
                ax,
                b[i],
                utility_b(m,theb,kap,rho,b0,bv,a[i],b[i],b[i]),
                viridis_new[i],
                deviation_equilibrium_marker_outer_size,
                deviation_equilibrium_marker_inner_size,
                equilibrium_marker_zorder,
            )
            plt.plot(
                b_vec,
                utility_b(m,theb,kap,rho,b0,bv,a[i],b[i],b_vec),
                color=viridis_new[i],
                label="_nolegend_" if single_equilibrium else equilibrium_pair_label(len(a),i),
            )
        legend_handles,legend_labels = ax.get_legend_handles_labels()
        if len(a) > 0:
            legend_color = viridis_new[0] if single_equilibrium else "black"
            legend_handles.append(
                bullseye_legend_handle(
                    legend_color,
                    deviation_equilibrium_marker_outer_size,
                    deviation_equilibrium_marker_inner_size,
                )
            )
            legend_labels.append(equilibrium_marker_label(len(a)))
        ax.legend(handles=legend_handles,labels=legend_labels,handler_map={tuple: HandlerTuple(ndivide=1)})
        set_axis_bounds(ax, xlim=(b0,b0+bv))
        plt.savefig("1d_plots/" + name + "_deviations_B.png",format="png",dpi=600,bbox_inches="tight")
        plt.close()

N = 500
eps = 1e-10

m = 10.0
kap = 0.5
rho = 2.0
thea = 3.0
theb = thea
av = 1.5
bv = 0.9
a0 = 0.5
b0 = 0.3


do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_no_eq")

m =40.0
rho =  5.0
thea =  2.9
theb = thea
a0 = 0.85
b0 = 0.60
av = 0.90
kap = 0.14
bv = b0*av/a0

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_5_eq",True)

m = 4.8
rho = 2.35
thea = 2.0
theb = thea
a0 = 0.5
b0 = 0.1
av = 1.25
bv = 1.25
kap = 0.1

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_1_eq_kap_0_1")

m = 20.0
kap = 0.35
rho = 10.0
thea = 0.4
theb = 1.8
a0 = 0.85
av = 1.65
b0 = 0.2
bv = 0.7

do_plot(
    m,
    kap,
    rho,
    thea,
    theb,
    av,
    bv,
    a0,
    b0,
    eps,
    N,
    "turnout_higher_with_nonpartisan",
    base_plot_name="turnout_higher_with_nonpartisan_equilibria",
)

m = 4.8
rho = 2.35
thea = 2.0
theb = thea
a0 = 0.5
b0 = 0.1
av = 1.25
bv = 1.25
kap = 0.1


# plot accessible A
for m in [5.0,10.0,20.0]:

    a= 1.2
    b = 1.2
    i=0
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(figsize=(10,10),ncols=2,nrows=2)
    for kap in [1.0,0.8,0.6,0.4,0.2]:
        ai_range = np.linspace(a0,a0+av,N)
        x_range = (1-kap)*a + kap*ai_range
        y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
        if i > 0:
            ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i-2])
            ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i-2])
        else:
            ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color="black")
            ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color="black")
        # ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i])
        # ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i])
        i+=1

    kap_arr = np.linspace(0.00,1.0,40)
    br_arr = []
    akap_arr = []
    utility_arr = []
    for i in range(0,len(kap_arr)):
        roots = roots_utility_a(m,thea,kap_arr[i],a0,av,a,b)
        roots = roots[np.imag(roots)< eps]
        roots = np.real(roots)
        roots = roots[roots > a0]
        roots = roots[roots < a0+av]
        roots = [roots[i] for i in range(0,len(roots))]
        roots.append(a0)
        roots.append(a0+av)
        utilities = []
        for j in range(0,len(roots)):
            utilities.append(utility_a(m,thea,kap_arr[i],a0,av,a,b,roots[j]))
        
        br_arr.append(roots[np.argmax(utilities)])
        akap_arr.append(br_arr[i]*kap_arr[i] + (1-kap_arr[i])*a)
        utility_arr.append(np.max(utilities))

    akap_arr = np.array(akap_arr)
    utility_arr = np.array(utility_arr)

    akap_dir = akap_arr[1:]- akap_arr[0:-1]
    utility_dir = utility_arr[1:] - utility_arr[0:-1]

    ax1.quiver(akap_arr[0:-1],utility_arr[0:-1],akap_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
    ax1.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\kappa\uparrow$",s=100)
    ax1.set_xlabel(r"$a^\kappa$")
    ax1.set_ylabel(r"$EU_A^\kappa(a,a^i)$")
    ax1.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")
    ax1.legend()
    ax2.set_xlabel(r"$a^\kappa$")
    ax2.set_ylabel("benefit term")
    ax2.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")

    ax2.legend()

    a= 0.9
    b = 1.2
    i=0
    for kap in [1.0,0.8,0.6,0.4,0.2]:
        x_range = np.linspace(np.max([a0,(1-kap)*a + kap*a0]),np.min([a0+av,(1-kap)*(a-a0)+kap*av + a0]),N)
        ai_range = (-(1-kap)*a + x_range)/kap
        y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
        if i > 0:
            ax4.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i-2])
            ax3.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i-2])
        else:
            ax4.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color="black")
            ax3.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color="black")
        i+=1

    kap_arr = np.linspace(0.00,1.0,320)
    br_arr = []
    akap_arr = []
    utility_arr = []
    for i in range(0,len(kap_arr)):
        roots = roots_utility_a(m,thea,kap_arr[i],a0,av,a,b)
        roots = roots[np.imag(roots)< eps]
        roots = np.real(roots)
        roots = roots[roots > a0]
        roots = roots[roots < a0+av]
        roots = [roots[i] for i in range(0,len(roots))]
        roots.append(a0)
        roots.append(a0+av)
        utilities = []
        for j in range(0,len(roots)):
            utilities.append(utility_a(m,thea,kap_arr[i],a0,av,a,b,roots[j]))
        
        br_arr.append(roots[np.argmax(utilities)])
        akap_arr.append(br_arr[i]*kap_arr[i] + (1-kap_arr[i])*a)
        utility_arr.append(np.max(utilities))

    akap_arr = np.array(akap_arr)
    utility_arr = np.array(utility_arr)

    akap_dir = akap_arr[1:]- akap_arr[0:-1]
    utility_dir = utility_arr[1:] - utility_arr[0:-1]

    if np.any(akap_dir > 0.1):
        cutoff_index = np.argmax(akap_dir > 0.1) + 1
        print(cutoff_index)
        akap_arr_1 = akap_arr[cutoff_index-1::-8][::-1]
        akap_arr_2 = akap_arr[cutoff_index::8]
        utility_arr_1 = utility_arr[cutoff_index-1::-8][::-1]
        utility_arr_2 = utility_arr[cutoff_index::8]

        akap_dir_1 = akap_arr_1[1:]- akap_arr_1[0:-1]
        utility_dir_1 = utility_arr_1[1:] - utility_arr_1[0:-1]
        akap_dir_2 = akap_arr_2[1:]- akap_arr_2[0:-1]
        utility_dir_2 = utility_arr_2[1:] - utility_arr_2[0:-1]

        ax3.quiver(akap_arr_1[:-1],utility_arr_1[:-1],akap_dir_1,utility_dir_1,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
        ax3.quiver(akap_arr_2[:-1],utility_arr_2[:-1],akap_dir_2,utility_dir_2,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")

        ax3.plot([akap_arr[cutoff_index-1],akap_arr[cutoff_index]],[utility_arr[cutoff_index-1],utility_arr[cutoff_index]],linestyle="dotted",color="black")


    else:
        akap_arr = akap_arr[::4]
        utility_arr = utility_arr[::4]
        akap_dir = akap_arr[1:]- akap_arr[0:-1]
        utility_dir = utility_arr[1:] - utility_arr[0:-1]
        ax3.quiver(akap_arr[:-1],utility_arr[:-1],akap_dir,utility_dir,scale=1,angles="xy",scale_units="xy",width=0.003,units="width")
    ax3.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\kappa\uparrow$",s=100)
    ax3.set_xlabel(r"$a^\kappa$")
    ax3.set_ylabel(r"$EU_A^\kappa(a,a^i)$")
    ax3.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")
    ax3.legend()
    ax4.set_xlabel(r"$a^\kappa$")
    ax4.set_ylabel("benefit term")
    ax4.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")

    ax4.legend()

    fig.text(0.5,0.05,r'$m='+ str(m) + r',\rho =' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center") 
    set_axes_bounds([ax1,ax2,ax3,ax4])
    plt.savefig(f"1d_plots/accessible_A_m={m}.png",format="png",dpi=600,bbox_inches="tight")

a= 1.1
b = 1.2
kap = 0.5
thea = 2.0
i=0
fig, ax1 = plt.subplots(figsize=(5,5),ncols=1,nrows=1)
fig2, ax2 = plt.subplots(figsize=(5,5),ncols=1,nrows=1)
for m in [1.0,5.0,10.0,15.0]:
    ai_range = np.linspace(a0,a0+av,N)
    y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
    ax1.plot(ai_range,y,label=r"$m=" + str(m) + r"$",color=viridis[2*i])
    i+=1

m_arr = np.linspace(1.0,15.0,25)
br_arr = []
utility_arr = []
for i in range(0,len(m_arr)):
    roots = roots_utility_a(m_arr[i],thea,kap,a0,av,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > a0]
    roots = roots[roots < a0+av]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(a0)
    roots.append(a0+av)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_a(m_arr[i],thea,kap,a0,av,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]

ax1.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax1.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $m\uparrow$",s=100)

m= 4.8



i=0
for thea in [1.0,2.0,4.0,6.0]:
    ai_range = np.linspace(a0,a0+av,N)
    y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
    ax2.plot(ai_range,y,label=r"$\theta_A=" + str(thea) + r"$",color=viridis[2*i])
    i+=1

thea_arr = np.linspace(1.0,6.0,25)
br_arr = []
utility_arr = []
for i in range(0,len(thea_arr)):
    roots = roots_utility_a(m,thea_arr[i],kap,a0,av,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > a0]
    roots = roots[roots < a0+av]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(a0)
    roots.append(a0+av)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_a(m,thea_arr[i],kap,a0,av,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]
ax2.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax2.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\theta_A\uparrow$",s=100)

ax1.set_xlabel(r"$a^i$")
ax2.set_xlabel(r"$a^i$")
ax1.set_ylabel(r"$EU_A^\kappa(a,a^i)$")
ax2.set_ylabel(r"$EU_A^\kappa(a,a^i)$")
ax1.legend()
ax2.legend()

thea = 2.0

fig2.text(0.5,-0.01, r'$m=' + str(m) + r',\kappa=' + str(kap) + r',\rho=' + str(rho) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig2.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center") 
fig.text(0.5,-0.01, r'$\kappa=' + str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center")

ax1.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
ax2.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
set_axis_bounds(ax1, xlim=(a0,a0+av))
set_axis_bounds(ax2, xlim=(a0,a0+av))
fig.savefig("1d_plots/comp_stat_BR_m.png",format="png",dpi=600,bbox_inches="tight")
fig2.savefig("1d_plots/comp_stat_BR_thea.png",format="png",dpi=600,bbox_inches="tight")
plt.close()


i = 0
b0 = 0.1
m = 10.0
a = 0.7
fig, ax = plt.subplots(figsize=(5,5))

for rho in [1.0,3.0,6.0,9.0]:
    bi_range = np.linspace(b0,b0+bv,N)
    y = utility_b(m,theb,kap,rho,b0,bv,a,b,bi_range)
    ax.plot(bi_range,y,label=r"$\rho=" + str(rho) + r"$",color=viridis[2*i])
    i+=1

rho_arr = np.linspace(1.0,9.0,25)
br_arr = []
utility_arr = []

for i in range(0,len(rho_arr)):
    roots = roots_utility_b(m,theb,kap,rho_arr[i],b0,bv,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > b0]
    roots = roots[roots < b0+bv]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(b0)
    roots.append(b0+bv)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_b(m,theb,kap,rho_arr[i],b0,bv,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]

ax.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\rho\uparrow$",s=100)

ax.set_xlabel(r"$b^i$")
ax.set_ylabel(r"$EU_B^\kappa(b,b^i)$")
ax.legend()

fig.text(0.5,-0.01, r'$m=' + str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center")
ax.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
set_axis_bounds(ax, xlim=(b0,b0+bv))
plt.savefig("1d_plots/comp_stat_BR_rho.png",format="png",dpi=600,bbox_inches="tight")

a = 1.1
rho = 2.35
kap = 0.3
b0 = 0.1
m = 4.8
m_space = np.linspace(1.0,30.0,1000)

ms = []
a = []
b = []

for m in m_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        ms.append(m)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(ms,a,color="black",s=1)
ax.scatter(ms,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(m_space,len(m_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(m_space,len(m_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(m_space,len(m_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(m_space,len(m_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(ax, xlim=(m_space[0],m_space[-1]), ylim=turnout_bounds(a0,av,b0,bv))
plt.savefig("1d_plots/comp_statics_m_1.png",format="png",dpi=600,bbox_inches="tight")

thea_eq_turnouts = 2.9
theb_eq_turnouts = 2.9
kap_eq_turnouts = 0.5
rho_eq_turnouts = 5.0
a0_eq_turnouts = 0.85
b0_eq_turnouts = 0.6
av_eq_turnouts = 0.9
bv_eq_turnouts = b0_eq_turnouts * av_eq_turnouts / a0_eq_turnouts
m_space_eq_turnouts = np.linspace(0.5,100.0,400)

ms_eq_turnouts = []
a_eq_turnouts = []
b_eq_turnouts = []

for m_eq_turnouts in m_space_eq_turnouts:
    a_new,b_new = find_all_equilibria(
        m_eq_turnouts,
        thea_eq_turnouts,
        theb_eq_turnouts,
        kap_eq_turnouts,
        rho_eq_turnouts,
        a0_eq_turnouts,
        b0_eq_turnouts,
        av_eq_turnouts,
        bv_eq_turnouts,
        eps,
    )
    for j in range(0,len(a_new)):
        ms_eq_turnouts.append(m_eq_turnouts)
        a_eq_turnouts.append(a_new[j])
        b_eq_turnouts.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(ms_eq_turnouts,a_eq_turnouts,color=color_a,s=1)
ax.scatter(ms_eq_turnouts,b_eq_turnouts,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color=color_a,label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(m_space_eq_turnouts,len(m_space_eq_turnouts)*[a0_eq_turnouts],linestyle="--",color=color_a,label=r"$a_0$")
ax.plot(m_space_eq_turnouts,len(m_space_eq_turnouts)*[b0_eq_turnouts],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(m_space_eq_turnouts,len(m_space_eq_turnouts)*[a0_eq_turnouts+av_eq_turnouts],linestyle="dotted",color=color_a,label=r"$a_0+a_v$")
ax.plot(m_space_eq_turnouts,len(m_space_eq_turnouts)*[b0_eq_turnouts+bv_eq_turnouts],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap_eq_turnouts) + r',\rho=' + str(rho_eq_turnouts) + r',\theta_A=' + str(thea_eq_turnouts) + r',\theta_B=' + str(theb_eq_turnouts) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0_eq_turnouts) +r',a_v=' + str(av_eq_turnouts) + r',b_0=' + str(b0_eq_turnouts) + r',b_v=\frac{b_0 a_v}{a_0}\approx' + str(round(bv_eq_turnouts,3)) + r'$',ha="center")

set_axis_bounds(
    ax,
    xlim=(m_space_eq_turnouts[0],m_space_eq_turnouts[-1]),
    ylim=turnout_bounds(a0_eq_turnouts,av_eq_turnouts,b0_eq_turnouts,bv_eq_turnouts),
)
plt.savefig("1d_plots/comp_statics_m_2.png",format="png",dpi=600,bbox_inches="tight")

thea_eq_turnouts_nonpartisan = 0.4
theb_eq_turnouts_nonpartisan = 1.8
kap_eq_turnouts_nonpartisan = 0.35
rho_eq_turnouts_nonpartisan = 10.0
a0_eq_turnouts_nonpartisan = 0.85
b0_eq_turnouts_nonpartisan = 0.2
av_eq_turnouts_nonpartisan = 1.65
bv_eq_turnouts_nonpartisan = 0.7
m_space_eq_turnouts_nonpartisan = np.linspace(0.5,30.0,1000)

ms_eq_turnouts_nonpartisan = []
a_eq_turnouts_nonpartisan = []
b_eq_turnouts_nonpartisan = []

for m_eq_turnouts_nonpartisan in m_space_eq_turnouts_nonpartisan:
    a_new,b_new = find_all_equilibria(
        m_eq_turnouts_nonpartisan,
        thea_eq_turnouts_nonpartisan,
        theb_eq_turnouts_nonpartisan,
        kap_eq_turnouts_nonpartisan,
        rho_eq_turnouts_nonpartisan,
        a0_eq_turnouts_nonpartisan,
        b0_eq_turnouts_nonpartisan,
        av_eq_turnouts_nonpartisan,
        bv_eq_turnouts_nonpartisan,
        eps,
    )
    for j in range(0,len(a_new)):
        ms_eq_turnouts_nonpartisan.append(m_eq_turnouts_nonpartisan)
        a_eq_turnouts_nonpartisan.append(a_new[j])
        b_eq_turnouts_nonpartisan.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(ms_eq_turnouts_nonpartisan,a_eq_turnouts_nonpartisan,color=color_a,s=1)
ax.scatter(ms_eq_turnouts_nonpartisan,b_eq_turnouts_nonpartisan,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color=color_a,label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(m_space_eq_turnouts_nonpartisan,len(m_space_eq_turnouts_nonpartisan)*[a0_eq_turnouts_nonpartisan],linestyle="--",color=color_a,label=r"$a_0$")
ax.plot(m_space_eq_turnouts_nonpartisan,len(m_space_eq_turnouts_nonpartisan)*[b0_eq_turnouts_nonpartisan],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(m_space_eq_turnouts_nonpartisan,len(m_space_eq_turnouts_nonpartisan)*[a0_eq_turnouts_nonpartisan+av_eq_turnouts_nonpartisan],linestyle="dotted",color=color_a,label=r"$a_0+a_v$")
ax.plot(m_space_eq_turnouts_nonpartisan,len(m_space_eq_turnouts_nonpartisan)*[b0_eq_turnouts_nonpartisan+bv_eq_turnouts_nonpartisan],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap_eq_turnouts_nonpartisan) + r',\rho=' + str(rho_eq_turnouts_nonpartisan) + r',\theta_A=' + str(thea_eq_turnouts_nonpartisan) + r',\theta_B=' + str(theb_eq_turnouts_nonpartisan) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0_eq_turnouts_nonpartisan) +r',a_v=' + str(av_eq_turnouts_nonpartisan) + r',b_0=' + str(b0_eq_turnouts_nonpartisan) + r',b_v=' + str(bv_eq_turnouts_nonpartisan) + r'$',ha="center")

set_axis_bounds(
    ax,
    xlim=(m_space_eq_turnouts_nonpartisan[0],m_space_eq_turnouts_nonpartisan[-1]),
    ylim=turnout_bounds(a0_eq_turnouts_nonpartisan,av_eq_turnouts_nonpartisan,b0_eq_turnouts_nonpartisan,bv_eq_turnouts_nonpartisan),
)
plt.savefig("1d_plots/comp_statics_m_3.png",format="png",dpi=600,bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
bs = [0.6,0.9,1.2,1.5]
i=0
for b in bs:
    ms_forward = []
    a_forward = []

    for m in m_space:
        a_new = find_group_br_a(m,thea,kap,a0,av,b,eps)
        for j in range(0,len(a_new)):
            ms_forward.append(m)
            a_forward.append(a_new[j])

    a_grid = np.linspace(a0,a0+av,len(m_space))
    ms_reverse,a_reverse = find_group_br_a_vecm_from_a(m_space,thea,kap,a0,av,b,a_grid,eps)

    ms = np.concatenate((np.array(ms_forward),ms_reverse))
    a = np.concatenate((np.array(a_forward),a_reverse))

    delta_m,delta_a = estimate_curve_steps(ms,a,m_space[1] - m_space[0],a_grid[1] - a_grid[0])
    plot_point_graph(ax,ms,a,viridis[2*i],delta_m,delta_a,"comp_statics_m_a_consistent b=" + str(b))
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$b=" + str(b) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$a$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av)  + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(ax, xlim=(m_space[0],m_space[-1]), ylim=(a0,a0+av))
plt.savefig("1d_plots/comp_statics_m_a_consistent.png",format="png",dpi=600,bbox_inches="tight")

m = 5.0
kap_space = np.linspace(0.1,0.9,1000)
kaps = []
a = []
b = []

for kap in kap_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        kaps.append(kap)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\kappa$")
ax.scatter(kaps,a,color="black",s=1)
ax.scatter(kaps,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(kap_space,len(kap_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(kap_space,len(kap_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(kap_space,len(kap_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(kap_space,len(kap_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(ax, xlim=(kap_space[0],kap_space[-1]), ylim=turnout_bounds(a0,av,b0,bv))
plt.savefig("1d_plots/comp_statics_kap.png",format="png",dpi=600,bbox_inches="tight")


m= 5.0
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\kappa$")


bs = [0.6,0.9,1.2,1.5]
i=0
for b in bs:
    kaps = []
    a = []

    for kap in kap_space:
        a_new = find_group_br_a(m,thea,kap,a0,av,b,eps)
        for j in range(0,len(a_new)):
            kaps.append(kap)
            a.append(a_new[j])

    ax.scatter(kaps,a,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$b=" + str(b) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$a$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',b=' + str(b) + r'$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r'$',ha="center")

set_axis_bounds(ax, xlim=(kap_space[0],kap_space[-1]), ylim=(a0,a0+av))
plt.savefig("1d_plots/comp_statics_kap_a_consistent.png",format="png",dpi=600,bbox_inches="tight")


kap = 0.3
rho_space = np.linspace(1.0,5.0,1000)
rhos = []
a = []
b = []

for rho in rho_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        rhos.append(rho)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
ax.scatter(rhos,a,color="black",s=1)
ax.scatter(rhos,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(rho_space,len(rho_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(rho_space,len(rho_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(rho_space,len(rho_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(rho_space,len(rho_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center") 
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center") 

set_axis_bounds(ax, xlim=(rho_space[0],rho_space[-1]), ylim=turnout_bounds(a0,av,b0,bv))
plt.savefig("1d_plots/comp_statics_rho.png",format="png",dpi=600,bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")

a_s = [0.6,0.9,1.2,1.5]
i=0
for a in a_s:
    rhos = []
    b = []

    for rho in rho_space:
        b_new = find_group_br_b(m,theb,kap,rho,b0,bv,a,eps)
        for j in range(0,len(b_new)):
            rhos.append(rho)
            b.append(b_new[j])

    ax.scatter(rhos,b,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$a=" + str(a) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$b$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',a_0=' + str(a0) +r',a_v=' + str(av) + r'$',ha="center")

fig.text(0.5,-0.04,r'$b_0=' + str(b0) +r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(ax, xlim=(rho_space[0],rho_space[-1]), ylim=(b0,b0+bv))
plt.savefig("1d_plots/comp_statics_rho_b_consistent.png",format="png",dpi=600,bbox_inches="tight")
kap = 0.3
do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_1_eq_kap_0_3")

m = 3.0
rho = 3.6
kap = 0.5
thea = 1.1
theb = 2.0
a0 = 0.5
b0 = 0.4
av = 1.4
bv = 1.1

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"minority_high_stake_majority_cost_advantage")

rho = 1
thea = 2.0
b0 = 0.7
av = 1.25
bv = 1.0
do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"minority_larger_base")

rho = 1.5
kap = 1
m = 5.7
theb = 0.1
a0 = 0.35
b0 = 0.05
av = 1.2
bv = 1.75
def difference(thea):
    a1 = np.real(roots_aux_a(m,thea,kap,a0,av,b0+bv)[2])
    return utility_a(m,thea,kap,a0,av,a1,b0+bv,a1) - utility_a(m,thea,kap,a0,av,a0+av,b0+bv,a0+av)

thea_root = sco.brentq(difference,1.0,1.5)
print(thea_root)

do_plot(m,kap,rho,thea_root,theb,av,bv,a0,b0,eps,N,"multi_kappa_1",add=(a0+av,b0+bv),approx2=True)

# example for limit
kap = 0.2
thea = 3.0
theb = 3.0
a0 = 0.2
b0 = 0.5
av = 0.6
bv = 0.15
rho = 1.2

m_vec_ln = np.linspace(-0.5,6,20000)
a = []
b = []
m = []

for i in range(0,len(m_vec_ln)):
    m_new = np.exp(np.log(10)*m_vec_ln[i])
    a_new,b_new = find_all_equilibria(m_new,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        a.append(a_new[j])
        b.append(b_new[j])
        m.append(m_new)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(m,a,color="black",s=1)
ax.scatter(m,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")
ax.set_ylabel(r"$a,b$")
ax.set_xscale("log")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(
    ax,
    xlim=(np.exp(np.log(10)*m_vec_ln[0]),np.exp(np.log(10)*m_vec_ln[-1])),
    ylim=turnout_bounds(a0,av,b0,bv),
)
plt.savefig("1d_plots/limit_ex_post.png",format="png",dpi=600,bbox_inches="tight")

# example for high turnout limit
kap = 0.2
thea = 1.5
theb = 1.5
a0 = 0.35
b0 = 0.25
av = 1.25
bv = 1.35
rho = 1.2

m_vec_ln = np.linspace(-0.5,6,1000)
a = []
b = []
m = []

for i in range(0,len(m_vec_ln)):
    m_new = np.exp(np.log(10)*m_vec_ln[i])
    a_new,b_new = find_all_equilibria(m_new,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        a.append(a_new[j])
        b.append(b_new[j])
        m.append(m_new)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(m,a,color="black",s=4)
ax.scatter(m,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",linewidth=2.0,label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,linewidth=1.0,label="$b$")
ax.set_xlabel(r"$m$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v = b_0 + b_v$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.legend(loc="right")
ax.set_ylabel(r"$a,b$")
ax.set_xscale("log")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

set_axis_bounds(
    ax,
    xlim=(np.exp(np.log(10)*m_vec_ln[0]),np.exp(np.log(10)*m_vec_ln[-1])),
    ylim=turnout_bounds(a0,av,b0,bv),
)
plt.savefig("1d_plots/limit_ex_post_high_turnout.png",format="png",dpi=600,bbox_inches="tight")

# example for partial underdog compensation

kap = 0.15
m = 42.0
thea = 2.9
theb = thea
a0 = 0.85
b0 = 0.60
av = 0.90
bv = b0*av / a0

rho_vec = np.linspace(1,5.5,400)

a = [[],[],[]]
b = [[],[],[]]
rho_list = [[],[],[]]

for i in range(0,len(rho_vec)):
    rho = rho_vec[i]
    a_new,b_new = find_equilibria_interior(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    if len(a_new) == 1:
        a[0].append(a_new[0])
        b[0].append(b_new[0])
        rho_list[0].append(rho)
    elif len(a_new) == 2:
        a[0].append(a_new[1])
        b[0].append(b_new[1])
        rho_list[0].append(rho)
        a[1].append(a_new[0])
        b[1].append(b_new[0])
        rho_list[1].append(rho)
    else:
        a[0].append(a_new[2])
        b[0].append(b_new[2])
        rho_list[0].append(rho)
        a[2].append(a_new[1])
        b[2].append(b_new[1])
        rho_list[2].append(rho)
        a[1].append(a_new[0])
        b[1].append(b_new[0])
        rho_list[1].append(rho)
    

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
for i in range(0,3):
    ax.plot(rho_list[i],np.array(a[i])/av,color=viridis[2*i],linestyle="--")
    ax.plot(rho_list[i],np.array(b[i])/bv,color=viridis[2*i])
    ax.plot(rho_list[i],np.array(rho_list[i])*np.array(a[i])/av,color=viridis[2*i],linestyle="dotted")

ax.plot([np.nan],[np.nan],color="black",linestyle="--",label=r"$\frac{a}{a_v}$")
ax.plot([np.nan],[np.nan],color="black",label=r"$\frac{b}{b_v}$")
ax.plot([np.nan],[np.nan],color="black",linestyle="dotted",label=r"$\rho \frac{a}{a_v}$")    
ax.set_ylabel(r"$a,b$")
ax.legend()
y_values = []
for i in range(0,3):
    if len(a[i]) > 0:
        y_values.append(np.array(a[i])/av)
        y_values.append(np.array(b[i])/bv)
        y_values.append(np.array(rho_list[i])*np.array(a[i])/av)
if len(y_values) > 0:
    set_axis_bounds(
        ax,
        xlim=(rho_vec[0],rho_vec[-1]),
        ylim=(min(np.min(arr) for arr in y_values),max(np.max(arr) for arr in y_values)),
    )
else:
    set_axis_bounds(ax, xlim=(rho_vec[0],rho_vec[-1]))
plt.savefig("1d_plots/partial_compensation.png",format="png",dpi=600,bbox_inches="tight")
plt.close()
