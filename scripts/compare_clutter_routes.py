"""Compare scene admission routes with the six recorded policy trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from cat_ppo.furniture.scenes import _project_route, _root_obstacles, _horizontal_distance
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = json.loads(args.selection.read_text())["examples"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 15), facecolor="#f5f8fa")
    report = []
    for ax, number in zip(axes.flat, (4001, 4002, 5001, 5002)):
        chosen = [e for e in examples if e["scene_seed"] == number]
        scene = json.loads((Path(chosen[0]["episode_dir"])/"scene.json").read_text())
        for part in sorted(scene["boxes"], key=lambda p:p["center"][2]):
            center, half = np.asarray(part["center"][:2]), np.asarray(part["half_size"][:2])
            c, s = np.cos(part["yaw"]), np.sin(part["yaw"])
            rotation = np.asarray([[c,-s],[s,c]])
            corners = np.asarray([[-1,-1],[1,-1],[1,1],[-1,1]])*half
            corners = np.einsum("ij,kj->ik", corners, rotation)+center
            color = ("#d8c1a0" if part["category"].startswith("table") else
                     "#a9bec3" if part["category"].startswith("chair") else
                     "#8596a5" if part["category"] == "wall" else "#c9c4be")
            ax.add_patch(Polygon(corners, facecolor=color, edgecolor="#86919a", linewidth=.4, zorder=1))
        route = np.asarray(scene["route"])
        ax.plot(route[:,0], route[:,1], color="white", linewidth=5.5, zorder=3)
        ax.plot(route[:,0], route[:,1], "--", color="#2466cd", linewidth=2.8, zorder=4)
        ax.scatter(route[:,0],route[:,1],s=14,color="#2466cd",zorder=4)
        records = []
        subtitles = []
        for example in chosen:
            with np.load(Path(example["episode_dir"])/"trajectory.npz") as trace:
                xy = trace["qpos"][:,:2]
            success = example["outcome"]["goal_reached"]
            color = "#00865c" if success else "#d54539"
            ax.plot(xy[:,0],xy[:,1],color=color,linewidth=2.1,zorder=5)
            ax.scatter(*xy[-1],marker="o" if success else "X",s=80,c=color,edgecolors="white",linewidths=.9,zorder=6)
            distances = np.asarray([_project_route(p, scene["route"])[0] for p in xy])
            obstacles = _root_obstacles(scene["boxes"])
            minimum = min(_horizontal_distance(p,b) for p in xy for b in obstacles)
            records.append(dict(seed=example["seed"], goal_reached=success,
                                actual_root_xy_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                                maximum_distance_from_admission_route_m=float(distances.max()),
                                minimum_xy_clearance_to_root_band_obstacles_m=float(minimum),
                                interpretation="Geometric projection diagnostic, not exact articulated-body collision test"))
            subtitles.append(f"{'success' if success else 'failure'} seed {example['seed']}")
        for label, point in (("A",scene["start"][:2]),("B",scene["goal"])):
            ax.scatter(*point,s=190,color="#17394f",edgecolor="white",linewidth=1.2,zorder=7)
            ax.text(*point,label,ha="center",va="center",color="white",weight="bold",fontsize=10,zorder=8)
        x,y=scene["room_dimensions"][:2]
        ax.set(xlim=(-.2,x+.2),ylim=(-.2,y+.2),aspect="equal",xlabel="x (m)",ylabel="y (m)")
        ax.set_facecolor("#ffffff")
        ax.set_title(f"{'Furniture' if number<5000 else 'Mixed clutter'} {number}  |  A* route {scene['route_length_m']:.2f} m\n"+" · ".join(subtitles),loc="left",fontsize=13,pad=12)
        ax.grid(alpha=.12)
        report.append(dict(scene=number,route=scene["route"],route_length_m=scene["route_length_m"],
                           root_route_guaranteed_margin_m=scene["feasibility"]["root_margin_lower_bound_m"],
                           route_used_for_guidance=scene["cat_field_generation"]["route_used_for_guidance"],recordings=records))
    fig.suptitle("Checked scene routes vs. recorded robot motion",fontsize=23,weight="bold",x=.07,ha="left",y=.988,color="#17394f")
    handles = [Line2D([0],[0],color="#2466cd",linestyle="--",lw=3,label="A* feasibility route — not sent to policy"),
               Line2D([0],[0],color="#00865c",lw=3,label="Recorded success"),
               Line2D([0],[0],color="#d54539",lw=3,label="Recorded failure")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.963),ncol=3,frameon=False,fontsize=11)
    fig.text(.07,.024,"A = nominal start    B = goal. Shaded shapes show projected obstacle geometry.\nBlue routes check a 0.23 m-radius root cylinder over z = 0.35–1.05 m; they do not certify limb clearance.",fontsize=11,color="#435c6b")
    fig.subplots_adjust(top=.88,bottom=.085,left=.07,right=.98,wspace=.19,hspace=.24)
    fig.savefig(args.output/"routes-vs-recorded-motion.png",dpi=160)
    fig.savefig(args.output/"routes-vs-recorded-motion.pdf")
    (args.output/"route-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
