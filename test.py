import numpy as np
import os
import pickle
import cv2
import torch
from torch_geometric.data import Data
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ezdxf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import box, Point


def get_primitive_center(u, v):
    return ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)


def extract_primitive_centers(G):
    centers = []
    edge_to_center = {}
    for u, v in G.edges():
        center = get_primitive_center(u, v)
        centers.append(center)
        edge_to_center[(u, v)] = center
    return np.array(centers), edge_to_center


def graph_to_pyg_data(G):
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    x = []
    for node in G.nodes(data=True):
        features = []
        features.extend(list(node[0]))  # x, y
        features.append(1 if node[1].get('type') == 'circle' else 0)
        features.append(node[1].get('radius', 0))
        features.append(node[1].get('start_angle', 0))
        features.append(node[1].get('end_angle', 0))
        x.append(features)

    edge_index = []
    for u, v in G.edges():
        if u in node_mapping and v in node_mapping:
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )


def save_graph_components(components, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for comp in components:
        filename = f"{comp['id']}.pkl"
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(comp, f)


def visualize_components_with_primitives(components, figsize=(12, 8), save_path: str = None):
    num = len(components)
    cols = 2
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, comp in enumerate(components):
        G = comp["nx_graph"]
        ax = axes[idx]

        # Отрисовка всех рёбер кроме служебных
        for u, v, data in G.edges(data=True):
            if data.get("type") in ("line", "polyline", "spline", "circle-edge", "arc"):
                ax.plot([u[0], v[0]], [u[1], v[1]], color='gray', linewidth=1)

        # Отрисовка окружностей и дуг по центрам
        for node, data in G.nodes(data=True):
            if data.get('type') == 'circle':
                r = data.get('radius', 0)
                circle = patches.Circle(node, r, fill=False, edgecolor='blue', linewidth=1.5)
                ax.add_patch(circle)
            elif data.get('type') == 'arc':
                r = data.get('radius', 0)
                start = np.rad2deg(data.get('start_angle', 0))
                end = np.rad2deg(data.get('end_angle', 0))
                arc = patches.Arc(node, 2 * r, 2 * r, angle=0, theta1=start, theta2=end,
                                  edgecolor='green', linewidth=1.5)
                ax.add_patch(arc)

        ax.set_title(f"Cluster {comp['cluster_id']}")
        ax.set_aspect('equal')
        ax.axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Сохранено изображение в: {save_path}")
    plt.show()



def parse_dxf_and_cluster(file_path: str,
                          epsilon: float = 1e-2,
                          dbscan_eps: float = 20,
                          min_samples: int = 1,
                          n_clusters: Optional[int] = None,
                          clustering_method: str = 'bbox',
                          bbox_margin: float = 10,
                          bbox_eps: float = 50):


    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    G = nx.Graph()

    def add_point(p):
        coords = list(p)
        pt = tuple(round(c, 6) for c in coords[:2])
        G.add_node(pt, type='point')
        return pt

    def connect_close_points():
        nodes = list(G.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                if G.has_edge(n1, n2): continue
                if G.nodes[n1].get('type') != 'point' or G.nodes[n2].get('type') != 'point': continue
                dist = np.linalg.norm(np.array(n1) - np.array(n2))
                if dist < epsilon:
                    G.add_edge(n1, n2, type='near')

    for e in msp:
        t = e.dxftype()

        if t == 'LINE':
            p1 = add_point(e.dxf.start)
            p2 = add_point(e.dxf.end)
            G.add_edge(p1, p2, type='line')

        elif t == 'CIRCLE':
            center = add_point(e.dxf.center)
            radius = e.dxf.radius
            G.nodes[center]['type'] = 'circle'
            G.nodes[center]['radius'] = radius

            num_points = 16
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_pts = [
                add_point((center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)))
                for a in angles
            ]
            for i in range(len(circle_pts)):
                G.add_edge(circle_pts[i], circle_pts[(i + 1) % len(circle_pts)], type='circle-edge')

        elif t == 'ARC':
            center = add_point(e.dxf.center)
            radius = e.dxf.radius
            start_angle = np.deg2rad(e.dxf.start_angle)
            end_angle = np.deg2rad(e.dxf.end_angle)
            if end_angle < start_angle:
                end_angle += 2 * np.pi
            G.nodes[center]['type'] = 'arc'
            G.nodes[center]['radius'] = radius
            G.nodes[center]['start_angle'] = start_angle
            G.nodes[center]['end_angle'] = end_angle

            num_arc_points = 10
            angles = np.linspace(start_angle, end_angle, num_arc_points)
            arc_pts = [
                add_point((center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)))
                for a in angles
            ]
            for i in range(len(arc_pts) - 1):
                G.add_edge(arc_pts[i], arc_pts[i + 1], type='arc')

        elif t in ['LWPOLYLINE', 'POLYLINE']:
            points = [add_point(p) for p in e.get_points('xy')]
            for i in range(len(points) - 1):
                G.add_edge(points[i], points[i + 1], type='polyline')
            if e.closed:
                G.add_edge(points[-1], points[0], type='polyline')

        elif t == 'SPLINE':
            try:
                spline_points = e.construction_tool().approximate(20)
                points = [add_point(p) for p in spline_points]
                for i in range(len(points) - 1):
                    G.add_edge(points[i], points[i + 1], type='spline')
            except Exception as ex:
                print(f"Ошибка при аппроксимации сплайна: {ex}")

        elif t == 'POINT':
            pt = add_point(e.dxf.location)
            G.nodes[pt]['type'] = 'single-point'

    connect_close_points()

    cluster_graphs = {}

    if clustering_method == 'bbox':
        node_coords = np.array([list(n) for n in G.nodes()])

        if n_clusters is not None:
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42).fit(node_coords)
            labels = clustering_model.labels_
        else:
            clustering_model = DBSCAN(eps=bbox_eps, min_samples=min_samples).fit(node_coords)
            labels = clustering_model.labels_

        node_to_label = dict(zip(G.nodes(), labels))

        # Построение bbox для каждого кластера
        bbox_by_cluster = {}
        for label in set(labels):
            if label == -1:
                continue  # шум
            coords = node_coords[labels == label]
            minx, miny = coords.min(axis=0)
            maxx, maxy = coords.max(axis=0)
            bbox_by_cluster[label] = box(minx - bbox_margin, miny - bbox_margin,
                                         maxx + bbox_margin, maxy + bbox_margin)

        cluster_graphs = defaultdict(nx.Graph)
        for u, v in G.edges():
            for label, bbox in bbox_by_cluster.items():
                if bbox.contains(Point(u)) and bbox.contains(Point(v)):
                    cluster_graphs[label].add_node(u, **G.nodes[u])
                    cluster_graphs[label].add_node(v, **G.nodes[v])
                    cluster_graphs[label].add_edge(u, v, **G.edges[u, v])
                    break


    elif clustering_method == 'connected':
        raw_components = []

        bboxes = []

        for component in nx.connected_components(G):
            subG = G.subgraph(component).copy()

            coords = np.array(list(subG.nodes))

            minx, miny = coords.min(axis=0)

            maxx, maxy = coords.max(axis=0)

            bbox = box(minx - bbox_margin, miny - bbox_margin, maxx + bbox_margin, maxy + bbox_margin)

            raw_components.append((subG, bbox))

            bboxes.append(bbox)

        # Шаг 2: группируем bbox'ы, которые пересекаются или рядом

        merged_groups = []

        used = set()

        for i in range(len(raw_components)):

            if i in used:
                continue

            group = [i]

            used.add(i)

            for j in range(i + 1, len(raw_components)):

                if j in used:
                    continue

                if raw_components[i][1].buffer(bbox_margin).intersects(raw_components[j][1]):
                    group.append(j)

                    used.add(j)

            merged_groups.append(group)

        # Шаг 3: собираем объединённые графы

        cluster_graphs = {}

        for cluster_id, group in enumerate(merged_groups):

            combined = nx.Graph()

            for idx in group:
                combined = nx.compose(combined, raw_components[idx][0])

            cluster_graphs[cluster_id] = combined

    elif clustering_method == 'kmeans' and n_clusters is not None:
        centers = [((u[0] + v[0]) / 2, (u[1] + v[1]) / 2) for u, v in G.edges()]
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42).fit(centers)
        labels = clustering_model.labels_
        for (u, v), label in zip(G.edges(), labels):
            if label not in cluster_graphs:
                cluster_graphs[label] = nx.Graph()
            cluster_graphs[label].add_node(u, **G.nodes[u])
            cluster_graphs[label].add_node(v, **G.nodes[v])
            cluster_graphs[label].add_edge(u, v, **G.edges[u, v])

    else:  # fallback to DBSCAN
        centers = [((u[0] + v[0]) / 2, (u[1] + v[1]) / 2) for u, v in G.edges()]
        clustering = DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_
        for (u, v), label in zip(G.edges(), labels):
            if label not in cluster_graphs:
                cluster_graphs[label] = nx.Graph()
            cluster_graphs[label].add_node(u, **G.nodes[u])
            cluster_graphs[label].add_node(v, **G.nodes[v])
            cluster_graphs[label].add_edge(u, v, **G.edges[u, v])

    def graph_to_pyg_data(G):
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        x = []
        for node in G.nodes(data=True):
            features = []
            features.extend(list(node[0]))  # x, y
            features.append(1 if node[1].get('type') == 'circle' else 0)
            features.append(node[1].get('radius', 0))
            features.append(node[1].get('start_angle', 0))
            features.append(node[1].get('end_angle', 0))
            x.append(features)

        edge_index = []
        for u, v in G.edges():
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_index.append([node_mapping[v], node_mapping[u]])

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

    results = []
    for label, subG in cluster_graphs.items():
        data = graph_to_pyg_data(subG)
        coords = np.array(list(subG.nodes))
        bbox = (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())
        results.append({
            "graph": data,
            "cluster_id": label,
            "nx_graph": subG,
            "bbox": bbox,
            "id": f"{os.path.basename(file_path).replace('.dxf', '')}_cluster_{label}"
        })

    return results



components = parse_dxf_and_cluster(
    "/Users/vrpivnev2h/PycharmProjects/diplom/data/dxf/example3.dxf",
    clustering_method='cv'  # Новый метод через OpenCV контуры
)

visualize_components_with_primitives(
    components,
    save_path="/Users/vrpivnev2h/PycharmProjects/diplom/data/png/example3_cv.png"
)

save_graph_components(
    components,
    output_dir="/Users/vrpivnev2h/PycharmProjects/diplom/output_graphs_cv/"
)

