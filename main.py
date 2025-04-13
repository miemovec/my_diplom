import ezdxf
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering


class DXFGraphProcessor:
    def __init__(self, dxf_path=None):
        self.dxf_path = dxf_path
        self.filename = dxf_path.split('/')[-1] if dxf_path else None
        self.graphs = {}

        if dxf_path:
            self.doc = ezdxf.readfile(dxf_path)
            self.modelspace = self.doc.modelspace()
            self.entities = list(self.modelspace)
        else:
            self.doc = None
            self.modelspace = None
            self.entities = []

    def arc_points(self, center, radius, start_angle, end_angle, n=4):
        angles = np.linspace(math.radians(start_angle), math.radians(end_angle), n)
        return [np.array([
            center[0] + radius * np.cos(a),
            center[1] + radius * np.sin(a)
        ]) for a in angles]

    def circle_points(self, center, radius, n=8):
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return [np.array([
            center[0] + radius * np.cos(a),
            center[1] + radius * np.sin(a)
        ]) for a in angles]

    def extract_points(self, entity):
        try:
            if entity.dxftype() == 'LINE':
                return [
                    np.array([entity.dxf.start.x, entity.dxf.start.y]),
                    np.array([entity.dxf.end.x, entity.dxf.end.y])
                ]
            elif entity.dxftype() == 'CIRCLE':
                center = np.array([entity.dxf.center.x, entity.dxf.center.y])
                return self.circle_points(center, entity.dxf.radius, n=8)
            elif entity.dxftype() == 'ARC':
                center = np.array([entity.dxf.center.x, entity.dxf.center.y])
                return self.arc_points(center, entity.dxf.radius, entity.dxf.start_angle, entity.dxf.end_angle, n=6)
            elif entity.dxftype() == 'ELLIPSE':
                return [np.array([entity.dxf.center.x, entity.dxf.center.y])]
            elif entity.dxftype() in {'TEXT', 'MTEXT'}:
                return [np.array([entity.dxf.insert.x, entity.dxf.insert.y])]
            elif entity.dxftype() == 'LWPOLYLINE':
                return [np.array(p[:2]) for p in entity.get_points()]
            elif entity.dxftype() == 'POLYLINE':
                return [np.array([v.dxf.location.x, v.dxf.location.y]) for v in entity.vertices]
        except Exception as e:
            print(f"[DEBUG] Ошибка в extract_points для {entity.dxftype()}: {e}")
        return []

    def extract_metadata(self, entity):
        if entity.dxftype() == 'CIRCLE':
            return {
                'radius': entity.dxf.radius,
                'center': (entity.dxf.center.x, entity.dxf.center.y)
            }
        elif entity.dxftype() == 'ARC':
            return {
                'radius': entity.dxf.radius,
                'center': (entity.dxf.center.x, entity.dxf.center.y),
                'start_angle': entity.dxf.start_angle,
                'end_angle': entity.dxf.end_angle
            }
        return {}

    def compute_auto_threshold(self, entity_points):
        distances = []
        for i in range(len(entity_points)):
            pts_i = entity_points[i][2]
            for j in range(i + 1, len(entity_points)):
                pts_j = entity_points[j][2]
                dists = cdist(pts_i, pts_j)
                min_dist = np.min(dists)
                distances.append(min_dist)

        if not distances:
            raise ValueError("Не удалось вычислить авто-threshold: недостаточно примитивов с координатами.")

        distances = np.sort(distances)
        diffs = np.diff(distances)

        if len(diffs) == 0:
            return np.median(distances) + 1

        jump_index = np.argmax(diffs)
        return distances[jump_index] + 1

    def points_intersect(self, pts1, pts2, epsilon=1e-3):
        for p1 in pts1:
            for p2 in pts2:
                if np.linalg.norm(p1 - p2) < epsilon:
                    return True
        return False

    def build_graphs(self, n_views=None, threshold=None, epsilon=1e-3):
        points_per_entity = []
        for i, entity in enumerate(self.entities):
            points = self.extract_points(entity)
            metadata = self.extract_metadata(entity)
            if points:
                center = np.mean(points, axis=0)
                points_per_entity.append((i, entity, points, center, metadata))

        if len(points_per_entity) == 0:
            print("[INFO] Нет подходящих примитивов для анализа.")
            return

        if n_views is not None:
            if n_views <= 1:
                print("[INFO] Указано только одно представление — деление не требуется.")
                return

            centers = np.array([item[3] for item in points_per_entity])
            clustering = AgglomerativeClustering(n_clusters=n_views).fit(centers)
            labels = clustering.labels_

            for k in range(n_views):
                G = nx.Graph()
                for (i, entity, points, _, metadata), label in zip(points_per_entity, labels):
                    if label == k:
                        G.add_node(i, entity=entity, metadata=metadata)
                self.graphs[f'view_{k + 1}'] = G
        else:
            G = nx.Graph()
            entity_points = []

            for i, entity, points, _, metadata in points_per_entity:
                entity_points.append((i, entity, np.array(points)))
                G.add_node(i, entity=entity, metadata=metadata)

            if threshold is None:
                threshold = self.compute_auto_threshold(entity_points)

            for i in range(len(entity_points)):
                idx_i, _, pts_i = entity_points[i]
                for j in range(i + 1, len(entity_points)):
                    idx_j, _, pts_j = entity_points[j]
                    if self.points_intersect(pts_i, pts_j, epsilon=epsilon):
                        G.add_edge(idx_i, idx_j)
                    elif np.any(cdist(pts_i, pts_j) < threshold):
                        G.add_edge(idx_i, idx_j)

            components = list(nx.connected_components(G))
            for k, comp in enumerate(components):
                subgraph = G.subgraph(comp).copy()
                self.graphs[f'view_{k + 1}'] = subgraph

        self.save_graph_info("all_graphs.csv")

    def save_graph_info(self, csv_path):
        record = {'filename': self.filename}
        for name, graph in self.graphs.items():
            serialized = []
            for _, data in graph.nodes(data=True):
                entity = data['entity']
                if entity.dxftype() == 'LINE':
                    serialized.append({
                        'type': 'LINE',
                        'start': [entity.dxf.start.x, entity.dxf.start.y],
                        'end': [entity.dxf.end.x, entity.dxf.end.y]
                    })
                elif entity.dxftype() == 'CIRCLE':
                    serialized.append({
                        'type': 'CIRCLE',
                        'center': [entity.dxf.center.x, entity.dxf.center.y],
                        'radius': entity.dxf.radius
                    })
                elif entity.dxftype() == 'ARC':
                    serialized.append({
                        'type': 'ARC',
                        'center': [entity.dxf.center.x, entity.dxf.center.y],
                        'radius': entity.dxf.radius,
                        'start_angle': entity.dxf.start_angle,
                        'end_angle': entity.dxf.end_angle
                    })
            record[name] = json.dumps(serialized)

        try:
            df = pd.read_csv(csv_path)
            df = df[df['filename'] != self.filename]
        except FileNotFoundError:
            df = pd.DataFrame()

        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    def export_from_saved_csv(self, csv_path, filename, output_path):
        df = pd.read_csv(csv_path)
        row = df[df['filename'] == filename]
        if row.empty:
            print(f"[INFO] Чертеж '{filename}' не найден в {csv_path}")
            return

        new_doc = ezdxf.new(dxfversion="R2010")
        msp = new_doc.modelspace()

        for column in row.columns:
            if column == 'filename':
                continue
            try:
                cell = row.iloc[0][column]
                if not isinstance(cell, str) or pd.isna(cell) or not cell.strip():
                    continue  # пропустить пустую/невалидную ячейку
                data = json.loads(cell)
                for entity in data:
                    if entity['type'] == 'LINE':
                        msp.add_line(entity['start'], entity['end'])
                    elif entity['type'] == 'CIRCLE':
                        msp.add_circle(entity['center'], entity['radius'])
                    elif entity['type'] == 'ARC':
                        msp.add_arc(entity['center'], entity['radius'], entity['start_angle'], entity['end_angle'])
            except Exception as e:
                print(f"[ERROR] Ошибка чтения графа {column}: {e}")

        new_doc.saveas(output_path)

    def plot_graphs(self, figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.colormaps.get_cmap("tab10")  # без второго аргумента
        color_list = [cmap(i % cmap.N) for i in range(len(self.graphs))]

        for i, (view_name, graph) in enumerate(self.graphs.items()):
            color = color_list[i]
            for _, data in graph.nodes(data=True):
                entity = data["entity"]
                points = self.extract_points(entity)
                if not points:
                    continue
                if entity.dxftype() == "LINE":
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.plot(x, y, color=color, linewidth=1)
                else:
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.scatter(x, y, color=color, s=10)

        ax.set_title("Разделённые виды (графы)")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.show()


processor = DXFGraphProcessor("/Users/vrpivnev2h/PycharmProjects/diplom/data/dxf/example3.dxf")

processor.build_graphs()

processor.plot_graphs()

processor = DXFGraphProcessor()

processor.export_from_saved_csv(
    csv_path="all_graphs.csv",
    filename="example3.dxf",
    output_path="my_part_graphs_output.dxf"
)