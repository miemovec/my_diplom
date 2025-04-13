import os
import pickle

def list_circles_in_pkl_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    if not files:
        print("❌ В папке нет .pkl-файлов.")
        return

    for file in sorted(files):
        file_path = os.path.join(folder_path, file)
        with open(file_path, "rb") as f:
            comp = pickle.load(f)

        G = comp.get("nx_graph", {})
        circles = [
            (n, d.get("radius"))
            for n, d in G.nodes(data=True)
            if d.get("type") == "circle"
        ]

        print(f"\n📄 Файл: {file}")
        if circles:
            print(f"  🔵 Найдено окружностей: {len(circles)}")
            for n, r in circles:
                print(f"    - Центр: {n}, Радиус: {r}")
        else:
            print("  ⚠️  Нет окружностей в этом графе.")

# Вызов
list_circles_in_pkl_folder("/Users/vrpivnev2h/PycharmProjects/diplom/data/pkl")
