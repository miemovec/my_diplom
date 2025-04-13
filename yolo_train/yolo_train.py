# python -c "import torch; print(torch.backends.mps.is_available())"

# yolo detect train data=/Users/vrpivnev2h/PycharmProjects/diplom/yolo_train/dataset1/data.yaml model=yolov8n.pt epochs=50 imgsz=640 device='mps'


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

input_dir = "/Users/vrpivnev2h/PycharmProjects/diplom/yolo_train/validation_drawings"
output_dir = "/Users/vrpivnev2h/PycharmProjects/diplom/yolo_train/validation_drawings_padded"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        h, w = img.shape[:2]

        # Проверка: должно быть 480x640
        if h != 480 or w != 640:
            print(f"[!] Пропущено {filename}, размер {w}x{h}")
            continue

        # Считаем отступы для паддинга до 640x640
        pad_vert = 640 - h
        top = pad_vert // 2
        bottom = pad_vert - top

        # Добавляем белую рамку
        padded = cv2.copyMakeBorder(
            img, top, bottom, 0, 0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        cv2.imwrite(os.path.join(output_dir, filename), padded)

print("✅ Все изображения дополнены до 640x640 и сохранены.")


# Загружаем модель
#model = YOLO("/Users/vrpivnev2h/PycharmProjects/diplom/runs/detect/train3/weights/best.pt")

#results = model.predict(source="/Users/vrpivnev2h/PycharmProjects/diplom/yolo_train/validation_drawings", save=False, conf=0.1)

#for r in results:
#    img = r.plot()
#    plt.figure(figsize=(10, 10))
#    plt.imshow(img)
#    plt.axis("off")
#    plt.show()


import os
import cv2