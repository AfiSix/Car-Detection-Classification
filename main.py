import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import time
import cv2
import os
import classifier


# Функция для загрузки и обработки изображения
def process_image(image_path):
    yolo_path = 'yolo-coco'

    # Загрузка меток классов для YOLO
    labelsPath = os.path.join(yolo_path, "coco.names")
    LABELS = open(labelsPath).read().strip().split("\n")

    # Инициализация цветов для каждого класса
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Пути к весам и конфигурации модели YOLO
    weightsPath = os.path.join(yolo_path, "yolov3.weights")
    configPath = os.path.join(yolo_path, "yolov3.cfg")

    # Инициализация классификатора
    car_color_classifier = classifier.Classifier()

    # Загрузка модели YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Параметры для детекции
    conf = 0.5
    thr = 0.6

    # Загрузка изображения
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # Получение имен выходных слоев YOLO
    layer_names = net.getLayerNames()
    if isinstance(net.getUnconnectedOutLayers()[0], list):
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    else:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Преобразование изображения в blob и выполнение прогноза YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Списки для хранения детектированных объектов
    boxes = []
    confidences = []
    classIDs = []

    # Проход по каждому детектированному объекту
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Фильтрация слабых предсказаний
            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Применение подавления ненужных пересечений
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thr)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            if classIDs[i] == 2:  # Класс "car" для автомобилей
                start = time.time()
                result = car_color_classifier.predict(image[max(y, 0):y + h, max(x, 0):x + w])
                end = time.time()
                print("[INFO] classifier took {:.6f} seconds".format(end - start))
                text = "{}: {:.4f}".format(result[0]['make'], float(result[0]['prob']))
                cv2.putText(image, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)
                cv2.putText(image, result[0]['model'], (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)

    # Показываем изображение с результатами
    imgresize = cv2.resize(image, (960, 520))
    return imgresize


# Функция для открытия окна выбора файла
def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    if file_path:
        # Проверяем, что загруженный файл имеет допустимое расширение
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path.set(file_path)  # Обновляем переменную с путем к изображению
            img = cv2.imread(file_path)
            img_resize = cv2.resize(img, (250, 140))  # Изменим размер для отображения на экране
            img_bgr = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_bgr))  # Преобразуем в формат Tkinter
            panel.configure(image=img_tk)  # Обновляем изображение на экране
            panel.image = img_tk  # Сохраняем ссылку на изображение
            status_label.config(text="Изображение загружено!", fg="blue")  # Обновляем статус
        else:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение с расширением .jpg, .jpeg или .png.")
            status_label.config(text="Ошибка загрузки изображения.", fg="red")

# Функция для запуска анализа
def start_analysis():
    if image_path.get():  # Проверяем, загружено ли изображение
        status_label.config(text="Обработка…", fg="orange")
        progress_bar.start()  # Запускаем индикатор прогресса
        root.update_idletasks()  # Обновляем интерфейс

        try:
            result_image = process_image(image_path.get())
            cv2.imshow("Processed Image", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            status_label.config(text="Анализ завершен.", fg="green")
            progress_bar.stop()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обработке изображения: {e}")
            status_label.config(text="Ошибка.", fg="red")
            progress_bar.stop()
    else:
        messagebox.showerror("Ошибка", "Сначала загрузите изображение!")



# Создаем GUI
root = tk.Tk()
root.title("Car Make and Model Classifier")
root.geometry("600x500")

# Виджет для отображения изображения
from PIL import Image, ImageTk

panel = tk.Label(root)
panel.pack(pady=10)

# Заголовок
title_label = tk.Label(root, text="Выберите изображение и начните анализ", font=("Arial", 14))
title_label.pack(pady=5)

# Кнопка для загрузки изображения
btn_open = tk.Button(root, text="Загрузить изображение", command=open_image, font=("Arial", 14))
btn_open.pack(pady=10)

# Статус обработки
status_label = tk.Label(root, text="Ожидание загрузки...", font=("Arial", 12))
status_label.pack(pady=5)

# Прогресс-бар
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
progress_bar.pack(pady=10)

# Кнопка для начала анализа
btn_start = tk.Button(root, text="Начать анализ", command=start_analysis, font=("Arial", 14))
btn_start.pack(pady=20)

# Переменная для хранения пути к изображению
image_path = tk.StringVar()

root.mainloop()
