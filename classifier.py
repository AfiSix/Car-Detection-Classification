import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import config

model_file = config.model_file
label_file = config.label_file
input_layer = config.input_layer
output_layer = config.output_layer
classifier_input_size = config.classifier_input_size


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    # Открытие файла модели и загрузка графа
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    # Открытие файла меток и загрузка списка меток
    with open(label_file, "r", encoding='cp1251') as ins:
        for line in ins:
            label.append(line.rstrip())

    return label


def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # Метод интерполяции для изменения размера изображения
    if h > sh or w > sw:  # Уменьшение изображения
        interp = cv2.INTER_AREA
    else:  # Растяжение изображения
        interp = cv2.INTER_CUBIC

    # Соотношение сторон изображения
    aspect = w / h
    # Вычисление масштаба и размера паддинга
    if aspect > 1:  # Горизонтальное изображение
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # Вертикальное изображение
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # Квадратное изображение
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # Установка цвета для паддинга
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple,
                                               np.ndarray)):  # Цветное изображение, но только один цвет для паддинга
        padColor = [padColor] * 3

    # Масштабирование и добавление паддинга
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


class Classifier():
    def __init__(self):
        # Загрузка графа модели
        self.graph = load_graph(model_file)
        # Загрузка меток (классов)
        self.labels = load_labels(label_file)

        # Имена входного и выходного слоев модели
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        # Получение операций для входа и выхода
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        # Создание сессии для работы с моделью
        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()  # Граф становится доступным только для чтения

    def predict(self, img):
        # Преобразование BGR в RGB (OpenCV использует BGR)
        img = img[:, :, ::-1]
        # Масштабирование и добавление паддинга
        img = resizeAndPad(img, classifier_input_size)

        # Добавляем четвёртое измерение, так как TensorFlow ожидает список изображений
        img = np.expand_dims(img, axis=0)

        # Масштабируем изображение в диапазон, используемый в обученной модели
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1.

        # Получаем результат предсказания модели
        results = self.sess.run(self.output_operation.outputs[0], {
            self.input_operation.outputs[0]: img
        })
        results = np.squeeze(results)

        # Получаем топ-3 предсказания
        top = 3
        top_indices = results.argsort()[-top:][::-1]
        classes = []
        for ix in top_indices:
            # Разделяем метку (марка и модель) и добавляем в список классов
            make_model = self.labels[ix].split('\t')
            classes.append({"make": make_model[0], "model": make_model[1], "prob": str(results[ix])})

        return (classes)
