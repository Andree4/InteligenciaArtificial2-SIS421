# Importa la clase YOLO para cargar el modelo y realizar predicciones.
from ultralytics import YOLO
# Librería para operaciones numéricas, utilizada aquí para manejar matrices.
import numpy as np
# Librería para visión por computadora, utilizada para la supresión de no máximos (NMS).
import cv2


class YoloDetector:
    """
    Clase para detección de objetos utilizando el modelo YOLO (Ultralytics).
    """

    def __init__(self, model_path, confidence=0.4, iou_threshold=0.5):
        """
        Inicializa el detector YOLO con los parámetros especificados.

        Args:
        - model_path (str): Ruta al archivo del modelo YOLO preentrenado.
        - confidence (float): Umbral mínimo de confianza para las detecciones.
        - iou_threshold (float): Umbral de IoU para la supresión de no máximos.
        """
        self.model = YOLO(
            model_path)  # Carga el modelo YOLO desde la ruta especificada.
        # Lista de clases que queremos detectar. Aquí solo se detectan personas.
        self.classList = ["person"]
        # Umbral mínimo para considerar una detección como válida.
        self.confidence = confidence
        # Umbral para determinar superposición máxima aceptable entre cajas.
        self.iou_threshold = iou_threshold

    def detect(self, image):
        """
        Realiza la detección de objetos en una imagen.

        Args:
        - image (numpy.ndarray): Imagen de entrada en la que se buscarán objetos.

        Returns:
        - list: Lista de detecciones en formato ([x, y, w, h], class_id, confianza).
        """
        # Realiza la predicción en la imagen con los umbrales especificados y sin mensajes verbosos.
        results = self.model.predict(
            image, conf=self.confidence, iou=self.iou_threshold, verbose=False
        )

        # Extrae el primer resultado (en caso de que haya múltiples)
        result = results[0]

        # Procesa los resultados de YOLO para generar una lista de detecciones filtradas.
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        """
        Procesa los resultados de YOLO y filtra detecciones según las clases y el umbral de confianza.

        Args:
        - result: Resultado del modelo YOLO (contiene cajas, clases, y confidencias).

        Returns:
        - list: Lista de detecciones en formato ([x, y, w, h], class_id, confianza).
        """
        boxes = result.boxes  # Obtiene las cajas detectadas del resultado de YOLO.
        detections = []  # Lista para almacenar las detecciones filtradas.

        for box in boxes:
            # Extrae las coordenadas de las esquinas superior izquierda (x1, y1) e inferior derecha (x2, y2).
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calcula el ancho y alto de la caja delimitadora.
            w, h = x2 - x1, y2 - y1

            # Obtiene el índice de la clase detectada.
            class_number = int(box.cls[0])

            # Filtra detecciones que no pertenezcan a las clases de interés.
            if result.names[class_number] not in self.classList:
                continue

            # Obtiene la confianza de la detección.
            conf = float(box.conf[0])

            # Filtra detecciones con confianza menor al umbral especificado.
            if conf < self.confidence:
                continue

            # Agrega la detección en el formato ([x, y, w, h], class_id, confianza).
            detections.append((([x1, y1, w, h]), class_number, conf))

        # Aplica una supresión de no máximos para eliminar cajas redundantes.
        detections = self.non_max_suppression(detections)
        return detections

    def non_max_suppression(self, detections, iou_threshold=0.5):
        """
        Realiza la supresión de no máximos (NMS) para eliminar cajas redundantes basándose en IoU.

        Args:
        - detections (list): Lista de detecciones en formato ([x, y, w, h], class_id, confianza).
        - iou_threshold (float): Umbral de IoU para la NMS.

        Returns:
        - list: Lista de detecciones filtradas después de aplicar NMS.
        """
        # Si no hay detecciones, devuelve una lista vacía.
        if len(detections) == 0:
            return []

        # Extrae las cajas y las puntuaciones de confianza.
        # Coordenadas de las cajas.
        boxes = np.array([det[0] for det in detections], dtype=np.int32)
        # Confidencias.
        scores = np.array([det[2] for det in detections], dtype=np.float32)

        # Si no hay cajas o puntajes válidos, devuelve una lista vacía.
        if len(boxes) == 0 or len(scores) == 0:
            return []

        # Aplica NMS utilizando la función `cv2.dnn.NMSBoxes`.
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),  # Lista de cajas en formato [x, y, w, h].
            scores=scores.tolist(),  # Lista de puntuaciones de confianza.
            score_threshold=self.confidence,  # Umbral de puntuación mínima.
            # Umbral de IoU para la supresión de cajas.
            nms_threshold=iou_threshold
        )

        # Si no hay índices válidos después de NMS, devuelve una lista vacía.
        if indices is None or len(indices) == 0:
            return []

        # Asegura que los índices estén en un formato plano (no matriz).
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()

        # Filtra las detecciones utilizando los índices seleccionados.
        filtered_detections = [detections[i] for i in indices]
        return filtered_detections
