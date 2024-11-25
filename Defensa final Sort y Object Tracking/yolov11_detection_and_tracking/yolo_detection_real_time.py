import cv2  # Librería para manejo de imágenes y video en tiempo real
import time  # Librería para medir tiempos de ejecución
# Importa una clase personalizada para detección YOLO
from yolo_detector import YoloDetector
# Importa una clase personalizada para el rastreador de objetos
from tracker import Tracker
# Ruta al modelo YOLO preentrenado
MODEL_PATH = "yolov11_detection_and_tracking/models/yolo11x.pt"


def main():
    # Inicializa el detector YOLO con un modelo específico y configura los umbrales de confianza e IOU
    detector = YoloDetector(model_path=MODEL_PATH,
                            confidence=0.4, iou_threshold=0.5)

    # Inicializa un rastreador de objetos (DeepSORT en este caso)
    tracker = Tracker()

    # Configura el acceso a la cámara (índice 0 indica la cámara predeterminada del sistema)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Configura la resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verifica si la cámara está disponible
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        exit()

    while True:
        # Captura un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            # Mensaje de error si no se puede leer el frame
            print("Failed to grab frame from camera.")
            break

        # Mide el tiempo de inicio del procesamiento
        start_time = time.perf_counter()

        # Realiza detección de objetos en el frame actual
        detections = detector.detect(frame)

        # Realiza el seguimiento de los objetos detectados en el frame
        tracking_ids, boxes = tracker.track(detections, frame)

        # Dibuja las cajas delimitadoras y los IDs de seguimiento en el frame
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            # Convierte las coordenadas a enteros
            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255),
                          2)  # Dibuja un rectángulo rojo
            cv2.putText(frame, f"ID {tracking_id}", (x1, y1 - 10),  # Escribe el ID encima del rectángulo
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mide el tiempo de fin del procesamiento y calcula los FPS
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps:.2f}")

        # Muestra el frame procesado con las detecciones y seguimientos
        cv2.imshow("Real-Time Object Detection and Tracking", frame)

        # Escucha si el usuario presiona 'q' o la tecla ESC para salir del bucle
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Libera los recursos de la cámara y cierra las ventanas abiertas
    cap.release()
    cv2.destroyAllWindows()


# Punto de entrada principal del programa
if __name__ == "__main__":
    main()
