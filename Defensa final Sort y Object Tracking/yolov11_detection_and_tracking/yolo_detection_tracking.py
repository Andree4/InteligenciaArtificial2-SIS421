import cv2  # Librería de visión por computadora para trabajar con imágenes y videos
import time  # Librería para medir y trabajar con tiempo
# Clase personalizada para detección basada en YOLO
from yolo_detector import YoloDetector
from tracker import Tracker  # Clase personalizada para el seguimiento de objetos

# Ruta al modelo YOLO preentrenado
MODEL_PATH = "yolov11_detection_and_tracking/models/yolo11x.pt"

# Ruta al archivo de video que se procesará
VIDEO_PATH = r"C:\Tareas hechas\FINAL IA 2\yolov11_detection_and_tracking\assets\clase.mp4"


def main():
    """
    Función principal que ejecuta el flujo de detección y seguimiento de objetos en un video.
    """
    # Inicializa el detector YOLO con el modelo especificado y un umbral de confianza bajo (0.2)
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)

    # Inicializa el rastreador (asume un sistema como DeepSORT para seguir objetos detectados entre frames)
    tracker = Tracker()

    # Abre el archivo de video especificado
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Comprueba si el archivo de video se pudo abrir correctamente
    if not cap.isOpened():
        # Imprime un error si el video no se puede abrir
        print("Error: Unable to open video file.")
        exit()  # Sale del programa si no se puede acceder al archivo

    # Bucle principal para procesar cada frame del video
    while True:
        # Lee un frame del video
        ret, frame = cap.read()

        # Si no se puede leer el frame (fin del video o error), se rompe el bucle
        if not ret:
            break

        # Marca el tiempo de inicio del procesamiento del frame
        start_time = time.perf_counter()

        # Realiza detecciones en el frame utilizando el modelo YOLO
        detections = detector.detect(frame)

        # Realiza el seguimiento de los objetos detectados en el frame actual
        tracking_ids, boxes = tracker.track(detections, frame)

        # Dibuja los resultados de detección y seguimiento en el frame
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            # Dibuja un rectángulo alrededor de cada objeto detectado (bounding box)
            cv2.rectangle(frame,
                          # Esquina superior izquierda
                          (int(bounding_box[0]), int(bounding_box[1])),
                          # Esquina inferior derecha
                          (int(bounding_box[2]), int(bounding_box[3])),
                          (0, 0, 255),  # Color del rectángulo (rojo)
                          2)  # Grosor del rectángulo

            # Escribe el ID del objeto sobre el cuadro detectado
            cv2.putText(frame,
                        f"{str(tracking_id)}",  # Texto con el ID del objeto
                        # Posición del texto
                        (int(bounding_box[0]), int(bounding_box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,  # Fuente del texto
                        0.5,  # Tamaño de la fuente
                        (0, 255, 0),  # Color del texto (verde)
                        2)  # Grosor del texto

        # Calcula el tiempo de procesamiento del frame y el FPS
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)  # FPS = 1 / Tiempo por frame
        print(f"Current fps: {fps}")  # Imprime los FPS calculados

        # Muestra el frame procesado con las detecciones y seguimientos
        cv2.imshow("Frame", frame)

        # Captura una tecla del usuario
        key = cv2.waitKey(1) & 0xFF
        # Si se presiona la tecla 'q' o ESC, se rompe el bucle para terminar el programa
        if key == ord("q") or key == 27:
            break

    # Libera el recurso del video una vez terminado el procesamiento
    cap.release()
    # Cierra todas las ventanas abiertas por OpenCV
    cv2.destroyAllWindows()


# Punto de entrada del programa: llama a la función principal cuando se ejecuta el script
if __name__ == "__main__":
    main()
