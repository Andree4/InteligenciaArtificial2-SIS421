from deep_sort_realtime.deepsort_tracker import DeepSort
# Se importa la clase `DeepSort` del módulo `deep_sort_realtime`, que es una implementación de seguimiento avanzado.


class Tracker:
    """
    Clase que encapsula el rastreador de objetos utilizando la librería DeepSort.
    Permite realizar el seguimiento de objetos detectados en una secuencia de frames.
    """

    def __init__(self):
        """
        Constructor de la clase `Tracker`.
        Inicializa un objeto `DeepSort` con configuraciones específicas para el rastreo.
        """
        # Configura e inicializa el rastreador `DeepSort` con parámetros ajustados
        self.object_tracker = DeepSort(
            # Número máximo de frames sin actualización antes de eliminar un objeto.
            max_age=40,
            # Número de detecciones consecutivas requeridas para confirmar un objeto.
            n_init=2,
            # Umbral de superposición máxima para Non-Maximum Suppression (NMS).
            nms_max_overlap=0.5,
            # Umbral para la distancia coseno al comparar embeddings (similaridad de objetos).
            max_cosine_distance=0.4,
            # Presupuesto del modelo (controla cuántos embeddings se almacenan para cada objeto).
            nn_budget=None,
            # Permite sobrescribir la clase de los objetos rastreados.
            override_track_class=None,
            # Modelo utilizado para generar embeddings de características (MobileNet en este caso).
            embedder="mobilenet",
            # Indica si se debe utilizar precisión mixta (FP16) para mejorar el rendimiento.
            half=True,
            # Indica que los embeddings deben generarse con imágenes en formato BGR (por defecto en OpenCV).
            bgr=True,
            # Nombre del modelo de embeddings (opcional, no se especifica aquí).
            embedder_model_name=None,
            # Pesos del modelo de embeddings (opcional, no se especifica aquí).
            embedder_wts=None,
            # Indica si las cajas de detección son polígonos (falso, aquí son rectángulos).
            polygon=False,
            # Permite especificar una fecha de referencia para seguimiento (opcional, no se utiliza).
            today=None
        )

    def track(self, detections, frame):
        """
        Método que realiza el seguimiento de objetos detectados en un frame.

        Args:
            detections (list): Lista de detecciones en el frame, cada detección contiene la información necesaria
                               para inicializar o actualizar un seguimiento.
            frame (numpy.ndarray): Imagen actual en la que se realizará el seguimiento.

        Returns:
            tracking_ids (list): Lista de IDs de seguimiento para los objetos confirmados.
            boxes (list): Lista de cajas delimitadoras (LTRB: left, top, right, bottom) asociadas a los objetos rastreados.
        """
        # Actualiza los rastreos con las detecciones actuales y el frame correspondiente
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        # Inicializa listas para almacenar los resultados del seguimiento
        tracking_ids = []  # IDs únicos de los objetos rastreados
        boxes = []  # Coordenadas de las cajas delimitadoras de los objetos

        # Recorre los objetos rastreados devueltos por `DeepSort`
        for track in tracks:
            # Verifica si el objeto rastreado está confirmado (tiene suficientes actualizaciones consecutivas)
            if not track.is_confirmed():
                continue  # Si no está confirmado, se omite este objeto

            # Agrega el ID único del objeto rastreado a la lista
            tracking_ids.append(track.track_id)

            # Obtiene las coordenadas de la caja delimitadora del objeto en formato LTRB (Left, Top, Right, Bottom)
            ltrb = track.to_ltrb()
            boxes.append(ltrb)  # Agrega las coordenadas de la caja a la lista

        # Devuelve las listas de IDs de seguimiento y cajas delimitadoras
        return tracking_ids, boxes
