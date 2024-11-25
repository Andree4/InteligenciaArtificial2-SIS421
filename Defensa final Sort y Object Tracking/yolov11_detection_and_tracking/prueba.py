import cv2
print(cv2.__version__)
cap = cv2.VideoCapture(0)  # Usar cámara para prueba
if not cap.isOpened():
    print("Error: Unable to open the camera.")
else:
    print("Camera opened successfully.")
cap.release()
