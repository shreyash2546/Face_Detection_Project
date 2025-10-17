import cv2
import os

# Create a folder to save face snapshots
save_dir = "detected_faces"
os.makedirs(save_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

saved_faces = []  # Store cropped faces for comparison
face_id = 0  # ID for each unique face

def is_new_face(face_img, saved_faces, threshold=0.6):
    """
    Compares new face with saved faces using simple pixel difference.
    Returns True if the face is new.
    """
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (100, 100))  # Resize for comparison

    for saved in saved_faces:
        diff = cv2.absdiff(face_gray, saved)
        non_zero = cv2.countNonZero(diff)
        if non_zero / (100*100) < threshold:  # If similar enough, consider duplicate
            return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = frame[y:y+h, x:x+w]

        if is_new_face(face_img, saved_faces):
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (100, 100))
            saved_faces.append(face_gray)

            face_filename = os.path.join(save_dir, f"face_{face_id}.jpg")
            cv2.imwrite(face_filename, face_img)
            face_id += 1

        # Label face with ID
        cv2.putText(frame, f"ID:{face_id-1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display number of faces detected
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Detection", frame)

    # Stop video if 'q' pressed or window closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video stopped by user.")
        break
    if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        print("Video window closed.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {face_id} unique face snapshots in '{save_dir}' folder.")
