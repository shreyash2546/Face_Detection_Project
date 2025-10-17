# Face_Detection_Project

👁️ Real-Time Face Detection & Unique Face Snapshot Saver
🎯 Overview

This Python project uses OpenCV to detect faces in real-time from your webcam. It automatically captures and saves only unique faces by comparing pixel differences — avoiding duplicate face snapshots. Each face is labeled with a unique ID and stored inside the detected_faces folder.

⚙️ Features

✅ Real-time face detection using Haar Cascade Classifier
✅ Automatic unique face recognition (prevents duplicates)
✅ Saves cropped face images locally
✅ Displays live face count and ID labels on the video feed
✅ Lightweight — no deep learning models required

🧩 Tech Stack

Language: Python

Library: OpenCV (cv2)

Algorithm: Haar Cascade Face Detection

🚀 How It Works

1.Opens your webcam feed.

2.Detects faces in each video frame.

3.Compares each new face with previously saved ones.

4.Saves only new unique faces to the detected_faces folder.

5.Displays a live window showing:

6.Number of detected faces

7.Unique ID labels for each face

💾 Output:
All unique face images are saved in the detected_faces folder after the program ends.
