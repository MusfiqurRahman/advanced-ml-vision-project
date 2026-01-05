from flask import Flask, render_template, request, Response
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from CustomCNN import CustomCNN

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=7)  # Ensure the number of output classes matches your dataset
# Load the state_dict from the file
state_dict = torch.load("custom_v2_7.pth", map_location=device, weights_only=True)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Define preprocessing pipeline for the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Update to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Class labels specific to your dataset
class_labels = ["marker", "monitor", "mouse", "pen", "pencil", "ruler", "scissors"]

# Global variable to store the video URL
video_url = ""


def generate_frames(video_url):
    cap = cv2.VideoCapture(video_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for prediction
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = class_labels[predicted.item()]

        # Overlay label on the frame
        cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/', methods=['GET', 'POST'])
def index():
    global video_url
    if request.method == 'POST':
        video_url = request.form.get('video_url')
    return render_template('index.html', video_url=video_url)


@app.route('/video_feed')
def video_feed():
    global video_url
    if not video_url:
        return "No video URL provided"
    return Response(generate_frames(video_url), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
