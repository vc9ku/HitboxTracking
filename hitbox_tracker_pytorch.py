import torch
from torchvision import models, transforms
import cv2
import numpy as np


def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval() 
    return model


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(frame)


def process_predictions(predictions, confidence_threshold=0.5):
    boxes = []
    labels = []
    scores = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score > confidence_threshold:
            boxes.append(box)
            labels.append(label.item())
            scores.append(score.item())
    return boxes, labels, scores


def load_coco_classes():

    return [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]


def process_video(video_path, output_path, confidence_threshold=0.5):

    model = load_model()
    classes = load_coco_classes()


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = preprocess_frame(rgb_frame)
        

        with torch.no_grad():
            predictions = model([tensor_frame])[0]
        

        boxes, labels, scores = process_predictions(predictions, confidence_threshold)
        

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            label_name = classes[label]
            color = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

        out.write(frame)
        

        cv2.imshow("Hitbox Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "Walkingman.mp4"
    output_path = "output.avi"
    process_video(video_path, output_path)
