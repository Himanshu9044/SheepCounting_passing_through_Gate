import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt") #for better accuracy we can use ("yolo26x.pt")

cap = cv2.VideoCapture("sheeps passing through the gate(1).mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
box_annotator = sv.BoxAnnotator()
tracker = sv.ByteTrack()
line_y = 510
gate_x1 = 400
gate_x2 = 960

sheep_count = 0
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[18])[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = tracker.update_with_detections(detections)

    for i in range(len(detections.xyxy)):

        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        class_id = detections.class_id[i]
        track_id = detections.tracker_id[i]

        label = model.names[class_id]

        if label == "sheep":

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # only consider sheep inside the gate
            if gate_x1 < cx < gate_x2:

                # Draw bounding box only inside gate
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

                
                if cy > line_y and track_id not in counted_ids:
                    sheep_count += 1
                    counted_ids.add(track_id)


    cv2.line(frame,(gate_x1,line_y),(gate_x2,line_y),(0,255,0),3)

    
    cv2.line(frame,(gate_x1,0),(gate_x1,frame.shape[0]),(0,255,255),2)
    cv2.line(frame,(gate_x2,0),(gate_x2,frame.shape[0]),(0,255,255),2)

    cv2.putText(frame,
                f"Sheep Passed Gate: {sheep_count}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Sheep Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
