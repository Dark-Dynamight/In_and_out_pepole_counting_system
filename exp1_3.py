import cv2
import numpy as np
from collections import OrderedDict
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CentroidTracker class to keep track of objects and their IDs across frames
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # Object ID and their centroids
        self.disappeared = OrderedDict()  # Tracks how long an object has been missing
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distance between current centroids and input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# Load the pre-trained MobileNet-SSD model
def load_model():
    try:
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
        return net
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Check if the video file exists
def check_video_file(video_path):
    if not os.path.isfile(video_path):
        logging.error(f"Video file {video_path} does not exist.")
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

# Main function
def main(video_path):
    check_video_file(video_path)
    net = load_model()

    # Labels for the classes in the COCO dataset
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Load the video stream
    cap = cv2.VideoCapture(video_path)

    # Counters for people moving up and down
    up_count = 0
    down_count = 0
    dwell_times = {}  # Dictionary to track dwell times
    frame_counts = []  # List to store counts per frame

    up_counts = []
    down_counts = []
    frame_numbers = []

    # Create an instance of CentroidTracker
    ct = CentroidTracker()
    previous_y = {}
    line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    frame_number = 0

    # Video writer to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream.")
            break

        # Get frame dimensions
        (h, w) = frame.shape[:2]

        # Prepare the input blob for object detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        # Perform forward pass to get detections
        detections = net.forward()

        # Initialize a list to hold detected centroids
        centroids = []

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Only process confident detections (greater than 50%)
            if confidence > 0.5:
                # Get the class label
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    # Compute the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    # Calculate the centroid
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    centroids.append(centroid)

                    # Draw the bounding box and label
                    label = f"{CLASSES[idx]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the centroid tracker with the new centroids
        objects = ct.update(centroids)

        # Loop over the tracked objects
        for (object_id, centroid) in objects.items():
            current_y = centroid[1]

            # Check if this object has been seen before
            if object_id in previous_y:
                prev_y = previous_y[object_id]

                # Check if the person is moving up (from below the line to above it)
                if prev_y > line_position and current_y < line_position:
                    up_count += 1
                    logging.info(f"Person {object_id} moved up.")

                # Check if the person is moving down (from above the line to below it)
                elif prev_y < line_position and current_y > line_position:
                    down_count += 1
                    logging.info(f"Person {object_id} moved down.")

                # Track dwell time
                if object_id not in dwell_times:
                    dwell_times[object_id] = 0
                dwell_times[object_id] += 1  # Increment dwell time for this object

            # Store the current y-coordinate as the previous y-coordinate for the next frame
            previous_y[object_id] = current_y

        # Draw the horizontal line across the frame
        line_color = (0, 255, 0) if (up_count + down_count) < 10 else (0, 0, 255)  # Change color based on counts
        cv2.line(frame, (0, line_position), (w, line_position), line_color, 2)

        # Display real-time statistics
        cv2.putText(frame, f"Total Up: {up_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Down: {down_count}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Current Count: {len(centroids)}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Store counts for analysis
        up_counts.append(up_count)
        down_counts.append(down_count)
        frame_counts.append(len(centroids))  # Store the number of detected people
        frame_numbers.append(frame_number)
        
        # Save the annotated frame to the output video
        out.write(frame)

        # Show the frame with detections and counters
        cv2.imshow("People Counting", frame)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting video stream.")
            break

        frame_number += 1

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Create a DataFrame to store the counts
    data = {
        'Frame Number': frame_numbers,
        'Up Count': up_counts,
        'Down Count': down_counts,
        'Frame Count': frame_counts,
    }
    df = pd.DataFrame(data)

    # Calculate average dwell time
    average_dwell_time = sum(dwell_times.values()) / len(dwell_times) if dwell_times else 0
    logging.info(f"Average Dwell Time: {average_dwell_time:.2f} frames")

    # Save the DataFrame to an Excel file
    excel_file = 'people_counting_analysis.xlsx'
    df.to_excel(excel_file, index=False)
    logging.info(f"Data saved to {excel_file}")

    # Save the DataFrame to a CSV file
    csv_file = 'people_counting_analysis.csv'
    df.to_csv(csv_file, index=False)
    logging.info(f"Data saved to {csv_file}")

    # Plotting the results using Plotly for interactive graphs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame_numbers, y=up_counts, mode='lines', name='Up Count', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=frame_numbers, y=down_counts, mode='lines', name='Down Count', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=frame_numbers, y=frame_counts, mode='lines', name='Frame Count', line=dict(color='blue')))
    fig.update_layout(title='People Counting Analysis', xaxis_title='Frame Number', yaxis_title='Count')
    pyo.plot(fig, filename='people_counting_analysis.html')  # Save the graph as an HTML file

    # Save the graph as a PNG file using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, up_counts, label='Up Count', color='green')
    plt.plot(frame_numbers, down_counts, label='Down Count', color='red')
    plt.plot(frame_numbers, frame_counts, label='Frame Count', color='blue')
    plt.title('People Counting Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()
    plt.savefig('people_counting_graph.png')  # Save the graph as a PNG file
    plt.show()

if __name__ == "__main__":
    video_path = 'test.mp4'  # Change this to your video file path
    try:
        main(video_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")