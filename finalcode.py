import cv2
import numpy as np

# Path settings
input_path = './image/bus.jpg'  # Path to input image
output_path = './output_with_detections.jpg'  # Path to save the output image

try:
    # Read the input image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image from {input_path}")
    height, width, channels = image.shape
    print('Original image shape:', height, width, channels)

    # Step 1: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)

    # Step 2: Apply edge detection
    edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Edge Detection', edges)

    # Step 3: Prepare YOLO model for object detection
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load YOLO configuration and weights
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Step 4: Perform object detection
    class_ids = []
    confidence_scores = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]  # Confidence scores for all classes
            class_id = np.argmax(scores)  # Get the class ID with the highest score
            confidence = scores[class_id]  # Get the highest confidence score

            # Filter out weak detections
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidence_scores.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
    print('Number of final objects:', len(indices))

    # Step 5: Draw bounding boxes and labels on the image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indices:
        i = i[0] if isinstance(i, tuple) else i
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        area = w * h
        print(f"Class: {label}, Position: ({x}, {y}), Size: {w}x{h}, Area: {area}")

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} ({w}x{h})", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Step 6: Combine edge detection with object detection
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels
    combined_image = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)

    # Save and display the final image
    cv2.imwrite(output_path, combined_image)
    print(f"Final image saved to {output_path}")

    # Show the combined result
    cv2.imshow('Object Detection + Edge Detection', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
