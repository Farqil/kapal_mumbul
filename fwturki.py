import cv2 as cv
import numpy as np

color_ranges = {
    "Blue": [([33, 100, 222],[145, 255, 255]), 
             ([90, 50, 50], [100, 255, 255]),
             ([130, 50, 50], [160, 255, 255])],

    "Red": [([0, 100, 100], [10, 255, 255]),
            ([0, 150, 100], [10, 255, 255]),
            ([170, 150, 100], [180, 255, 255])]
}

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    center_x = width // 2
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    detected_color = None
    largest_contour = None
    largest_area = 0

    for color, ranges in color_ranges.items():
        for lower, upper in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            mask = cv.inRange(hsv, lower, upper)

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv.contourArea(contour)

                approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)

                if len(approx) == 4 and area > 1000 and area > largest_area:
                    largest_area = area
                    largest_contour = approx
                    detected_color = color

    if largest_contour is not None:
        x, y, w, h = cv.boundingRect(largest_contour)
        box_center_x = x + w // 2

        if detected_color == "Blue": box_color = (255, 0, 0)
        elif detected_color == "Red": box_color = (0, 0, 255)
        higlight_color = (0, 255, 0)

        cv.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv.polylines(frame, [largest_contour], True, higlight_color, 3)
        cv.line(frame, (box_center_x, 0), (box_center_x, height), box_color, 2)

        cv.putText(frame, "Target Confirmed", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.putText(frame, f"{detected_color} Square", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else:
        cv.putText(frame, "No Target Acquired", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)

    cv.imshow("output", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
