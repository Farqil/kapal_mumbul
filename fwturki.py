import cv2 as cv
import numpy as np

color_ranges = {
    "Biru": ([33, 100, 222],[145, 255, 255])
}

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    center_x = width // 2
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(frame, frame, mask=mask)
        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv.contourArea)
            approx = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
            num_sides = len(approx)
            shape = None
            if num_sides == 3:
                shape = "Segitiga"
            elif num_sides == 4:
                x, y, w, h = cv.boundingRect(max_contour)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Persegi"

            if shape:
                if shape == "Persegi":
                    x, y, w, h = cv.boundingRect(max_contour)
                    box_center_x = x + w // 2
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv.line(frame, (box_center_x, 0), (box_center_x, height), (0, 255, 255), 2)
                cv.putText(frame, f"{color} - {shape}", (x, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    cv.imshow("Deteksi Segitiga & Persegi", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
