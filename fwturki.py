import cv2 as cv
import numpy as np

color_ranges = {
    "Biru": ([100, 50, 50], [140, 255, 255]),
    "Merah": ([0, 150, 70], [10, 255, 255]),
    "Hijau": ([40, 40, 40], [80, 255, 255])
}

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(frame, frame, mask=mask)

        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            num_sides = len(approx)

            if num_sides == 3:
                shape = "Segitiga"
            elif num_sides == 4:
                shape = "Persegi/Panjang"
            elif num_sides > 4:
                shape = "Lingkaran"
            else:
                shape = "Tidak Diketahui"

            cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            x, y = approx[0][0]
            cv.putText(frame, f"{color} - {shape}", (x, y - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow("Deteksi Warna & Bentuk", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
