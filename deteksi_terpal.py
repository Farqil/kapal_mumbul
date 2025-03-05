import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge

class TerpalDetector(Node):
    def __init__(self):
        super().__init__('terpal_detector')
        self.publisher_x = self.create_publisher(Float32, 'target_x', 10)
        self.publisher_color = self.create_publisher(String, 'target_color', 10)
        self.bridge = CvBridge()

        self.cap = cv.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.detect_terpal)

    def detect_terpal(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        height, width, _ = frame.shape
        center_x = width // 2
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Rentang warna untuk biru dan merah
        color_ranges = {
            "Blue": [([90, 50, 50], [100, 255, 255]),
                     ([130, 50, 50], [160, 255, 255])],
            "Red": [([0, 100, 100], [10, 255, 255]),
                    ([170, 150, 100], [180, 255, 255])]
        }

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

            # Kirim data ke ROS2
            msg_x = Float32()
            msg_x.data = float(box_center_x - center_x)
            self.publisher_x.publish(msg_x)

            msg_color = String()
            msg_color.data = detected_color
            self.publisher_color.publish(msg_color)

            if detected_color == "Blue":
                box_color = (255, 0, 0)
            else:
                box_color = (0, 0, 255)

            cv.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv.line(frame, (box_center_x, 0), (box_center_x, height), box_color, 2)
            cv.putText(frame, f"Target: {detected_color}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
        cv.imshow("Terpal Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = TerpalDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
