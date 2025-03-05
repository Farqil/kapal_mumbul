import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandLong
from mavros_msgs.msg import State, Altitude
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')

        self.command_client = self.create_client(CommandLong, '/mavros/cmd/command')

        self.altitude_sub = self.create_subscription(Altitude, '/mavros/altitude', self.altitude_callback, 10)
        self.terpal_sub = self.create_subscription(Bool, '/terpal_detected', self.terpal_callback, 10)

        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.current_altitude = 60.0 
        self.terpal_detected = False
        self.state = "SEARCHING"

    def altitude_callback(self, msg):
        self.current_altitude = msg.amsl
        self.get_logger().info(f"Altitude: {self.current_altitude:.2f} m")

    def terpal_callback(self, msg):
        self.terpal_detected = msg.data

    def send_velocity(self, x, y, z):
        """Mengirim perintah ke UAV untuk bergerak"""
        vel_msg = Twist()
        vel_msg.linear.x = x
        vel_msg.linear.y = y
        vel_msg.linear.z = z
        self.vel_pub.publish(vel_msg)

    def drop_payload(self):
        """Mengirim perintah MAV_CMD_DO_SET_SERVO untuk dropping payload"""
        self.get_logger().info("Executing Payload Drop!")

        request = CommandLong.Request()
        request.command = 183
        request.param1 = float(9)
        request.param2 = float(2000)
        request.param3 = 0.0
        request.param4 = 0.0
        request.param5 = 0.0
        request.param6 = 0.0
        request.param7 = 0.0

        future = self.command_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("Payload Dropped Successfully!")

    def execute_mission(self):
        """Logika autonomous UAV"""
        if self.state == "SEARCHING":
            if self.terpal_detected:
                self.get_logger().info("Terpal detected, moving to ALIGNING.")
                self.state = "ALIGNING"
        
        elif self.state == "ALIGNING":
            self.get_logger().info("Aligning with target...")
            self.send_velocity(0.0, 0.0, 0.0)
            self.state = "APPROACHING"
        
        elif self.state == "APPROACHING":
            self.get_logger().info("Moving away and descending to 2m...")
            self.send_velocity(-2.0, 0.0, -2.0)
            if self.current_altitude <= 2.0:
                self.state = "DROPPING"
        
        elif self.state == "DROPPING":
            self.drop_payload()
            self.state = "MISSION_COMPLETE"
        
        elif self.state == "MISSION_COMPLETE":
            self.get_logger().info("Mission complete. Holding position.")
            self.send_velocity(0.0, 0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigator()
    rate = node.create_rate(1)

    while rclpy.ok():
        node.execute_mission()
        rclpy.spin_once(node)
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
