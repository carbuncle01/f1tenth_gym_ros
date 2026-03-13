import rclpy
from rclpy.node import Node
import os
import sys
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from transforms3d import euler

from f110_jax.simulator import F110JaxSimulator


class JaxGymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')

        # 名前空間やトピックなどの既存パラメータ
        self.declare_parameter('ego_namespace', 'ego_racecar')
        self.declare_parameter('ego_odom_topic', 'odom')
        self.declare_parameter('ego_opp_odom_topic', 'opp_odom')
        self.declare_parameter('ego_scan_topic', 'scan')
        self.declare_parameter('ego_drive_topic', 'drive')
        self.declare_parameter('opp_namespace', 'opp_racecar')
        self.declare_parameter('opp_odom_topic', 'odom')
        self.declare_parameter('opp_ego_odom_topic', 'opp_odom')
        self.declare_parameter('opp_scan_topic', 'scan')
        self.declare_parameter('opp_drive_topic', 'drive')
        self.declare_parameter('scan_distance_to_base_link', 0.27)
        self.declare_parameter('scan_fov', 4.7)
        self.declare_parameter('scan_beams', 1080)
        self.declare_parameter('map_path', '')
        self.declare_parameter('map_img_ext', '.png')
        self.declare_parameter('num_agent', 2)

        self.declare_parameter('sx', 0.0)
        self.declare_parameter('sy', 0.0)
        self.declare_parameter('stheta', 0.0)
        self.declare_parameter('sx1', 1.0)
        self.declare_parameter('sy1', 1.0)
        self.declare_parameter('stheta1', 0.0)
        self.declare_parameter('kb_teleop', False)

        self.declare_parameter('sim_rate', 100.0)
        self.declare_parameter('scan_rate', 40.0)
        self.declare_parameter('odom_rate', 100.0)

        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')

        self.get_logger().info(f"Loading map: {self.get_parameter('map_path').value}")
        self.env = F110JaxSimulator(
            map_path=self.get_parameter('map_path').value,
            map_ext=self.get_parameter('map_img_ext').value,
            num_agents=num_agents,
            num_beams=self.get_parameter('scan_beams').value,
            fov=self.get_parameter('scan_fov').value,
            lidar_dist=self.get_parameter('scan_distance_to_base_link').value
        )

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_scan_topic = f"{self.ego_namespace}/{self.get_parameter('ego_scan_topic').value}"
        ego_drive_topic = f"{self.ego_namespace}/{self.get_parameter('ego_drive_topic').value}"
        ego_odom_topic = f"{self.ego_namespace}/{self.get_parameter('ego_odom_topic').value}"
        
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        
        if num_agents == 2:
            self.has_opp = True
            
            self.opp_namespace = self.get_parameter('opp_namespace').value
            opp_scan_topic = f"{self.opp_namespace}/{self.get_parameter('opp_scan_topic').value}"
            opp_drive_topic = f"{self.opp_namespace}/{self.get_parameter('opp_drive_topic').value}"
            opp_odom_topic = f"{self.opp_namespace}/{self.get_parameter('opp_odom_topic').value}"
            ego_opp_odom_topic = f"{self.ego_namespace}/{self.get_parameter('ego_opp_odom_topic').value}"
            opp_ego_odom_topic = f"{self.opp_namespace}/{self.get_parameter('opp_ego_odom_topic').value}"
            
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.obs, _, self.done, _ = self.env.reset(np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])
        else:
            self.has_opp = False
            self.obs, _, self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        # Hzから周期(秒)への変換
        sim_period = 1.0 / self.get_parameter('sim_rate').value
        scan_period = 1.0 / self.get_parameter('scan_rate').value
        odom_period = 1.0 / self.get_parameter('odom_rate').value

        # 分割・独立させたタイマーの設定
        self.drive_timer = self.create_timer(sim_period, self.drive_timer_callback)
        self.scan_timer = self.create_timer(scan_period, self.scan_timer_callback)
        self.odom_timer = self.create_timer(odom_period, self.odom_timer_callback)
        
        self.br = TransformBroadcaster(self)

        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=10)
        self.ego_drive_sub = self.create_subscription(AckermannDriveStamped, ego_drive_topic, self.drive_callback, 10)
        self.ego_reset_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.ego_reset_callback, qos_profile=best_effort_qos)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(AckermannDriveStamped, opp_drive_topic, self.opp_drive_callback, 10)
            self.opp_reset_sub = self.create_subscription(PoseStamped, '/goal_pose', self.opp_reset_callback, 10)
        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(Twist, '/cmd_vel', self.teleop_callback, 10)

    def drive_callback(self, msg):
        self.ego_requested_speed = msg.drive.speed
        self.ego_steer = msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, msg):
        self.opp_requested_speed = msg.drive.speed
        self.opp_steer = msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, msg):
        rx = msg.pose.pose.position.x; ry = msg.pose.pose.position.y
        _, _, rtheta = euler.quat2euler([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z], axes='sxyz')
        if self.has_opp:
            opp = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.obs, _, self.done, _ = self.env.reset(np.array([[rx, ry, rtheta], opp]))
        else:
            self.obs, _, self.done, _ = self.env.reset(np.array([[rx, ry, rtheta]]))
        self._update_sim_state()

    def opp_reset_callback(self, msg):
        if self.has_opp:
            rx = msg.pose.position.x; ry = msg.pose.position.y
            _, _, rtheta = euler.quat2euler([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z], axes='sxyz')
            self.obs, _, self.done, _ = self.env.reset(np.array([list(self.ego_pose), [rx, ry, rtheta]]))
            self._update_sim_state()

    def teleop_callback(self, msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True
        self.ego_requested_speed = msg.linear.x
        self.ego_steer = 0.3 if msg.angular.z > 0.0 else (-0.3 if msg.angular.z < 0.0 else 0.0)

    def drive_timer_callback(self):
        if self.ego_drive_published and not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        if self.done:
            self.get_logger().warn("Done! Auto-resetting.")
            sx = self.get_parameter('sx').value; sy = self.get_parameter('sy').value; stheta = self.get_parameter('stheta').value
            if self.has_opp:
                sx1 = self.get_parameter('sx1').value; sy1 = self.get_parameter('sy1').value; stheta1 = self.get_parameter('stheta1').value
                self.obs, _, self.done, _ = self.env.reset(np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            else:
                self.obs, _, self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
        
        # 内部状態の更新のみを行い、タイムスタンプ取得は各配信用のタイマーに委ねる
        self._update_sim_state()

    def scan_timer_callback(self):
        ts = self.get_clock().now().to_msg()
        scan = LaserScan()
        scan.header.stamp = ts; scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min; scan.angle_max = self.angle_max; scan.angle_increment = self.angle_inc
        scan.range_min = 0.; scan.range_max = 30.; scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)
        
        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts; opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min; opp_scan.angle_max = self.angle_max; opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.; opp_scan.range_max = 30.; opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

    def odom_timer_callback(self):
        ts = self.get_clock().now().to_msg()
        self._publish_odom(ts)
        self._publish_transforms(ts)
        self._publish_wheel_transforms(ts)

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        self.ego_pose = [self.obs['poses_x'][0], self.obs['poses_y'][0], self.obs['poses_theta'][0]]
        self.ego_speed = [self.obs['linear_vels_x'][0], self.obs['linear_vels_y'][0], self.obs['ang_vels_z'][0]]
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.opp_speed = [self.obs['linear_vels_x'][1], self.obs['linear_vels_y'][1], self.obs['ang_vels_z'][1]]

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts; ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = float(self.ego_pose[0])
        ego_odom.pose.pose.position.y = float(self.ego_pose[1])
        q = euler.euler2quat(0., 0., float(self.ego_pose[2]), axes='sxyz')
        ego_odom.pose.pose.orientation.w = float(q[0]); ego_odom.pose.pose.orientation.x = float(q[1])
        ego_odom.pose.pose.orientation.y = float(q[2]); ego_odom.pose.pose.orientation.z = float(q[3])
        ego_odom.twist.twist.linear.x = float(self.ego_speed[0])
        ego_odom.twist.twist.linear.y = float(self.ego_speed[1])
        ego_odom.twist.twist.angular.z = float(self.ego_speed[2])
        self.ego_odom_pub.publish(ego_odom)
        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts; opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = float(self.opp_pose[0])
            opp_odom.pose.pose.position.y = float(self.opp_pose[1])
            q2 = euler.euler2quat(0., 0., float(self.opp_pose[2]), axes='sxyz')
            opp_odom.pose.pose.orientation.w = float(q2[0]); opp_odom.pose.pose.orientation.x = float(q2[1])
            opp_odom.pose.pose.orientation.y = float(q2[2]); opp_odom.pose.pose.orientation.z = float(q2[3])
            opp_odom.twist.twist.linear.x = float(self.opp_speed[0])
            opp_odom.twist.twist.linear.y = float(self.opp_speed[1])
            opp_odom.twist.twist.angular.z = float(self.opp_speed[2])
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_ts = TransformStamped()
        ego_ts.header.stamp = ts; ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        ego_ts.transform.translation.x = float(self.ego_pose[0])
        ego_ts.transform.translation.y = float(self.ego_pose[1])
        q = euler.euler2quat(0., 0., float(self.ego_pose[2]), axes='sxyz')
        ego_ts.transform.rotation.w = float(q[0]); ego_ts.transform.rotation.x = float(q[1])
        ego_ts.transform.rotation.y = float(q[2]); ego_ts.transform.rotation.z = float(q[3])
        self.br.sendTransform(ego_ts)
        if self.has_opp:
            opp_ts = TransformStamped()
            opp_ts.header.stamp = ts; opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            opp_ts.transform.translation.x = float(self.opp_pose[0])
            opp_ts.transform.translation.y = float(self.opp_pose[1])
            q2 = euler.euler2quat(0., 0., float(self.opp_pose[2]), axes='sxyz')
            opp_ts.transform.rotation.w = float(q2[0]); opp_ts.transform.rotation.x = float(q2[1])
            opp_ts.transform.rotation.y = float(q2[2]); opp_ts.transform.rotation.z = float(q2[3])
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wt = TransformStamped()
        q = euler.euler2quat(0., 0., float(self.ego_steer), axes='sxyz')
        ego_wt.transform.rotation.w = float(q[0]); ego_wt.transform.rotation.x = float(q[1])
        ego_wt.transform.rotation.y = float(q[2]); ego_wt.transform.rotation.z = float(q[3])
        ego_wt.header.stamp = ts
        ego_wt.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wt.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wt)
        ego_wt.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wt.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wt)
        if self.has_opp:
            opp_wt = TransformStamped()
            q2 = euler.euler2quat(0., 0., float(self.opp_steer), axes='sxyz')
            opp_wt.transform.rotation.w = float(q2[0]); opp_wt.transform.rotation.x = float(q2[1])
            opp_wt.transform.rotation.y = float(q2[2]); opp_wt.transform.rotation.z = float(q2[3])
            opp_wt.header.stamp = ts
            opp_wt.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wt.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wt)
            opp_wt.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wt.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wt)


def main(args=None):
    rclpy.init(args=args)
    bridge = JaxGymBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    try:
        executor.spin()
    except KeyboardInterrupt:
        bridge.get_logger().info('Exiting')
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()