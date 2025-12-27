import numpy as np
import math
import time
from .interpolation import TrajectoryInterpolator
from .landing_analyzer import LandingAnalyzer
from .trajectory_recorder import TrajectoryRecorder

class LowPassFilter:
    def __init__(self, alpha, init_value=None):
        self.s = init_value
        self.alpha = alpha
        
    def filter(self, value, alpha=None):
        if alpha is not None: 
            self.alpha = alpha
        if self.s is None: 
            self.s = value
        else: 
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

class OneEuroFilter:
    def __init__(self, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(0)
        self.dx_filter = LowPassFilter(0)
        self.last_timestamp = None
        
    def compute_alpha(self, cutoff, dt):
        if dt <= 0: return 1.0
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
        
    def filter(self, x, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.x_filter.s = x
            self.dx_filter.s = np.zeros_like(x)
            return x
            
        dt = timestamp - self.last_timestamp
        if dt <= 0: return self.x_filter.s
        
        self.last_timestamp = timestamp
        dx = (x - self.x_filter.s) / dt
        edx = self.dx_filter.filter(dx, alpha=self.compute_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        return self.x_filter.filter(x, alpha=self.compute_alpha(cutoff, dt))

class TrajectoryProcessor:
    def __init__(self, save_folder_path=None):
        # 1. 滤波器初始化
        self.one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.15, d_cutoff=1.5)
        self.interpolator = TrajectoryInterpolator()
        self.landing_analyzer = LandingAnalyzer(save_folder_path)
        self.recorder = TrajectoryRecorder(save_folder_path, use_simulator_format=True)
        
        # 2. 核心状态变量 (必须在此全部初始化)
        self.last_valid_pos = None
        self.last_valid_time = 0.0      # <--- 确保这一行存在
        self.velocity = np.zeros(3)      # 3D 速度矢量
        self.max_jump_distance = 1500.0  # 增大阈值以容纳高速杀球
        self.timeout_threshold = 0.5     # 断流判定阈值
        
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0

        self.is_evaluating = False
        self.current_serve_buffer = []  # 用于存储当前发球的轨迹点序列

    def process_realtime_step(self, raw_pos, timestamp):
        """核心算法逻辑：断流检测 -> 预测去噪 -> 动态滤波 -> 状态更新"""
        pos = np.array(raw_pos)
        self.frame_count += 1
        
        # A. 计算时间步长并检查是否为断流后的“新回合”
        is_new_session = False
        if self.last_valid_time > 0:
            dt = timestamp - self.last_valid_time
            if dt > self.timeout_threshold:
                is_new_session = True
                dt = 0.033 
        else:
            dt = 0.033
            is_new_session = True

        # B. 运动模型去噪 (仅在连续追踪时生效)
        if not is_new_session and self.last_valid_pos is not None:
            predicted_pos = self.last_valid_pos + self.velocity * dt
            actual_dist = np.linalg.norm(pos - predicted_pos)
            
            # 如果偏离预测位置过远，判定为噪点
            if actual_dist > self.max_jump_distance:
                return None, 0, {"frame_count": self.frame_count, "timeout": False}

        # C. 状态重置：若是新回合，清空滤波器历史，防止产生错误的瞬时高位移
        if is_new_session:
            self.velocity = np.zeros(3)
            self.one_euro_filter.last_timestamp = None 
            self.one_euro_filter.x_filter.s = None
            self.one_euro_filter.dx_filter.s = None

        # D. 动态滤波平滑
        filtered_pos = self.one_euro_filter.filter(pos, timestamp)
        
        # E. 更新速度矢量
        speed = 0
        if self.last_valid_pos is not None and dt > 0:
            current_v = (filtered_pos - self.last_valid_pos) / dt
            if is_new_session:
                self.velocity = current_v
            else:
                self.velocity = 0.8 * current_v + 0.2 * self.velocity
            speed = np.linalg.norm(self.velocity)
        
        # F. 速度分析与落点检测
        y_trend_changed = False
        if self.prev_pos is not None and self.prev_time is not None:
            _, y_trend_changed, _ = self.recorder.analyze_speed_and_trend(
                filtered_pos, self.prev_pos, timestamp, self.prev_time
            )

        landing_detected = False
        if filtered_pos[2] < 80:
            landing_detected = self.landing_analyzer.analyze_realtime_landing(filtered_pos, timestamp)

        # G. 更新历史状态
        self.prev_pos = filtered_pos.copy()
        self.prev_time = timestamp
        self.last_valid_pos = filtered_pos.copy()
        self.last_valid_time = timestamp

        events = {
            "landing_detected": landing_detected,
            "y_trend_changed": y_trend_changed,
            "shot_count": self.recorder.get_shot_count(),
            "frame_count": self.frame_count,
            "timeout": False
        }
        return filtered_pos, speed, events

    def reset(self):
        """完全重置处理器状态"""
        self.last_valid_pos = None
        self.last_valid_time = 0.0
        self.velocity = np.zeros(3)
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0
        self.landing_analyzer.reset_landing_analysis()
    
    def start_serve_session(self):
        """开启发球采集"""
        self.is_evaluating = True
        self.current_serve_buffer = []
        print("🚀 发球监控已就绪...")

    def stop_serve_session(self):
        """结束采集并返回结果"""
        self.is_evaluating = False
        if not self.current_serve_buffer:
            return None
        
        result = self.analyze_current_serve()
        self.current_serve_buffer = []
        return result

    def analyze_current_serve(self):
        """分析缓冲区内的发球质量"""
        if len(self.current_serve_buffer) < 5:
            return None

        # 提取位置和时间
        positions = np.array([p['pos'] for p in self.current_serve_buffer])
        times = np.array([p['time'] for p in self.current_serve_buffer])
        
        # 计算特征
        # 1. 峰值速度 (m/s)
        deltas = np.diff(positions, axis=0) / 1000.0 # 转为米
        dts = np.diff(times)
        speeds = [np.linalg.norm(d)/dt for d, dt in zip(deltas, dts) if dt > 0]
        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0

        # 2. 轨迹弧度 (最高点高度)
        max_height = np.max(positions[:, 2])

        # 3. 落点 (最后一个有效点，或高度最低点)
        landing_point = positions[-1] 

        return {
            "max_speed": max_speed,
            "avg_speed": avg_speed,
            "max_height": max_height,
            "landing_x": landing_point[0],
            "landing_y": landing_point[1],
            "trajectory": positions.tolist(), # 用于回放
            "timestamp": times[0]
        }
    
    def get_serve_features(self, points):
        """分析整段发球轨迹的特征"""
        if len(points) < 5:
            return None
        
        pos_array = np.array([p['pos'] for p in points])
        time_array = np.array([p['time'] for p in points])
        
        # 1. 计算峰值速度 (m/s)
        dist = np.linalg.norm(np.diff(pos_array, axis=0), axis=1) / 1000.0
        dt = np.diff(time_array)
        dt[dt == 0] = 0.001 # 防止除零
        speeds = dist / dt
        max_speed = np.max(speeds)
        
        # 2. 轨迹最高点 (mm)
        peak_height = np.max(pos_array[:, 2])
        
        # 3. 最终落点 (评估精度)
        landing_pos = pos_array[-1]
        
        return {
            "max_speed": max_speed,
            "peak_height": peak_height,
            "landing_x": landing_pos[0],
            "landing_y": landing_pos[1],
            "duration": time_array[-1] - time_array[0]
        }