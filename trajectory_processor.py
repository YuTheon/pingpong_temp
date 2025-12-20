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
        if alpha is not None: self.alpha = alpha
        if self.s is None: self.s = value
        else: self.s = self.alpha * value + (1.0 - self.alpha) * self.s
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
        self.one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.15, d_cutoff=1.5)
        self.interpolator = TrajectoryInterpolator()
        self.landing_analyzer = LandingAnalyzer(save_folder_path)
        self.recorder = TrajectoryRecorder(save_folder_path, use_simulator_format=True)
        self.last_valid_pos = None
        self.max_jump_distance = 450.0 
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0

    def process_realtime_step(self, raw_pos, timestamp):
        """核心算法逻辑：去噪 -> 滤波 -> 速度分析 -> 落点检测"""
        pos = np.array(raw_pos)
        self.frame_count += 1

        # 1. 距离去噪
        if self.last_valid_pos is not None:
            if np.linalg.norm(pos - self.last_valid_pos) > self.max_jump_distance:
                return None, 0, {"frame_count": self.frame_count}

        # 2. 平滑滤波
        filtered_pos = self.one_euro_filter.filter(pos, timestamp)
        
        # 3. 速度与趋势
        speed = 0.0
        y_trend_changed = False
        if self.prev_pos is not None and self.prev_time is not None:
            speed, y_trend_changed, _ = self.recorder.analyze_speed_and_trend(
                filtered_pos, self.prev_pos, timestamp, self.prev_time
            )

        # 4. 落点检测
        landing_detected = False
        if filtered_pos[2] < 80:
            landing_detected = self.landing_analyzer.analyze_realtime_landing(filtered_pos, timestamp)

        res_pos = filtered_pos.copy()
        self.prev_pos = filtered_pos.copy()
        self.prev_time = timestamp
        self.last_valid_pos = filtered_pos.copy()

        events = {
            "landing_detected": landing_detected,
            "y_trend_changed": y_trend_changed,
            "shot_count": self.recorder.get_shot_count(),
            "frame_count": self.frame_count
        }
        return res_pos, speed, events

    def reset(self):
        self.last_valid_pos = None
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0
        self.landing_analyzer.reset_landing_analysis()
