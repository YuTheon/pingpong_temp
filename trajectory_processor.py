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
        """核心算法逻辑：预测去噪 -> 动态滤波 -> 速度分析 -> 落点检测"""
        pos = np.array(raw_pos)
        self.frame_count += 1
        
        # 1. 计算步长并检查数据是否属于新序列（断流重置逻辑）
        is_new_session = False
        if self.last_valid_time > 0:
            dt = timestamp - self.last_valid_time
            # 如果断流超过阈值（0.5s），视为新球出现，不进行去噪判断
            if dt > self.timeout_threshold:
                is_new_session = True
                dt = 0.033  # 给定默认步长
        else:
            dt = 0.033
            is_new_session = True

        # 2. 运动模型预测与去噪 (仅在非新回合时执行)
        if not is_new_session and self.last_valid_pos is not None:
            # 根据上一帧位置和速度预测当前位置
            predicted_pos = self.last_valid_pos + self.velocity * dt
            actual_dist = np.linalg.norm(pos - predicted_pos)
            
            # 如果实际检测点偏离预测轨迹过远（默认800mm），判定为噪点并拦截
            # 建议：如果乒乓球速度极快，可将 self.max_jump_distance 设为 1200-1500
            if actual_dist > self.max_jump_distance:
                return None, 0, {"frame_count": self.frame_count, "timeout": False}

        # 3. 状态重置：如果是新回合，清空滤波器历史状态，防止产生巨大的相位拉扯
        if is_new_session:
            self.velocity = np.zeros(3)
            self.one_euro_filter.last_timestamp = None  # 强制 OneEuroFilter 重新初始化
            self.one_euro_filter.x_filter.s = None
            self.one_euro_filter.dx_filter.s = None

        # 4. 动态滤波平滑
        filtered_pos = self.one_euro_filter.filter(pos, timestamp)
        
        # 5. 更新速度矢量（用于下一帧预测和UI显示）
        speed = 0
        if self.last_valid_pos is not None and dt > 0:
            current_v = (filtered_pos - self.last_valid_pos) / dt
            if is_new_session:
                # 新序列第一帧，直接采用当前速度
                self.velocity = current_v
            else:
                # 非新序列，对速度矢量进行平滑，避免预测轨迹抖动
                self.velocity = 0.8 * current_v + 0.2 * self.velocity
            speed = np.linalg.norm(self.velocity)
        
        # 6. 速度趋势分析（用于记录击球瞬间）
        y_trend_changed = False
        if self.prev_pos is not None and self.prev_time is not None:
            _, y_trend_changed, _ = self.recorder.analyze_speed_and_trend(
                filtered_pos, self.prev_pos, timestamp, self.prev_time
            )

        # 7. 落点检测 (仅在低高度触发分析)
        landing_detected = False
        if filtered_pos[2] < 80:
            landing_detected = self.landing_analyzer.analyze_realtime_landing(filtered_pos, timestamp)

        # 8. 更新历史状态
        self.prev_pos = filtered_pos.copy()
        self.prev_time = timestamp
        self.last_valid_pos = filtered_pos.copy()
        self.last_valid_time = timestamp

        events = {
            "landing_detected": landing_detected,
            "y_trend_changed": y_trend_changed,
            "shot_count": self.recorder.get_shot_count(),
            "frame_count": self.frame_count,
            "timeout": False,
            "is_new_session": is_new_session  # 新增标志位，方便调试
        }
        return filtered_pos, speed, events
    
    def reset(self):
        self.last_valid_pos = None
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0
        self.landing_analyzer.reset_landing_analysis()
