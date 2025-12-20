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
    def __init__(self, min_cutoff=1.5, beta=0.15, d_cutoff=1.5):
        """
        beta: 灵敏度。增大该值可减少高速运动时的相位延迟。
        d_cutoff: 衍生截止频率。增加可过滤更剧烈的速度抖动。
        """
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
        # 计算当前速度 (衍生量)
        dx = (x - self.x_filter.s) / dt
        edx = self.dx_filter.filter(dx, alpha=self.compute_alpha(self.d_cutoff, dt))
        
        # 动态计算 alpha：速度越快，截止频率越高，滤波越轻（响应更实时）
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        return self.x_filter.filter(x, alpha=self.compute_alpha(cutoff, dt))

class TrajectoryProcessor:
    def __init__(self, save_folder_path=None):
        # 优化后的滤波器：增强对普通杀球速度的响应
        self.one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.15, d_cutoff=1.5)
        self.interpolator = TrajectoryInterpolator()
        self.landing_analyzer = LandingAnalyzer(save_folder_path)
        self.recorder = TrajectoryRecorder(save_folder_path, use_simulator_format=True)
        
        # 核心算法状态
        self.last_valid_pos = None
        self.last_valid_time = 0
        self.velocity = np.zeros(3)      # 3D 速度矢量 (mm/s)
        self.max_jump_distance = 800.0   # 增大阈值以容纳普通杀球
        self.timeout_threshold = 0.5     # 0.5秒未收到信号则认为球消失
        
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0

    def process_realtime_step(self, raw_pos, timestamp):
        """核心算法逻辑：预测去噪 -> 动态滤波 -> 速度分析 -> 落点检测"""
        pos = np.array(raw_pos)
        self.frame_count += 1
        
        # 计算步长
        dt = timestamp - self.last_valid_time if self.last_valid_time > 0 else 0.033

        # 1. 运动模型预测与去噪
        if self.last_valid_pos is not None:
            # 预测当前位置 = 上一有效位置 + 速度 * 时间
            predicted_pos = self.last_valid_pos + self.velocity * dt
            actual_dist = np.linalg.norm(pos - predicted_pos)
            
            # 如果实际检测点偏离预测轨迹过远（如 > 800mm），判定为噪点
            if actual_dist > self.max_jump_distance:
                return None, 0, {"frame_count": self.frame_count, "timeout": False}

        # 2. 动态滤波平滑
        filtered_pos = self.one_euro_filter.filter(pos, timestamp)
        
        # 3. 更新速度矢量（用于下一帧预测和UI显示）
        if self.last_valid_pos is not None and dt > 0:
            current_v = (filtered_pos - self.last_valid_pos) / dt
            # 对速度矢量进行轻微平滑，避免预测轨迹抖动
            self.velocity = 0.8 * current_v + 0.2 * self.velocity
        
        # 4. 速度标量与趋势分析（用于记录和落点检测）
        speed = np.linalg.norm(self.velocity)
        y_trend_changed = False
        if self.prev_pos is not None and self.prev_time is not None:
            _, y_trend_changed, _ = self.recorder.analyze_speed_and_trend(
                filtered_pos, self.prev_pos, timestamp, self.prev_time
            )

        # 5. 落点检测 (仅在低高度触发分析)
        landing_detected = False
        if filtered_pos[2] < 80:
            landing_detected = self.landing_analyzer.analyze_realtime_landing(filtered_pos, timestamp)

        # 更新历史状态
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

    def check_timeout(self, current_time):
        """
        检查数据是否断流。
        如果返回 True，调用者应隐藏或移除渲染界面上的球。
        """
        if self.last_valid_time == 0: 
            return True
        return (current_time - self.last_valid_time) > self.timeout_threshold

    def reset(self):
        self.last_valid_pos = None
        self.last_valid_time = 0
        self.velocity = np.zeros(3)
        self.prev_pos = None
        self.prev_time = None
        self.frame_count = 0
        self.landing_analyzer.reset_landing_analysis()