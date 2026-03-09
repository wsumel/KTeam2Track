"""
klm2.py

功能：
- 基于 8D 状态的简单卡尔曼滤波（状态为 [x,y,w,h,vx,vy,vw,vh]），
  实现对缺失检测框（用全 -1 表示）的逐帧填补（fill 模式）。
- 提供从序列前若干有效帧预测未来若干帧的功能（predict 模式），
  支持可调超参数并在源文件中给出明确中文注释说明含义与建议范围。

超参数说明（在代码中也有注释）：
- dt: 时间步长，表示相邻帧之间的时间间隔（默认 1.0）。
  建议：若帧率已知，可设置为帧间隔（秒），否则保持 1.0 即可。
  影响：影响速度积分到位置的预测，dt 较大则速度对位置影响更明显。

- Q_scale: 过程噪声尺度（过程协方差 = Q_scale * I）。
  含义：模型不确定性（系统噪声）的大小。
  建议范围：1e-4 ~ 1.0。较大值表示更不信任模型预测，滤波更依赖观测；较小值表示更信任模型动态。

- R_scale: 观测噪声尺度（观测协方差 = R_scale * I）。
  含义：测量噪声的大小（来自检测器的不确定性）。
  建议范围：1e-3 ~ 10.0。较大值会减弱观测对状态的影响，滤波更平滑但可能欠跟踪。

- P_init: 初始状态协方差尺度（P = P_init * I）。
  含义：滤波器初始时对状态的不确定性估计。
  建议：若初始位置可信可设小值（1~10），若不可信可设大值（100 或更大）。

预测相关参数：
- num_input_frames: 用于估计初始速度的有效前序帧数（例如只用前 5 帧）。
  含义：从序列头部（或最近可用的前序部分）挑选 num_input_frames 个非缺失帧用于估计速度。
  建议：3~10，根据场景稳定性调整。帧太少会导致速度估计不稳定。

- pred_horizon: 需要预测的未来帧数。返回形状 (pred_horizon, 4) 的 bbox 预测。

其他说明：
- 缺失检测框的表示：一帧若全部元素均为 -1 则视作缺失。
- 速度估计方法：对前序若干有效帧取相邻差分的平均值作为速度估计。

用法示例：
python klm2.py /path/to/folder --mode both --num-input-frames 5 --pred-horizon 10 --dt 1.0 --Q 0.01 --R 1.0 --P 10

"""

import argparse
from pathlib import Path
import numpy as np


class KalmanFilterBBox:
    """8D 卡尔曼滤波器：状态 x = [x,y,w,h,vx,vy,vw,vh]

    参数：
    - dt: 时间步长
    - Q_scale: 过程噪声尺度 -> Q = Q_scale * I(8)
    - R_scale: 观测噪声尺度 -> R = R_scale * I(4)
    - P_init: 初始协方差尺度 -> P = P_init * I(8)
    """

    def __init__(self, dt=1.0, Q_scale=0.01, R_scale=1.0, P_init=10.0):
        self.dt = float(dt)
        dt = self.dt

        # 状态转移矩阵：位置由位置 + 速度 * dt 更新
        self.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # 观测矩阵：我们只观测位置 (x,y,w,h)
        self.H = np.zeros((4, 8), dtype=float)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # 协方差矩阵
        self.P = np.eye(8, dtype=float) * float(P_init)

        # 过程噪声与观测噪声
        self.Q = np.eye(8, dtype=float) * float(Q_scale)
        self.R = np.eye(4, dtype=float) * float(R_scale)

        self.x = None

    def init(self, bbox, velocity=None):
        """用一个 bbox 初始化状态向量。

        bbox: 长度 4 的数组 [x,y,w,h]
        velocity: 可选长度 4 的速度向量 [vx,vy,vw,vh]，若不提供则默认 0。
        """
        self.x = np.zeros(8, dtype=float)
        self.x[:4] = bbox
        if velocity is not None:
            self.x[4:8] = velocity

    def predict_step(self):
        """前向一步预测并返回预测的位置部分（长度 4）。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def predict_n(self, n):
        """连续预测 n 步并返回 (n,4) 的位置序列。"""
        preds = []
        for _ in range(n):
            preds.append(self.predict_step())
        return np.stack(preds, axis=0)

    def update(self, bbox):
        """用观测 bbox 更新滤波器。bbox 长度为 4。"""
        z = np.asarray(bbox, dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8, dtype=float)
        self.P = (I - K @ self.H) @ self.P


def is_missing(bbox):
    """判定 bbox 是否缺失：如果所有元素均为 -1 则认为缺失。"""
    return np.all(np.asarray(bbox) == -1)


def fill_with_kalman(data, dt=1.0, Q_scale=0.01, R_scale=1.0, P_init=10.0):
    """遍历序列并用卡尔曼滤波填补缺失帧（与原始 klm.py 功能一致）。

    参数含义参见文件顶部注释。
    data: (T,4) 数组或可被转换为数组的对象。
    返回：填补后的 (T,4) 浮点数组。
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("data 必须为形状 (T,4) 的数组")

    kf = KalmanFilterBBox(dt=dt, Q_scale=Q_scale, R_scale=R_scale, P_init=P_init)
    filled = data.copy().astype(float)
    initialized = False

    for i in range(len(data)):
        bbox = data[i]
        if not initialized and not is_missing(bbox):
            # 用序列中第一个可见框初始化卡尔曼
            kf.init(bbox)
            filled[i] = bbox
            initialized = True
            continue

        if not initialized:
            # 尚未找到第一个可见框，无法初始化，跳过
            continue

        pred = kf.predict_step()

        if is_missing(bbox):
            # 若检测缺失，使用预测值填补
            filled[i] = pred
        else:
            # 否则用观测更新滤波器并保留观测
            kf.update(bbox)
            filled[i] = bbox

    return filled


def predict_future_from_past(data, num_input_frames=5, pred_horizon=10, dt=1.0, Q_scale=0.01, R_scale=1.0, P_init=10.0):
    """从序列前部的前若干有效帧估计速度并预测未来 pred_horizon 帧。

    细节：
    - 从 `data` 中挑选前 num_input_frames 个非缺失帧（若不足则使用所有可用帧）。
    - 速度估计：对相邻有效帧取差分并取平均，作为速度初始值。
    - 用最后一个有效帧和估计速度初始化卡尔曼滤波器，然后连续预测 pred_horizon 步。

    返回：浮点数组，形状 (pred_horizon, 4)。若序列中没有有效帧则返回全 -1 的数组。
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("data 必须为形状 (T,4) 的数组")

    valid_idx = [i for i in range(len(data)) if not is_missing(data[i])]
    if len(valid_idx) == 0:
        # 没有可用帧，无法预测
        return np.full((pred_horizon, 4), -1.0, dtype=float)

    # 选择前 num_input_frames 个有效帧
    use_idx = valid_idx[:max(1, int(num_input_frames))]
    frames = data[use_idx]

    # 估计速度：若只有一帧则速度为零；否则取相邻差的均值
    if frames.shape[0] == 1:
        velocity = np.zeros(4, dtype=float)
    else:
        diffs = np.diff(frames, axis=0)
        velocity = np.mean(diffs, axis=0)

    # 初始化卡尔曼并预测
    kf = KalmanFilterBBox(dt=dt, Q_scale=Q_scale, R_scale=R_scale, P_init=P_init)
    last = frames[-1]
    kf.init(last, velocity=velocity)

    preds = kf.predict_n(int(pred_horizon))
    return preds


def process_folder(folder, mode="fill", num_input_frames=5, pred_horizon=10, dt=1.0, Q_scale=0.01, R_scale=1.0, P_init=10.0):
    """对文件夹中每个 .txt 文件执行填补或预测或两者。

    - folder: 目标文件夹路径，包含若干以行列式格式保存的 bbox 文本文件（每行为 x y w h）。
    - mode: 'fill' | 'predict' | 'both'
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    print(f"发现 {len(files)} 个文件: 模式={mode}")

    if mode in ("fill", "both"):
        save_folder = folder.parent / (folder.name + "_kalman2")
        save_folder.mkdir(exist_ok=True)

    if mode in ("predict", "both"):
        save_pred_folder = folder.parent / (folder.name + f"_pred_in{num_input_frames}_out{pred_horizon}")
        save_pred_folder.mkdir(exist_ok=True)

    for file in files:
        print(f"处理 {file.name}")
        data = np.loadtxt(file)

        if mode in ("fill", "both"):
            filled = fill_with_kalman(data, dt=dt, Q_scale=Q_scale, R_scale=R_scale, P_init=P_init)
            save_path = save_folder / file.name
            np.savetxt(save_path, filled, fmt="%.6f")

        if mode in ("predict", "both"):
            preds = predict_future_from_past(data, num_input_frames=num_input_frames, pred_horizon=pred_horizon, dt=dt, Q_scale=Q_scale, R_scale=R_scale, P_init=P_init)
            save_path = save_pred_folder / (file.stem + f"_pred.txt")
            np.savetxt(save_path, preds, fmt="%.6f")

    print("处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="klm2.py：卡尔曼填补与基于前序帧的预测，超参数在脚本顶部有说明")
    parser.add_argument("--folder", type=str, default="/disk3/wsl_tmp/Workspace210/MDTrack/results/LasHeR/mdtrack_b224_lasher_30_2")
    parser.add_argument("--mode", choices=("fill", "predict", "both"), default="fill", help="fill: 只填补缺失；predict: 只做预测；both: 两者都做")
    parser.add_argument("--num-input-frames", type=int, default=5, help="用于估计速度的前序有效帧数（例如只用前5帧）")
    parser.add_argument("--pred-horizon", type=int, default=1, help="要预测的未来帧数")
    parser.add_argument("--dt", type=float, default=1.0, help="时间步长 dt")
    parser.add_argument("--Q", type=float, default=0.01, help="过程噪声尺度 Q_scale")
    parser.add_argument("--R", type=float, default=1.0, help="观测噪声尺度 R_scale")
    parser.add_argument("--P", type=float, default=10.0, help="初始协方差尺度 P_init")

    args = parser.parse_args()

    process_folder(args.folder, mode=args.mode, num_input_frames=args.num_input_frames, pred_horizon=args.pred_horizon, dt=args.dt, Q_scale=args.Q, R_scale=args.R, P_init=args.P)
