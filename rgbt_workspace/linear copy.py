"""
linear.py

功能：
- 使用线性插值（当前帧前后有可用观测时）填补缺失框；
- 当序列尾部缺失时使用线性外推（基于前若干有效帧的平均速度）进行填补/预测；
- 提供与 klm2.py 类似的命令行接口：mode(fill/predict/both)、num-input-frames、pred-horizon 等。

超参数说明（代码中亦有注释）：
- num_input_frames: 用于估计速度的前序有效帧数（用于外推/预测），建议 3~10；
- pred_horizon: 需要预测的未来帧数；
- min_neighbors: 插值时前后至少各需多少帧（通常为1，即标准线性插值）；
- eps: 在无有效帧时返回的占位值（默认 -1）。

缺失表示：一帧若所有元素均为 -1 则视为缺失。

用法示例：
python linear.py /path/to/folder --mode both --num-input-frames 5 --pred-horizon 10
"""

import argparse
from pathlib import Path
import numpy as np


def is_missing(bbox):
    return np.all(np.asarray(bbox) == -1)


def expand_boxes(boxes, expand=0.0):
    """按中心对 boxes 放大/缩小。

    boxes: (N,4) 或 (4,) 数组，格式为 [x,y,w,h]（x,y 假定为左上角）。
    expand: 相对放大比例，例如 0.05 表示宽高扩大 5%，-0.1 表示缩小 10%。

    行为：对于每个非缺失框，计算中心 (cx,cy)，新的宽高 = w*(1+expand)、h*(1+expand)，
    并返回新的 [x,y,w,h]，使得框以中心为锚点放缩。
    缺失框（全 -1）保持不变。
    """
    arr = np.asarray(boxes, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
        single = True

    out = arr.copy()
    for i in range(out.shape[0]):
        b = out[i]
        if is_missing(b):
            continue
        x, y, w, h = b
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = w * (1.0 + float(expand))
        new_h = h * (1.0 + float(expand))
        new_x = cx - new_w / 2.0
        new_y = cy - new_h / 2.0
        out[i] = [new_x, new_y, new_w, new_h]

    if single:
        return out.reshape(4,)
    return out


def linear_interpolate_series(data, min_neighbors=1, eps=-1.0):
    """对序列中间缺失段做线性插值；对于开头/结尾不足两侧观测的段不处理（保留原值）。

    参数：
    - data: (T,4) 数组
    - min_neighbors: 两侧至少各需多少帧才做插值（默认1）
    - eps: 缺失填充值标记（默认 -1）
    返回：含插值后的数组
    """
    data = np.asarray(data, dtype=float).copy()
    T = data.shape[0]

    # 找到所有有效索引
    valid_indices = [i for i in range(T) if not is_missing(data[i])]
    if len(valid_indices) == 0:
        return np.full_like(data, eps, dtype=float)

    # 遍历相邻有效点对并对它们之间的缺失做线性插值
    for i in range(len(valid_indices) - 1):
        a = valid_indices[i]
        b = valid_indices[i + 1]
        gap = b - a - 1
        if gap <= 0:
            continue
        # 只在两侧满足 min_neighbors 时插值（通常 min_neighbors==1）
        if a >= 0 and b < T and True:
            # 对每个维度做线性插值
            start = data[a]
            end = data[b]
            for k in range(1, gap + 1):
                alpha = k / (gap + 1)
                data[a + k] = (1 - alpha) * start + alpha * end

    return data


def extrapolate_by_linear_velocity(data, num_input_frames=5, pred_horizon=10, eps=-1.0):
    """对序列结尾的缺失或做未来预测使用线性外推：
    - 从序列中选取最后 num_input_frames 个有效帧，计算相邻差的平均速度 v（长度4），
    - 从最后一帧开始按 v * step 进行线性外推，返回 pred_horizon 步的预测。

    返回 shape (pred_horizon,4) 的数组，若没有任何有效帧则返回全 eps。
    """
    data = np.asarray(data, dtype=float)
    T = len(data)
    valid_idx = [i for i in range(T) if not is_missing(data[i])]
    if len(valid_idx) == 0:
        return np.full((pred_horizon, 4), eps, dtype=float)

    # 选取最后若干有效帧
    use_idx = valid_idx[-max(1, int(num_input_frames)):]
    frames = data[use_idx]

    if frames.shape[0] == 1:
        velocity = np.zeros(4, dtype=float)
    else:
        diffs = np.diff(frames, axis=0)
        velocity = np.mean(diffs, axis=0)

    last = frames[-1]
    preds = []
    cur = last.copy()
    for step in range(1, int(pred_horizon) + 1):
        cur = cur + velocity
        preds.append(cur.copy())
    return np.stack(preds, axis=0)


def fill_with_linear(data, min_neighbors=1, num_input_frames=5, pred_horizon=0, eps=-1.0):
    """对数据进行线性插值 + 结尾线性外推填补。

    - 首先做中间段的线性插值（需要前后至少各 min_neighbors 帧）。
    - 然后对于结尾若存在缺失且 pred_horizon>0，可用外推补足 pred_horizon 步；
      若 pred_horizon==0，则仅插值，中间段外的缺失保持不变（或按需求用外推覆盖）。

    返回：填补后的数组（若提供 pred_horizon 并希望保存额外预测，请用 process_folder 的 predict 模式）。
    """
    data = np.asarray(data, dtype=float).copy()
    filled = linear_interpolate_series(data, min_neighbors=min_neighbors, eps=eps)

    # 若序列最后仍有缺失且需要外推/预测，则用外推填充
    if pred_horizon > 0:
        # 若序列尾部有缺失，从最后一个有效帧开始外推并覆盖缺失位置（最多 pred_horizon 或直到填满）
        T = len(filled)
        valid_idx = [i for i in range(T) if not is_missing(filled[i])]
        if len(valid_idx) == 0:
            return filled
        last_valid = valid_idx[-1]
        # 外推序列
        preds = extrapolate_by_linear_velocity(filled, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps)
        # 覆盖从 last_valid+1 开始的位置
        for i, p in enumerate(preds):
            idx = last_valid + 1 + i
            if idx >= T:
                break
            # 仅覆盖原本缺失的位置
            if is_missing(data[idx]):
                filled[idx] = p

    return filled


def predict_future_from_past(data, num_input_frames=5, pred_horizon=10, eps=-1.0):
    """基于前 num_input_frames 个有效帧做线性外推预测未来 pred_horizon 帧（接口与 klm2.py 保持一致）。"""
    return extrapolate_by_linear_velocity(data, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps)


def process_folder(folder, mode="fill", min_neighbors=1, num_input_frames=5, pred_horizon=10, eps=-1.0, expand=0.0):
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    print(f"发现 {len(files)} 个文件: 模式={mode}")

    if mode in ("fill", "both"):
        save_folder = folder.parent / (folder.name + "_linear")
        save_folder.mkdir(exist_ok=True)

    if mode in ("predict", "both"):
        save_pred_folder = folder.parent / (folder.name + f"_pred_lin_in{num_input_frames}_out{pred_horizon}")
        save_pred_folder.mkdir(exist_ok=True)

    for file in files:
        print(f"处理 {file.name}")
        data = np.loadtxt(file)

        if mode in ("fill", "both"):
            filled = fill_with_linear(data, min_neighbors=min_neighbors, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps)
            if float(expand) != 0.0:
                # 对所有非缺失框按中心放大/缩小
                filled = expand_boxes(filled, expand=expand)
            save_path = save_folder / file.name
            np.savetxt(save_path, filled, fmt="%.6f")

        if mode in ("predict", "both"):
            preds = predict_future_from_past(data, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps)
            if float(expand) != 0.0:
                preds = expand_boxes(preds, expand=expand)
            save_path = save_pred_folder / (file.stem + f"_pred_lin.txt")
            np.savetxt(save_path, preds, fmt="%.6f")

    print("处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linear.py：线性插值/外推填补与预测")
    parser.add_argument("--folder", type=str, default="/disk3/wsl_tmp/Workspace210/MDTrack/results/LasHeR/mdtrack_b224_lasher_17_2")
    parser.add_argument("--mode", choices=("fill", "predict", "both"), default="fill")
    parser.add_argument("--min-neighbors", type=int, default=1, help="两侧至少各需多少帧才做线性插值，默认1")
    parser.add_argument("--num-input-frames", type=int, default=3, help="用于估计速度的前序有效帧数（外推使用）")
    parser.add_argument("--pred-horizon", type=int, default=1, help="要预测的未来帧数")
    parser.add_argument("--eps", type=float, default=-1.0, help="缺失占位值")
    parser.add_argument("--expand", type=float, default=0.01, help="按中心放大/缩小 bbox 的相对比例，例如 0.05=扩大5%%，-0.1=缩小10%%")

    args = parser.parse_args()

    process_folder(args.folder, mode=args.mode, min_neighbors=args.min_neighbors, num_input_frames=args.num_input_frames, pred_horizon=args.pred_horizon, eps=args.eps, expand=args.expand)
