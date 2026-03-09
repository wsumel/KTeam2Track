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


def expand_boxes(boxes, expand=0.0, expand_pixels=0.0):
    """按中心对 boxes 放大/缩小。

    boxes: (N,4) 或 (4,) 数组，格式为 [x,y,w,h]（x,y 假定为左上角）。
    - expand: 相对放大比例，例如 0.05 表示宽高扩大 5%，-0.1 表示缩小 10%。
    - expand_pixels: 每边扩展像素数（per-side pixels）。若非 0，则优先使用像素扩展；

    行为：对于每个非缺失框，计算中心 (cx,cy)，并按参数计算新的宽高与左上角。
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
        if float(expand_pixels) != 0.0:
            # expand_pixels 表示每边扩展的像素数，因此宽高应增加 2*expand_pixels
            new_w = w + 2.0 * float(expand_pixels)
            new_h = h + 2.0 * float(expand_pixels)
        else:
            new_w = w * (1.0 + float(expand))
            new_h = h * (1.0 + float(expand))
        new_x = cx - new_w / 2.0
        new_y = cy - new_h / 2.0
        out[i] = [new_x, new_y, new_w, new_h]

    if single:
        return out.reshape(4,)
    return out


def fill_from_secondary(primary, secondary, eps=-1.0):
    """用 secondary 中非缺失帧填补 primary 中缺失帧（按行）。

    - primary, secondary: (T,4) 数组
    - 若形状不一致则返回 primary 原样（并打印警告）
    """
    p = np.asarray(primary, dtype=float).copy()
    s = np.asarray(secondary, dtype=float)
    if p.shape != s.shape:
        print(f"警告：primary 与 secondary 形状不一致：{p.shape} vs {s.shape}，跳过从 secondary 填补")
        return p

    T = p.shape[0]
    for i in range(T):
        if is_missing(p[i]) and not is_missing(s[i]):
            p[i] = s[i]
    return p


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


def forward_fill_series(data, eps=-1.0):
    """用前一帧的值向前填充缺失帧（leading 缺失保持为 eps）。

    例如： [A, -1, -1, B, -1] -> [A, A, A, B, B]
    如果序列以缺失开始，则这些前导位置保持 eps（默认 -1）。
    """
    data = np.asarray(data, dtype=float).copy()
    T = data.shape[0]
    last = None
    for i in range(T):
        if not is_missing(data[i]):
            last = data[i].copy()
        else:
            if last is not None:
                data[i] = last
            else:
                # 前导缺失保留 eps
                data[i] = np.full(4, eps, dtype=float)
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


def fill_with_linear(data, min_neighbors=1, num_input_frames=5, pred_horizon=0, eps=-1.0, use_last=False):
    """对数据进行线性插值 + 结尾线性外推填补。

    - 首先做中间段的线性插值（需要前后至少各 min_neighbors 帧）。
    - 然后对于结尾若存在缺失且 pred_horizon>0，可用外推补足 pred_horizon 步；
      若 pred_horizon==0，则仅插值，中间段外的缺失保持不变（或按需求用外推覆盖）。

    返回：填补后的数组（若提供 pred_horizon 并希望保存额外预测，请用 process_folder 的 predict 模式）。
    """
    print(use_last)
    data = np.asarray(data, dtype=float).copy()
    if use_last:

        # 使用上一帧值填充（不做线性插值）
        filled = forward_fill_series(data, eps=eps)
    else:
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


def process_folder(folder, folder2=None, mode="fill", min_neighbors=1, num_input_frames=5, pred_horizon=10, eps=-1.0, expand=0.0, expand_pixels=0.0, use_last=False, outputdir=None):
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    print(f"发现 {len(files)} 个文件: 模式={mode}")

    # 计算输出基目录：若提供 outputdir 则使用之，否则使用 folder.parent
    base_out = Path(outputdir) if (outputdir is not None and str(outputdir) != "") else folder.parent

    if mode in ("fill", "both"):
        save_folder = base_out #/ (folder.name + "_linear")
        save_folder.mkdir(parents=True, exist_ok=True)

    if mode in ("predict", "both"):
        save_pred_folder = base_out / (folder.name + f"_pred_lin_in{num_input_frames}_out{pred_horizon}")
        save_pred_folder.mkdir(parents=True, exist_ok=True)

    # 如果提供了 folder2，则准备 Path
    sec_folder = None
    if folder2 is not None:
        sec_folder = Path(folder2)
        if not sec_folder.exists():
            print(f"警告：提供的 folder2 不存在：{sec_folder}，将仅使用 folder1 进行插值")
            sec_folder = None

    for file in files:
        print(f"处理 {file.name}")
        data = np.loadtxt(file)
        # 若提供了 folder2，尝试用 secondary 的非缺失值先填补 primary 的 -1
        if sec_folder is not None:
            sec_file = sec_folder / file.name
            if sec_file.exists():
                sec_data = np.loadtxt(sec_file)
                data = fill_from_secondary(data, sec_data, eps=eps)
            else:
                print(f"注意：在 folder2 中未找到对应文件 {file.name}，跳过从 folder2 填补")

        if mode in ("fill", "both"):
            filled = fill_with_linear(data, min_neighbors=min_neighbors, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps, use_last=use_last)
            if float(expand) != 0.0 or float(expand_pixels) != 0.0:
                filled = expand_boxes(filled, expand=expand, expand_pixels=expand_pixels)
            save_path = save_folder / file.name
            np.savetxt(save_path, filled, fmt="%.6f")

        if mode in ("predict", "both"):
            preds = predict_future_from_past(data, num_input_frames=num_input_frames, pred_horizon=pred_horizon, eps=eps)
            if float(expand) != 0.0 or float(expand_pixels) != 0.0:
                preds = expand_boxes(preds, expand=expand, expand_pixels=expand_pixels)
            save_path = save_pred_folder / (file.stem + f"_pred_lin.txt")
            np.savetxt(save_path, preds, fmt="%.6f")

    print("处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linear.py：线性插值/外推填补与预测")
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--folder2", type=str, default='', help="可选：用于先填补 primary 的第二目录（会优先用 folder2 非 -1 值填补 primary 的 -1）")
    parser.add_argument("--mode", choices=("fill", "predict", "both"), default="fill")
    parser.add_argument("--min-neighbors", type=int, default=1, help="两侧至少各需多少帧才做线性插值，默认1")
    parser.add_argument("--num-input-frames", type=int, default=3, help="用于估计速度的前序有效帧数（外推使用）")
    parser.add_argument("--pred-horizon", type=int, default=1, help="要预测的未来帧数")
    parser.add_argument("--eps", type=float, default=-1.0, help="缺失占位值")
    parser.add_argument("--expand", type=float, default=0.0, help="按中心放大/缩小 bbox 的相对比例，例如 0.05=扩大5%%，-0.1=缩小10%%")
    parser.add_argument("--expand-px", dest="expand_px", type=float, default=1.5, help="按中心每边扩展多少像素（像素优先于相对比例）")
    parser.add_argument("--use-last", dest="use_last", default=True, help="启用后对缺失帧使用上一帧结果填充（不做线性插值）")
    parser.add_argument("--outputdir", dest="outputdir", type=str, default='', help="可选：指定输出目录，若提供则在该目录下创建输出子目录")

    args = parser.parse_args()

    process_folder(args.folder, folder2=args.folder2, mode=args.mode, min_neighbors=args.min_neighbors, num_input_frames=args.num_input_frames, pred_horizon=args.pred_horizon, eps=args.eps, expand=args.expand, expand_pixels=args.expand_px, use_last=args.use_last, outputdir=args.outputdir)
