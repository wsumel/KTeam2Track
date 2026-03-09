import numpy as np
from pathlib import Path


class KalmanFilterBBox:

    def __init__(self):

        dt = 1.0

        # 状态转移矩阵
        self.F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ])

        # 观测矩阵
        self.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ])

        # 协方差
        self.P = np.eye(8) * 10

        # 过程噪声
        self.Q = np.eye(8) * 0.01

        # 观测噪声
        self.R = np.eye(4) * 1

        self.x = None


    def init(self, bbox):

        self.x = np.zeros(8)
        self.x[:4] = bbox


    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:4]


    def update(self, bbox):

        z = bbox

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P


def fill_with_kalman(data):

    kf = KalmanFilterBBox()

    filled = data.copy()

    initialized = False

    for i in range(len(data)):

        bbox = data[i]

        if not initialized and not np.all(bbox == -1):

            kf.init(bbox)
            filled[i] = bbox
            initialized = True
            continue

        if not initialized:
            continue

        pred = kf.predict()

        if np.all(bbox == -1):

            filled[i] = pred

        else:

            kf.update(bbox)
            filled[i] = bbox

    return filled


def process_folder(folder):

    folder = Path(folder)

    save_folder = folder.parent / (folder.name + "_kalman")
    save_folder.mkdir(exist_ok=True)

    files = sorted(folder.glob("*.txt"))

    print(f"发现 {len(files)} 个文件")

    for file in files:

        print(f"处理 {file.name}")

        data = np.loadtxt(file)

        filled = fill_with_kalman(data)

        save_path = save_folder / file.name

        np.savetxt(
            save_path,
            filled,
            fmt="%.6f"
        )

    print("Kalman 填补完成")


if __name__ == "__main__":

    process_folder("/disk3/wsl_tmp/Workspace210/MDTrack/results/LasHeR/mdtrack_b384_lasher_12_2/")
