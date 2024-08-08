import cv2
import numpy as np
import os
import json
import yaml


config = yaml.safe_load(open("config.yaml"))
work_dir = config['work_dir']
video_filename = os.path.join(work_dir, "video_recording.mp4")
output_frame_dir = os.path.join(work_dir, "input")

os.makedirs(output_frame_dir, exist_ok=True)


cap = cv2.VideoCapture(video_filename)
assert cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ", fps)


fi = 0
skip = int(config["video_skip"])
keep = int(config["video_keep"])

iqms = []
frames_buf = []
frames = []

frames_info = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.bilateralFilter(frame, 5, 40, 40)

    s = 256
    gray = cv2.cvtColor(cv2.resize(frame, (2*s, 2*s)), cv2.COLOR_BGR2GRAY)
    mu, sigma = np.mean(gray), np.std(gray)
    gray = (gray-mu)#/sigma

    iqm = cv2.Laplacian(gray, cv2.CV_64F).var()

    time = (fi+0.5) / fps
    iqms.append((time, mu, sigma, iqm))

    frames_buf.append((time, iqm, frame))
    if len(frames_buf) > keep:
        del frames_buf[0]

    fi += 1
    if fi % skip == 0:
        frame = sorted(frames_buf, key=lambda x: x[1], reverse=True)[0]
        frames.append(frame)

        time = frame[0]
        frame = frame[2]

        filename = 'frame_{:05d}.jpg'.format(fi//skip-1)
        cv2.imwrite(os.path.join(output_frame_dir, filename),
                    frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(filename)
        frames_info.append({
            "file_path": "images/" + filename,
            "time": time
        })

cap.release()

print(len(frames), '/', len(iqms), 'frames')

with open(os.path.join(work_dir, "timestamps.json"), "w") as fp:
    json.dump(frames_info, fp, indent=4)


import matplotlib.pyplot as plt

iqms = np.array(iqms)
times, iqms = iqms[:,0], iqms[:,1:]
iqms /= np.mean(iqms, axis=0)
plt.plot(times, iqms[:,0], label='mean')
plt.plot(times, iqms[:,1], label='std')
plt.plot(times, iqms[:,2], 'k', label='iqm')
plt.vlines([_[0] for _ in frames], np.min(iqms), np.max(iqms), 'm', '-')
plt.xlabel("time [s]")
plt.legend()
plt.grid()
plt.show()
