import cv2
import numpy as np
import os


video_filename = "/home/harry7557558/20240423_132951.mp4"
video_filename = "/media/harry7557558/OS/Archive/sfm_videos/20240428_174233.mp4"
video_filename = "/media/harry7557558/OS/Archive/sfm_videos/20230624_201539.mp4"

cap = cv2.VideoCapture(video_filename)
assert cap.isOpened()


# pit: 50, 40, 11 frames
# arena: 20, 15, 40 frames
# float: 12, 8, 30 frames
fi = 0
skip = 5
keep = 4

iqms = []
frames_buf = []
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.bilateralFilter(frame, 5, 40, 40)

    s = 256
    gray = cv2.cvtColor(cv2.resize(frame, (2*s, 2*s)), cv2.COLOR_BGR2GRAY)
    mu, sigma = np.mean(gray), np.std(gray)
    gray = (gray-mu)#/sigma

    # fft = np.fft.fft2(gray)[:s, :s]
    # fft = (fft * np.conjugate(fft)).real
    # iqm = np.sum(fft[5:, 5:]) / np.sum(fft)
    # iqm = np.mean(fft > (np.amax(fft)/1000))

    # plt.imshow(np.log(fft))
    # break

    iqm = cv2.Laplacian(gray, cv2.CV_64F).var()

    iqms.append((mu, sigma, iqm))

    frames_buf.append((fi, iqm, frame))
    if len(frames_buf) > keep:
        del frames_buf[0]

    fi += 1
    if fi % skip == 0:
        frame = sorted(frames_buf, key=lambda x: x[1], reverse=True)[0]
        frames.append(frame)

        frame = cv2.resize(frame[2], (960, 540))

        cv2.imwrite('temp_{:d}.jpg'.format(fi//skip-1), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if len(frames) >= 100:
        break

cap.release()

print(len(frames), '/', len(iqms), 'frames')



import matplotlib.pyplot as plt

iqms = np.array(iqms)
iqms /= np.mean(iqms, axis=0)
plt.plot(iqms[:,0], label='mean')
plt.plot(iqms[:,1], label='std')
plt.plot(iqms[:,2], 'k', label='iqm')

plt.vlines([_[0] for _ in frames], np.min(iqms), np.max(iqms), 'm', '-')

plt.legend()
plt.grid()
plt.show()

def imshow(img):
    import IPython
    # img = cv2.resize(img, (512, 512))
    if len(img.shape) == 2:
        img = img.reshape((1, *img.shape))
    if img.shape[0] == 1:
        img = np.array([img, img, img])
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
# for fi, iqm, frame in frames:
#     imshow(frame)
