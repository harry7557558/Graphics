import cv2
import numpy as np
import os


video_filename = [
# [0] car (easy one)
"/media/harry7557558/OS/Archive/sfm_videos/20240110_135513.mp4",
# [1] arena 1
"/media/harry7557558/OS/Archive/sfm_videos/20240428_174233.mp4",
# [2] pride float, afternoon
"/media/harry7557558/OS/Archive/sfm_videos/20230624_201539.mp4",
# [3] pride float, front of GB
"/media/harry7557558/OS/Archive/sfm_videos/20230626_204354.mp4",
# [4] dye tank
"/media/harry7557558/OS/Archive/sfm_videos/20230903_162404.mp4",
# [5] pride float, morning (with abrupt camera motion)
"/media/harry7557558/OS/Archive/sfm_videos/20230625_072208.mp4",
# [6] desk with flowers and stickers (with poor lighting)
"/media/harry7557558/OS/Archive/sfm_videos/20230713_180332.mp4",
# [7] tree stump (easy one)
"/media/harry7557558/OS/Archive/sfm_videos/20230729_134050.mp4",
# [8] magazines on couch (with repetitive/periodic patterns)
"/media/harry7557558/OS/Archive/sfm_videos/20240217_191644.mp4",
# [9] wood bridge on table (with reflection/highlight)
"/media/harry7557558/OS/Archive/sfm_videos/20231206_210138.mp4",
# [10] pit 1 (starts with pure rotation)
"/media/harry7557558/OS/Archive/sfm_videos/20230618_230143.mp4",
# [11] pit 2 (pure rotation + poor lighting)
"/media/harry7557558/OS/Archive/sfm_videos/20230623_232927.mp4",
# [12] arena 2 (abrupt motion + poor lighting + moving shadow)
"/media/harry7557558/OS/Archive/sfm_videos/20230726_180931.mp4",
# [13] wood castle (poor lighting)
"/media/harry7557558/OS/Archive/sfm_videos/20231121_213351.mp4",
# [14] breadboard (poor lighting + extreme low focal length)
"/media/harry7557558/OS/Archive/sfm_videos/20230925_161429.mp4",
# [15] building
"/media/harry7557558/OS/Archive/sfm_videos/20230627_174847.mp4",
# [16] pit 3 (average lighting)
"/media/harry7557558/OS/Archive/sfm_videos/20230715_113856.mp4",
# [17] flight simulator
"/media/harry7557558/OS/Archive/sfm_videos/20240421_200438.mp4",
# [18] grass
"/media/harry7557558/OS/Archive/sfm_videos/20240516_171423.mp4",
# [19] pipe hole
"/media/harry7557558/OS/Archive/sfm_videos/20240518_182001.mp4",
# [20] coffee cups
"/media/harry7557558/OS/Archive/sfm_videos/20240521_183201.mp4",
# [21] crab shell (depth of field + pure rotation)
"/media/harry7557558/OS/Archive/sfm_videos/20240607_135514.mp4",
# [22] bottles (poor lighting)
"/media/harry7557558/OS/Archive/sfm_videos/20240526_235851.mp4",
# [23] pumpkins (poor lighting + dynamic obstacles)
"/media/harry7557558/OS/Archive/sfm_videos/20231026_193050.mp4",
# [24] pit 4 (poor lighting + motion blur)
"/media/harry7557558/OS/Archive/sfm_videos/20230928_144035.mp4",
# [25] eyes
"/media/harry7557558/OS/Archive/sfm_videos/20240523_185059.mp4",
# [26] woods 1, rock formation
"/media/harry7557558/OS/Archive/sfm_videos/20230729_125608.mp4",
# [27] woods 2, logs
"/media/harry7557558/OS/Archive/sfm_videos/20230729_181801.mp4",
# [28] woods 3, rock with grass
"/media/harry7557558/OS/Archive/sfm_videos/20230729_144827.mp4",
# [29] woods 4, cottage
"/media/harry7557558/OS/Archive/sfm_videos/20230729_183025.mp4",
# [30] woods 5, rock
"/media/harry7557558/OS/Archive/sfm_videos/20230729_125806.mp4",
][30]

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

        frame = cv2.resize(frame[2], (frame[2].shape[1]//2, frame[2].shape[0]//2))

        filename = 'temp_{:d}.jpg'.format(fi//skip-1)
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(filename)

    if len(frames) >= 200:
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
