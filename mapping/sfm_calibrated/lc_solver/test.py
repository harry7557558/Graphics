import lc_solver
from time import perf_counter

print(lc_solver.test())  # 123

vocab_path = "/home/harry7557558/GitHub/external/fbow/vocabularies/orb_mur.fbow"

voc = lc_solver.Vocabulary()

t0 = perf_counter()
voc.readFromFile(vocab_path)
t1 = perf_counter()
print("Loading vocab:", round(1000*(t1-t0)), 'ms')

assert voc.isValid()
print(voc.size(), "words")
print("K =", voc.getK())
print("desc type:", voc.getDescType())
print("desc size:", voc.getDescSize())
print("desc name:", voc.getDescName())


import cv2 as cv
import numpy as np


t0 = perf_counter()
imgs = [
    cv.imread(f"img/temp_{i}.jpg", cv.IMREAD_COLOR)
    for i in range(0, 200, 1)
]
t1 = perf_counter()
print("Load images:", round(1000*(t1-t0)), 'ms')

t0 = perf_counter()
img_features = []
detector = cv.ORB_create(6000)
for img in imgs:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    features = detector.detectAndCompute(gray, None)
    img_features.append(features)
t1 = perf_counter()
print("Extracting features:", round(1000*(t1-t0)), 'ms')

t0 = perf_counter()
bows = []
for keypoints, descriptors in img_features:
    bow = voc.transform(descriptors)
    bows.append(bow)
t1 = perf_counter()
print("Computing BOWs:", round(1000*(t1-t0)), 'ms')

t0 = perf_counter()
score_function = bows[0].score
scores = np.zeros((len(bows), len(bows)), dtype=np.float32)
for i in range(len(bows)):
    for j in range(i):
        score = score_function(bows[i], bows[j])
        scores[i,j] = scores[j,i] = score
t1 = perf_counter()
print("Pairwise scoring:", round(1000*(t1-t0)), 'ms')


import matplotlib.pyplot as plt
plt.imshow(np.log(scores+np.exp(-6)))
plt.colorbar()
plt.show()