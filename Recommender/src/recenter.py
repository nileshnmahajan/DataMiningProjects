import numpy as np

old_pred = np.loadtxt('../predictions/model7_pred.txt')
old = list(old_pred)

new_pred = []

for val in old_pred:
    if val < 1.0:
        new_pred.append(1.0)
    elif val > 5.0:
        new_pred.append(5.0)
    else:
        new_pred.append(val)

pred = np.asarray(new_pred)

np.savetxt('../predictions/model7_rnd.txt', pred, delimiter='\n', fmt='%0.5f')
