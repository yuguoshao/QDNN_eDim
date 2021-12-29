import numpy as np
import matplotlib.pyplot as plt

# This code generates the distribution of eigenvalues for the noisy results in the Supplementary Information
# colors:
blou = np.array([32,26,184])/255
groen = np.array([0,116,0])/255
path = 'insert path here'
# set up figure
fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(8, 14))

# input size = 4

# load data
fhat_easy_qnn = np.load(path + "/data/4qubits_fhats_noise_linearZ_full.npy")
fhat_qnn = np.load(path+ "/data/4qubits_fhats_noise_linearZZ_full.npy")

# get eigenvalues
e2=[]
e3=[]
for i in range(100):
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(4,2,1)
plt.title('Easy quantum model with noise')
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='A1')
plt.ylim((0,1))
plt.legend()

plt.subplot(4,2,2)
plt.title('Quantum neural network with noise')
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='A2')
plt.ylim((0,1))
plt.legend()

# input size = 6
# load data
fhat_easy_qnn = np.load(path+"/data/6qubits_fhats_noise_linearZ_full.npy")
fhat_qnn = np.load(path+"/data/6qubits_fhats_noise_linearZZ_full.npy")
e2=[]
e3=[]

# get eigenvalues
for i in range(100):
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(4,2,3)
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='B1')
plt.ylim((0,1))
plt.legend()
plt.subplot(4,2,4)
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='B2')
plt.ylim((0,1))
plt.legend()

# input size = 8
# laod data
fhat_easy_qnn = np.load(path+"/data/8qubits_fhats_noise_linearZ_full.npy")
fhat_qnn = np.load(path+"/data/8qubits_fhats_noise_linearZZ_full.npy")
e2=[]
e3=[]

# get eigenvalues
for i in range(100):
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(4,2,5)
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='C1')
plt.ylim((0,1))
plt.legend()
plt.subplot(4,2,6)
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='C2')
plt.ylim((0,1))
plt.legend()

# input size = 10
# load data
fhat_easy_qnn = np.load(path+"/data/10qubits_fhats_noise_linearZ_full.npy")
fhat_qnn = np.load(path+"/data/10qubits_fhats_noise_linearZZ_full.npy")
e2=[]
e3=[]

# get eigenvalues
for i in range(100):
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(4,2,7)
plt.xlabel('')
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='D1')
plt.ylim((0,1))
plt.legend()

plt.subplot(4,2,8)
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 6))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='D2')
plt.ylim((0,1))
plt.legend()

# create shared axis labels
fig.text(0.5, 0.04, 'value of the eigenvalues', ha='center')
fig.text(0.04, 0.5, 'density', va='center', rotation='vertical')
plt.savefig('cumplot_noisy.eps', format='eps', dpi=1000)
plt.show()