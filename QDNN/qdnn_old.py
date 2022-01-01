#math related packages
import numpy as np
import scipy as sc
import qutip as qt
#further packages
from time import time
from random import sample
from random import choice
import matplotlib.pyplot as plt
import csv

def partialTraceRem(obj, rem):
    #prepare keep list
    rem.sort(reverse=True)
    keep = list(range(len(obj.dims[0])))
    for x in rem:
        keep.pop(x)
    #return partial trace:
    res = obj.ptrace(keep)
    return res

def partialTraceKeep(obj, keep):
    #return partial trace:
    return obj.ptrace(keep)

def swappedOp(obj, i, j):
    if i==j: return obj
    numberOfQubits = len(obj.dims[0])
    permute = list(range(numberOfQubits))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)

def tensoredId(N):
    #Make Identity matrix
    res = qt.qeye(2**N)
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def tensoredQubit0(N):
    #Make Qubit matrix
    res = qt.fock(2**N).proj() #For some reason ran faster than fock_dm(2**N) in tests
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res
def unitariesCopy(unitaries):
    newUnitaries = []
    for layer in unitaries:
        newLayer = []
        for unitary in layer:
            newLayer.append(unitary.copy())
        newUnitaries.append(newLayer)
    return newUnitaries
def randomQubitUnitary(numQubits):
    dim = 2**numQubits
    #Make unitary matrix
    res = sc.random.normal(size=(dim,dim)) + 1j * sc.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    #Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res


def blochvec(inputState):
    mat = 2 * sc.array(inputState)
    x = mat[0][1].real
    y = mat[1][0].imag
    z = mat[0][0].real - 1
    return [x, y, z]


def randomNetwork(qnnArch, numTrainingPairs, nVector, b):
    assert qnnArch[0] == qnnArch[-1], "Not a valid QNN-Architecture."

    # Create the targeted network unitary and corresponding training data
    networkTrainingData = randomTrainingData(numTrainingPairs, nVector, b)

    # Create the initial random perceptron unitaries for the network
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l - 1]
        numOutputQubits = qnnArch[l]

        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = qt.tensor(randomQubitUnitary(numInputQubits + 1), tensoredId(numOutputQubits - 1))
            unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)

    # Return
    return (qnnArch, networkUnitaries, networkTrainingData)

def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += trainingData[i][1].dag() * outputStates[i] * trainingData[i][1]
    return costSum.tr()/len(trainingData)
def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    #Multiply and tensor out input state
    return partialTraceRem(layerUni * state * layerUni.dag(), list(range(numInputQubits)))


def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Prepare needed states
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)

    # Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    # Multiply and tensor out output state
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))
def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0] * trainingData[x][0].dag()
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(trainingData))) * summ
    return summ.expm()


def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(storedStates[x][l - 1], tensoredQubit0(numOutputQubits))

    # Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j + 1):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni * state * productUni.dag()


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state = trainingData[x][1] * trainingData[x][1].dag()
    for i in range(len(qnnArch) - 1, l, -1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    # Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)

    # Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j + 1, numOutputQubits):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni.dag() * state * productUni


def makeUpdateMatrixTensored(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    res = qt.tensor(makeUpdateMatrix(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j),
                    tensoredId(numOutputQubits - 1))

    return swappedOp(res, numInputQubits, numInputQubits + j)


def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)]]

    # Optional
    runtime = time()

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l - 1]
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (
                            makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingData, storedStates, lda, ep, l,
                                                     j) * currentUnitaries[l][j])

        ### FEEDFORWARD
        # Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])
        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))

    # Optional
    runtime = time() - runtime
    print("Trained " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
        len(trainingData)) + " training pairs in " + str(round(runtime, 2)) + " seconds")

    # Return
    return [plotlist, currentUnitaries]


def measurements(qnnArch, unitaries, trainingData, R):
    feed = feedforward(qnnArch, unitaries, trainingData)
    prob = []
    for i in range(len(trainingData)):
        state = feed[i][-1]
        a = [] # probabilities with tr(E_i*rho)
        a.append(np.trace(np.array(qubit0mat*state)).real)
        a.append(np.trace(np.array(qubit1mat*state)).real)
        results = np.random.choice([1,-1], R, p=a)
        #number of measurements with outcome +1:
        k = 0
        for i in range(R):
            if results[i] == 1:
                k = k+1
        #empirical probability distribution:
        plus = k/R
        minus = (R - k)/R
        prob.append([plus, minus])
    return prob

def howManyWrong(qnnArch, unitaries, trainingData, R):
    prob = measurements(qnnArch, unitaries, trainingData, R)
    m = len(trainingData)
    x = 0
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0] < 0.5:
                x = x + 1
        else:
            if prob[i][1] < 0.5:
                x = x + 1
    x = x/m
    return x
def howManyWrong2(qnnArch, unitaries, trainingData, R):
    prob = measurements(qnnArch, unitaries, trainingData, R)
    m = len(trainingData)
    high = 1
    low = 0
    x = 0
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0]<high:
                high = prob[i][0]
        else:
            if prob[i][0]>low:
                low = prob[i][0]
    c = (high + low) / 2
    bCalculated = 1 - 2*c
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0] < c:
                x = x + 1
        else:
            if prob[i][1] < 1 - c:
                x = x + 1
    x = x/m
    return [x, c, bCalculated]

def howManyWrong2(qnnArch, unitaries, trainingData, R):
    prob = measurements(qnnArch, unitaries, trainingData, R)
    m = len(trainingData)
    high = 1
    low = 0
    x = 0
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0]<high:
                high = prob[i][0]
        else:
            if prob[i][0]>low:
                low = prob[i][0]
    c = (high + low) / 2
    bCalculated = 1 - 2*c
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0] < c:
                x = x + 1
        else:
            if prob[i][1] < 1 - c:
                x = x + 1
    x = x/m
    return [x, c, bCalculated]

if __name__ == '__main__':
    pass