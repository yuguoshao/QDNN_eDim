import numpy as np
# from qiskit import QuantumCircuit, transpile
import qiskit
from qulacs import ParametricQuantumCircuit,QuantumState
from typing import List, Union
from collections import OrderedDict
import itertools
from qiskit.quantum_info import Statevector
import multiprocessing as mp
from effective_dimension import Model
import qulacs

# This class creates a quantum model that comprises of a feature map and variational form
# The measurement and label strategy is hardcoded and the output size = 2
def u(circuit, start, end):
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[1,1],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[2,2],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[3,3],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[1,2],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[2,1],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[1,3],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[3,1],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[2,3],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start,end],[3,2],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start],[1],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start],[2],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([start],[3],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([end],[1],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([end],[2],np.random.rand())
    circuit.add_parametric_multi_Pauli_rotation_gate([end],[3],np.random.rand())
    pass


def instructure_QubitArray(instructure):
    QubitArray = []
    last = 0
    for layer in range(len(instructure)):
        QubitArray.append([last + i for i in range(instructure[layer])])
        last += instructure[layer]
    return QubitArray


def qnn_instructure(instructure, circuit):
    QubitArray = instructure_QubitArray(instructure)
    for layer in range(len(instructure)):
        if layer == 0:
            pass
        else:
            for target in QubitArray[layer]:
                for source in QubitArray[layer - 1]:
                    u(circuit, source, target)
                    pass
                pass
            pass
        pass
    pass


class DeepQuantumNeuralNetwork(Model):
    """Creates a quantum neural network with a specified feature map and variational form."""

    def __init__(self, instructure, feature_map, neural_qubit=1, post_processing=None):
        """
        :param feature_map: quantum circuit, feature map for the data
        :param var_form: quantum circuit, parameterised variational circuit
        :param post_processing: function, returns dictionary of lists containing indices which determine post processing
        """
        super(DeepQuantumNeuralNetwork, self).__init__()
        num_qubits = np.sum(instructure)
        self.instructure=instructure
        self.QubitArray=instructure_QubitArray(self.instructure)
        self.circuit = ParametricQuantumCircuit(num_qubits)
        qnn_instructure(instructure,self.circuit)



        self.d = self.circuit.get_parameter_count()

        #self.var_form = var_form
        self.feature_map = feature_map
        #
        #self.circuit = self.circuit.combine(feature_map)
        #self.circuit = self.circuit.combine(var_form)
        #self.circuit = transpile(self.circuit)
        self.inputsize = self.instructure[0]
        self.post_processing = OrderedDict(self._parity())  # get the dictionary
        self.outputsize = len(self._parity())

    def _data_vec(self,data):
        circuit = qiskit.QuantumCircuit(self.circuit.get_qubit_count())
        sv = Statevector.from_label('0' * circuit.num_qubits)
        circuit.append(self.feature_map,[i for i in range(self.feature_map.num_qubits)])
        circuit = qiskit.transpile(circuit)
        circuit_ = circuit.assign_parameters(self._get_params_dict(data))
        result = sv.evolve(circuit_).data
        return result

    def _get_params_dict(self, x):
        """Get the parameters dict for the circuit"""
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = x[i]
        return parameters

    def assign_parameters(self,circuit,theta):
        for i,param in enumerate(theta):
            circuit.set_parameter(i,param)

    def probabilities(self,result,outqubit):
        """
        if num_qubit is None:
            num_qubit=np.log2(result.__len__())
        OtherqubitList=[i for i in range(num_qubit) if i not in outqubit]
        Densitymartix=partial_trace(result,OtherqubitList)
        """
        return Statevector(result).probabilities(outqubit)

    def forward(self, params: Union[List, np.ndarray], x: Union[List, np.ndarray]):
        """
        Computes the model output (p_theta) for given data input and parameter set
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: ndarray, p_theta for every possible basis state
        """

        # specify function to be run in parallel by each process
        def get_probs(inds, thetas, datas, circuit, results,outqubit):
            for i, theta, data in zip(inds, thetas, datas):
                init_sv=self._data_vec(data)
                init_state = QuantumState(circuit.get_qubit_count())
                init_state.load(init_sv)

                self.assign_parameters(circuit,theta)

                circuit.update_quantum_state(init_state)
                result = init_state.get_vector()

                start = i * 2 ** len(outqubit)
                end = (i + 1) * 2 ** len(outqubit)

                results[start:end] = self.probabilities(result,outqubit)

        # map input to arrays
        params = np.array(params)
        x = np.array(x)
        # specify number of parallel processes
        num_processes = 2
        # construct index set per process
        indices = []
        start = 0
        size = len(x) // num_processes
        for i in range(num_processes - 1):
            end = start + size
            indices += [list(range(start, end))]
            start = end
        indices += [list(range(end, len(x)))]

        # initialize shared array to store results (only supports 1D-array, needs reshaping later)
        results = mp.Array('d', (len(x) * 2 ** self.instructure[-1]))

        # construct processes to be run in parallel
        processes = [mp.Process(target=get_probs, args=(inds, params[inds], x[inds], self.circuit, results,self.QubitArray[-1]))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        aggregated_results = np.zeros((len(x), self.outputsize))
        num_probs = 2 ** self.instructure[-1]
        for i in range(len(x)):
            start = i * num_probs
            end = (i + 1) * num_probs
            probabilities = results[start:end]
            temp_ = []
            for y in self.post_processing.keys():
                index = self.post_processing[y]  # index for each label
                temp_.append([sum(probabilities[u] for u in index)])
            temp_ = np.reshape(temp_, (1, self.outputsize))
            aggregated_results[i] = temp_
        return aggregated_results

    def get_gradient(self, params, x):
        """
        Computes the gradients wrt parameter for every x using the forward passes.
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: numpy array, gradients
        """
        grads = []
        qc_plus = []
        qc_minus = []
        zeros = np.zeros(np.shape(params))
        for i in range(self.d):
            zeros[:, i] = np.pi / 2.
            qc_plus += [self._get_probabilities(params + zeros, x)]
            qc_minus += [self._get_probabilities(params - zeros, x)]
            zeros[:, i] = 0
            grads.append((np.array(qc_plus[i]) - np.array(qc_minus[i])) * 0.5)
        grads = np.array(grads)

        # reshape the dp_thetas
        full = np.zeros((len(x), self.d, 2 ** self.instructure[-1]))
        for j in range(len(x)):
            row = np.zeros((self.d, 2 ** self.instructure[-1]))
            for i in range(self.d):
                tensor = grads[i]
                row[i] += tensor[j]
            full[j] += row
        return full

    def get_fisher(self, gradients, model_output):
        """
        Computes the jacobian as we defined it and then returns the average jacobian:
        1/K(sum_k(sum_i dp_theta_i/sum_i p_theta_i)) for i in index for label k
        :param gradients: ndarray, dp_theta
        :param model_output: ndarray, p_theta
        :return: ndarray, average jacobian for every set of gradients and model output given
        """
        gradvectors = []
        for k in range(len(gradients)):
            jacobian = []
            m_output = model_output[k]  # p_theta size: (1, outputsize)
            jacobians_ = gradients[k, :, :]  # dp_theta size: (d, 2**num_qubits)
            for idx, y in enumerate(self.post_processing.keys()):
                index = self.post_processing[y]  # index for each label
                denominator = m_output[idx]  # get correct model output sum(p_theta) for indices
                for j in range(self.d):
                    row = jacobians_[j, :]
                    # for each row of a particular dp_theta, do sum(dp_theta)/sum(p_theta) for indices
                    # multiply by sqrt(sum(p_theta)) so that the outer product cross term is correct
                    jacobian.append(np.sqrt(denominator) * (sum(row[u] for u in index) / denominator))
            # append gradient vectors for every output for all data points
            gradvectors.append(np.reshape(jacobian, (self.outputsize, self.d)))
        # full gradient vector
        gradients = np.reshape(gradvectors, (len(gradients), self.outputsize, self.d))

        fishers = np.zeros((len(gradients), self.d, self.d))
        for i in range(len(gradients)):
            grads = gradients[i]  # size = (outputsize, d)
            temp_sum = np.zeros((self.outputsize, self.d, self.d))
            for j in range(self.outputsize):
                temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
            fishers[i] += np.sum(temp_sum, axis=0)  # sum the two matrices to get fisher estimate
        return fishers

    def _get_probabilities(self, params: Union[List, np.ndarray], x: Union[List, np.ndarray]):
        """
        Computes the model output (p_theta) for given data input and parameter set
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: ndarray, p_theta for every possible basis state
        """

        # specify function to be run in parallel by each process
        def get_probs(inds, thetas, datas, circuit, results, outqubit):
            for i, theta, data in zip(inds, thetas, datas):
                init_sv = self._data_vec(data)
                init_state = QuantumState(circuit.get_qubit_count())
                init_state.load(init_sv)

                self.assign_parameters(circuit, theta)

                circuit.update_quantum_state(init_state)
                result = init_state.get_vector()

                start = i * 2 ** len(outqubit)
                end = (i + 1) * 2 ** len(outqubit)

                results[start:end] = self.probabilities(result, outqubit)

        # map input to arrays
        params = np.array(params)
        x = np.array(x)
        # specify number of parallel processes
        num_processes = 2
        # construct index set per process
        indices = []
        start = 0
        size = len(x) // num_processes
        for i in range(num_processes - 1):
            end = start + size
            indices += [list(range(start, end))]
            start = end
        indices += [list(range(end, len(x)))]

        # initialize shared array to store results (only supports 1D-array, needs reshaping later)
        results = mp.Array('d', (len(x) * 2 ** self.instructure[-1]))

        # construct processes to be run in parallel
        processes = [mp.Process(target=get_probs, args=(inds, params[inds], x[inds], self.circuit, results,self.QubitArray[-1]))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        probabilities = []
        num_probs = 2 ** self.instructure[-1]
        for i in range(len(x)):
            start = i * num_probs
            end = (i + 1) * num_probs
            probabilities += [results[start:end]]

        # return results
        return probabilities

    # an example of post_processing function
    def _parity(self):
        y1 = []
        y2 = []
        basis_states = [list(i) for i in itertools.product([0, 1], repeat=self.instructure[-1])]
        for idx, k in enumerate(basis_states):
            parity = sum(int(k[i]) for i in range(len(k)))
            if parity % 2 == 0:
                y1.append(idx)
            else:
                y2.append(idx)
        return {'y1': y1, 'y2': y2}


if __name__ == '__main__':
    from effective_dimension import EffectiveDimension
    from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
    import matplotlib.pyplot as plt
    import numpy as np
    n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

    # number of times to repeat the variational circuit
    blocks = 1

    # number of qubits, data samples and parameter sets to estimate the effective dimension
    num_qubits = 4
    num_inputs = 10
    num_thetas = 10

    # create a feature map
    fm = ZFeatureMap(feature_dimension=num_qubits, reps=1)

    # create a variational circuit
    circ = RealAmplitudes(num_qubits, reps=blocks)

    # set up the combined quantum model
    qnet = DeepQuantumNeuralNetwork([4, 4, 4, 2], feature_map=fm)

    # number of model parameters is d
    d = qnet.d
    # set up the effective dimension and compute
    ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
    f, trace = ed.get_fhat()

    # compute the effective dimension
    effdim = ed.eff_dim(f, n)
    ###############################

    # plot the normalised effective dimension for the model
    plt.plot(n, np.array(effdim) / d)
    plt.xlabel('number of data')
    plt.ylabel('normalised effective dimension')
    plt.show()