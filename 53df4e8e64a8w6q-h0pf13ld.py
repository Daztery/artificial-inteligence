import ANN
import numpy as np
import copy
from pprint import pprint
from collections import defaultdict as ddict

numNeurons = 100

class hopfieldNetwork:
    def __init__(self, Input):
        self.Input = Input
        self.InputComplement = -1 * Input
        self.numNeurons = self.Input.shape[1]
        self.neuronList = [hopfieldNeuron(1,i) for i in range(self.numNeurons)]
        self.mapInputToNeurons(self.Input)
        self.connections = ddict(list)
        self.connect_neurons()
        self.compute()
        
    def mapInputToNeurons(self, Input):
        for index, neuron in enumerate(self.neuronList):
            neuron.input = Input[:,index]
        
    def connect_neurons(self):
        for i, neuron in enumerate(self.neuronList):
            for j, neuron_2 in enumerate(self.neuronList, i+1):
                self.connections[neuron].append((neuron_2, int(neuron.input*neuron_2.input)))
                self.connections[neuron_2].append((neuron, int(neuron.input*neuron_2.input)))
    
    def compute(self, mode ='async'):
        if mode == 'sync':
            new_output = []
            for neuron in self.connections:
                for neuron_in, weight in self.connections[neuron]:
                    neuron.localField += weight*neuron_in.input
                neuron.output = neuron.activation(neuron.localField)
                new_output.append(neuron.output)
                neuron.localField = 0
            
            new_output = np.array([new_output])
            self.mapInputToNeurons(new_output)
        elif mode == 'async':
            for neuron in self.connections:
                for neuron_in, weight in self.connections[neuron]:
                    neuron.localField += weight*neuron_in.input
                neuron.output = neuron.activation(neuron.localField)
                neuron.localField = 0
                neuron.input = neuron.output
            new_output = np.array([[i.output for i in self.neuronList]])
        return new_output
    
    def runHopfieldNetwork(self, test, count=25):
        hop.mapInputToNeurons(test)
        curr_state = test
        states = [curr_state]
        while not ((curr_state == self.Input).all() or
                   (curr_state == self.InputComplement).all() or
                  count == 0):            
            curr_state = self.compute()
            states.append(curr_state)
            count -= 1
        return states
                
class hopfieldNeuron(ANN.neuron):
    def __init__(self, layer, index, activation_method='step'):
        self.input = []
        self.localField = 0
        self.activation_method= activation_method


a = np.array([[-1,1,1,1,-1,1,-1,1]])

hop = hopfieldNetwork(a)

pprint(hop.runHopfieldNetwork(np.array([[-1,-1,1,1,1,1,1,1]])))

