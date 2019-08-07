from random import randint


class Perceptron:
    def __init__(self, weights):
        if isinstance(weights, list):
            self.weights = weights[1:]
            self.bias = weights[0]
        elif isinstance(weights, int):
            nweights = []
            for _ in range(weights):
                nweights.append(randint(-2, 2))
            self.bias = randint(-2, 2)
            self.weights = nweights
        else:
            raise TypeError("Invalid input type")

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        if not isinstance(weights, list):
            raise TypeError("Invalid type")
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        if not (isinstance(bias, int) or isinstance(bias, float) ):
            raise TypeError("Invalid type")
        self.bias = bias

    def calculate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(
                "The input quantity it's different than the weights quantity")

        total = self.bias
        for i in range(len(inputs)):
            total += self.weights[i]*inputs[i]
            #print('total: {}'.format(total))

        if total > 0:
            return 1

        return 0

    '''muchos inputs con sus desired outputs
        inputs son muchos inputs distintos es una lista de listas
        desired output es una lista del mismo largo que inputs con los resultados esperados de estos
    '''
    def learn(self, inputs, desiredOutput, lr=0.1):
        if not isinstance(inputs, list):
            raise ValueError("Invalid inputs type")
        elif not isinstance(desiredOutput, int):
            raise ValueError("Invalid desiredOutput type")
        elif not (isinstance(lr, float) or isinstance(lr, int)):
            raise ValueError("Invalid lr type")

        diff = desiredOutput - self.calculate(inputs)
        nweights = self.weights.copy()
        for i in range(len(nweights)):
            nweights[i] += lr * inputs[i] * diff

        self.setWeights(nweights)
        self.setBias(self.bias + lr * diff)
