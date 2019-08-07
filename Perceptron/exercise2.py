from perceptron import Perceptron
from random import randint


def createPoints(n):
    a = randint(-100, 100)/10  # from -10-0 to 10.0 with 1 decimal
    b = randint(-100, 100)/10  # from -10.0 to 10.0 with 1 decimal
    #print("a:{}, b:{}".format(a, b))

    points = []
    results = []
    for _ in range(n):
        x = randint(-600, 600)/10  # from -60.0 to 60.0 with 1 decimal
        y = randint(-600, 600)/10  # from -60.0 to 60.0 with 1 decimal

        if y > a*x+b:
            results.append(1)
            #print("({}, {}) - 1".format(x, y))

        elif y < a*x+b:
            results.append(0)
            #print("({}, {}) - 0".format(x, y))

        else:
            y += 0.1
            results.append(1)
            #print("({}, {}) - 1".format(x, y))

        points.append([x, y])

    return (points, results, (a, b))


def calculatePrecision(expected, real):
    total = 0
    for i in range(len(expected)):
        if expected[i] == real[i]:
            total += 1
    return total/len(expected)


def main():
    seasons = 100  # int(input("Seasons: "))
    n = 1000000  # int(input("Points: "))
    lr = 0.1  # float(input("Learning rate: "))

    myPerceptron = Perceptron(2)
    print("Creating Points")
    data = createPoints(n)
    print("\tDONE")
    points = data[0]
    desired = data[1]
    # line = data[2]
    results = []

    initialValues = (myPerceptron.getBias(), myPerceptron.getWeights())

    print("----------Starting----------")
    for i in range(0, seasons+1):
        print("Season {}".format(i))
        experiment = []
        if i > 0:
            print("\tTrainig")
            for j in range(len(points)):
                myPerceptron.learn(points[j], desired[j], lr)
            print("\t\tDONE")
        print("\tCalculating")
        for j in range(len(points)):
            experiment.append(myPerceptron.calculate(points[j]))
        print("\t\tDONE")
        results.append(experiment)
        print("\tPrecision: {}%".format(calculatePrecision(desired, experiment)*100))

    print("----------Finished----------")
    print("\n----------Results-----------")

    '''for i in range(len(results)):
        print("Season {}".format(i))
        print("\t precision: {}%".format(calculatePrecision(desired, results[i])*100))
    '''
    print("\n-----------Extras-----------")
    print("Initial values")
    print("    Bias:   ", initialValues[0])
    print("    Weights:", initialValues[1])
    print("Final values")
    print("    Bias:   ", myPerceptron.getBias())
    print("    Weights:", myPerceptron.getWeights())


if __name__ == "__main__":
    main()
