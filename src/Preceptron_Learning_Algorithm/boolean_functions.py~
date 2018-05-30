import random
from preceptron import Preceptron

def or_prediction():
    or_preceptron = Preceptron(3)
    #     input 1, input 2, bias
    val0 = [0,0, 1]
    val1 = [0,1, 1]
    val2 = [1,0, 1]
    val3 = [1,1, 1]
    val_labels = [0, 1, 1, 1]
    inputs = [val0, val1, val2, val3]

    for _ in range(100):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        # expects target to be -1 or 1
        if label == 0:
            label = -1
        else:
            labe = 1
        # print('randnum:',rand_num, 'input:', input, 'label:', label)
        or_preceptron.train(input, label)

    print("x or y")
    for i in inputs:
        # return -1 or 1
        pred = or_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} | {} -> {}".format(i[0], i[1], pred))

def not_prediction():
    not_preceptron = Preceptron(2)
    #     input 1, bias
    val0 = [0, 1]
    val1 = [1, 1]
    val_labels = [1, 0]
    inputs = [val0, val1]

    for _ in range(100):
        rand_num = random.randrange(0, 2)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        # expects target to be -1 or 1
        if label == 0:
            label = -1
        else:
            labe = 1
        # print('randnum:',rand_num, 'input:', input, 'label:', label)
        not_preceptron.train(input, label)

    print("not x")
    for i in inputs:
        # return -1 or 1
        pred = not_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} -> {}".format(i[0], pred))

def and_prediction():
    and_preceptron = Preceptron(3)
    #     input 1, input 2, bias
    val0 = [0,0, 1]
    val1 = [0,1, 1]
    val2 = [1,0, 1]
    val3 = [1,1, 1]
    val_labels = [0, 0, 0, 1]
    inputs = [val0, val1, val2, val3]

    for _ in range(100):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        # expects target to be -1 or 1
        if label == 0:
            label = -1
        else:
            labe = 1
        # print('randnum:',rand_num, 'input:', input, 'label:', label)
        and_preceptron.train(input, label)

    print("x and y")
    for i in inputs:
        # return -1 or 1
        pred = and_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} | {} -> {}".format(i[0], i[1], pred))

def xor_prediction():
    xor_preceptron = Preceptron(3)
    #     input 1, input 2, bias
    val0 = [0,0, 1]
    val1 = [0,1, 1]
    val2 = [1,0, 1]
    val3 = [1,1, 1]
    val_labels = [0, 1, 1, 0]
    inputs = [val0, val1, val2, val3]

    for _ in range(100):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        # expects target to be -1 or 1
        if label == 0:
            label = -1
        else:
            labe = 1
        # print('randnum:',rand_num, 'input:', input, 'label:', label)
        xor_preceptron.train(input, label)

    print("x xor y (note this function is not linearly seprable)")
    for i in inputs:
        # return -1 or 1
        pred = xor_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} | {} -> {}".format(i[0], i[1], pred))


if __name__ == "__main__":
    or_prediction()
    and_prediction()
    not_prediction()
    xor_prediction()
