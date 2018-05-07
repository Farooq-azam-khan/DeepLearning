from preceptron import Preceptron
import random

def sign(val):
    if (val>0):
        return 1
    else:
        return -1
def main():
    or_preceptron = Preceptron(3)
    #     input 1, input 2, bias
    val0 = [0,0, 1]
    val1 = [0,1, 1]
    val2 = [1,0, 1]
    val3 = [1,1, 1]
    val_labels = [0, 1, 1, 1]
    inputs = [val0, val1, val2, val3]

    weights = or_preceptron.weights
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    # (sum(x*w0+y*w1+w2))
    value = val0[0]*w0 + val0[1]*w1 + w2
    sign_value = sign(value)

    ##########
    print("{:10s}".format("inputs"))
    # for i in range(len(inputs)):
    space_20 = "                    "
    space_5 = "     "
    dashes_5 = "-----"
    dashes_20 = "--------------------"
    line1 = space_20
    line2 = "   sign[sigma(Ii + wi)]"
    line3 = " = sign[x*w0 + y*w1 + w2]"
    line4 = " = sign[{}*{:.2f}+{}*{:.1f}+{:.1f}]".format(val0[0], w0, val0[1], w1, val0[2], w2)
    line5 = " = sign[{:.2f}] {:.11s}".format(value, space_5)
    line6 = " = {:1} {:.14s}".format(sign_value, space_20)
    line7 = space_20
    print("{:.5s} INPUTS/BIAS ".format(space_5), end="")
    print("{:.25s} PRECEPTRON ".format(space_5), end="")
    print("{:.40s} OUTPUT ".format(space_20))
    # print("{:.20s} ----------- ".format(space_20), end="")
    # print("{:.25s} -----------".format(space_5), end="")
    # print("{:.25s} ------ ".format(space_5+space_5))

    print("{:.20s}+{:.30s}+".format(space_20,2*dashes_20))
    print("{:.20s}|{:.30s}{:10s}|".format(space_20, line1, space_5))
    print("{:.20s}|{:.30s}{:7s}|".format(space_20, line2, space_5))
    print("{:.20s}|{:.30s}{:5s}|".format(space_20, line3, space_5))
    print("{:.20s}|{:.30s}{:5s}|".format(space_20, line4, space_5))
    print("{:.20s}|{:.30s}{:10s}|".format(space_20, line5, space_5))
    print("{:.20s}|{:.30s}{:10s}|".format(space_20, line6, space_5))
    print("{:.20s}|{:.30s}{:10s}|".format(space_20, line7, space_5))
    print("{:.20s}+{:.30s}+".format(space_20,2*dashes_20))



    ##########
    '''
    for _ in range(100):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        if label == 0:
            label = -1
        else:
            labe = 1
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
    '''

if __name__ == "__main__":
    main()
