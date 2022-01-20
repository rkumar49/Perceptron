import csv
import argparse
import math

def neural_network() :
    args =parser.parse_args()
    file = args.data
    eta=float(args.eta)
    data =[]
    iterator = float(args.iterations)
    with open(file) as csvfile:
        my_reader = csv.reader(csvfile, delimiter=',')
        for row in my_reader:
            data.append(row)
        w_bias_h1, w_a_h1, w_b_h1, w_bias_h2, w_a_h2, w_b_h2, w_bias_h3, w_a_h3, w_b_h3, w_bias_o, w_h1_o, w_h2_o, w_h3_o = 0.2, -0.3, 0.4, -0.5, -0.1, -0.4, 0.3, 0.2, 0.1, -0.1, 0.1, 0.3, -0.4
        print('a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o')
        print('-,-,-,-,-,-,-,-,-,-,-,0.20000,-0.30000,0.40000,-0.50000,-0.10000,-0.40000,0.30000,0.20000,0.10000,-0.10000,0.10000,0.30000,-0.40000')
        while (iterator>0):

            for i in range(0,len(data),1):
                a=float(data[i][0])
                b = float(data[i][1])
                t = float(data[i][2])

                h1 = float((a * w_a_h1) + (b * w_b_h1) + (1 * w_bias_h1));
                h1_sigma = float(sigma(h1));
                h2 = float((a * w_a_h2) + (b * w_b_h2) + (1 * w_bias_h2));
                h2_sigma = float(sigma(h2));
                h3 = float((a * w_a_h3) + (b * w_b_h3) + (1 * w_bias_h3));
                h3_sigma = float(sigma(h3));
                o = (h1_sigma*w_h1_o) + (h2_sigma*w_h2_o) + (h3_sigma*w_h3_o) + (w_bias_o*1)
                o_sigma = float(sigma(o))

                #Back Propogation
                delta_o = float(o_sigma*(1-o_sigma)*(t-o_sigma))
                delta_h1 = float(h1_sigma * (1 - h1_sigma) * (delta_o * w_h1_o))
                delta_h2 = float(h2_sigma * (1 - h2_sigma) * (delta_o * w_h2_o))
                delta_h3 = float(h3_sigma * (1 - h3_sigma) * (delta_o * w_h3_o))

                #updating weights after each iteration
                w_h1_o = w_h1_o +(eta*delta_o*h1_sigma)
                w_h2_o = w_h2_o +(eta*delta_o*h2_sigma)
                w_h3_o = w_h3_o +(eta*delta_o*h3_sigma)
                w_bias_o = w_bias_o +(eta*delta_o*1)
                w_a_h1 = w_a_h1 +(eta*delta_h1*a)
                w_b_h1 = w_b_h1 +(eta*delta_h1*b)
                w_bias_h1 = w_bias_h1 +(eta*delta_h1*1)
                w_a_h2 = w_a_h2 +(eta*delta_h2*a)
                w_b_h2 = w_b_h2 +(eta*delta_h2*b)
                w_bias_h2 = w_bias_h2 +(eta*delta_h2*1)
                w_a_h3 = w_a_h3 +(eta*delta_h3*a)
                w_b_h3 = w_b_h3 +(eta*delta_h3*b)
                w_bias_h3 = w_bias_h3 +(eta*delta_h3*1)
                result=str("%.5f" % (a))+','+str("%.5f" % (b))+','+str("%.5f" % (h1_sigma))+','+str("%.5f" % (h2_sigma))+','+str("%.5f" % (h3_sigma))+','+str("%.5f" % (o_sigma))+','+str("%.5f" % (t))+','+str("%.5f" % (delta_h1))+','+str("%.5f" % (delta_h2))+','+str("%.5f" % (delta_h3))+','+str("%.5f" % (delta_o))+','+str("%.5f" % (w_bias_h1))+','+str("%.5f" % (w_a_h1))+','+str("%.5f" % (w_b_h1))+','+str("%.5f" % (w_bias_h2))+','+str("%.5f" % (w_a_h2))+','+str("%.5f" % (w_b_h2))+','+str("%.5f" % (w_bias_h3))+','+str("%.5f" % (w_a_h3))+','+str("%.5f" % (w_b_h3))+','+str("%.5f" % (w_bias_o))+','+str("%.5f" % (w_h1_o))+','+str("%.5f" % (w_h2_o))+','+str("%.5f" % (w_h3_o))
                print(result)
            iterator = iterator - 1

#Activation Function
def sigma(x):
    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--eta", help="Learning Rate")
    parser.add_argument("-i", "--iterations", help="iterations")
    neural_network()