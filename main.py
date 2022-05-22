import numpy as np
import pprint
import matplotlib.pyplot as plt

def sigmoid(x, derivative=0):
    x = np.array(x)
    if derivative==0:
        return 1/(1+np.exp(-x))
    else:
        exp = np.exp(-x)
        return exp*(1-exp)

x = [0.05, 0.1] # x0, x1, x2 where x0 is the bias
bias = [0.35, 0.6]
y_actual = [0.01, 0.99] # y1, y2

# Already prepending the bias
weights_layer1 = [[0.15, 0.2],  # w11, w12
                  [0.25, 0.3]]  # w21,

weights_layer2 = [[0.4, 0.45],  # w11, w12
                  [0.5, 0.55]]  # w21, w22

h_net = [0, 0] 

for p in range(0, 1000):
    # ----- Feed Forward
    # First Layer
    for i in range(0,len(x)):
        for j in range(0,len(weights_layer1[0])):
            h_net[j] += x[i] * weights_layer1[j][i]
        h_net[i] += bias[0] # Adding the first layer bias

    #print("Intermediate result is",h_net)
    h_out = sigmoid(h_net)
    #print("Output after activation is",h_out)

    y_net = [0,0]
    y_out = [0,0]
    # First Layer
    for i in range(0,len(x)):
        for j in range(0,len(weights_layer1[0])):
            y_net[j] += h_out[i] * weights_layer2[j][i]
        y_net[i] += bias[1] # Adding the second layer bias

    #print("Y Intermediate result is", y_net)
    y_out = sigmoid(y_net)
    #print("Y Output after activation is",y_out)

    E_y = []
    for i in range(len(y_out)):
        E_y.append((1/2)*(y_actual[i]-y_out[i])**2)
    #print(E_y)

    E_total = np.sum(E_y)
    #print(E_total)

    # Output Layer
    # Derivative for the output layer dE/dw
    # Based on the fact that it is a sigmoid
    der_E_wrt_w = []
    for i in range(len(y_out)):
        der_E_wrt_w.append( -(y_actual[i]-y_out[i])*(y_out[i]*(1-y_out[i]))*h_out[i])

    #print(der_E_wrt_w)

    new_weights_layer_2 = [[0,0],[0,0]]
    for j in range(len(der_E_wrt_w)):
        for i in range(len(weights_layer2[0])):
            new_weights_layer_2[j][i] = weights_layer2[j][i] - 0.5*der_E_wrt_w[j]

    #print(new_weights_layer_2)

    # We still need to keep track of the old weights for the entire backprop 
    der_Etotal_wrt_out = []
    # Layer 1 weights
    for j in range(0, len(weights_layer2)):
        der_E_wrt_y = []
        for i in range(len(y_out)):
            # Calculating the error wrt to net output y
            der_E_wrt_y.append(-(y_actual[i]-y_out[i])*(y_out[i]*(1-y_out[i]))*(weights_layer2[i][j]))
        # derivative of error wrt y per node
        der_Etotal_wrt_out.append(sum(der_E_wrt_y))
    #print(der_Etotal_wrt_out)

    out_h_wrt_h_net = []
    for i in range(len(h_out)):
        out_h_wrt_h_net.append(h_out[i]*(1-h_out[i]))
    #print(out_h_wrt_h_net)

    Etotal_wrt_w = []
    for i in range(len(out_h_wrt_h_net)):
        Etotal_wrt_w.append(der_Etotal_wrt_out[i]*out_h_wrt_h_net[i]*x[i])
    Etotal_wrt_w

    # Updating the hidden layer to the first layers weights 
    new_weights_layer_1 = []
    for i in range(len(weights_layer1[0])):
        temp_weights = []
        for j in range(0, len(x)):
            temp_weights.append(weights_layer1[i][j] - 0.5*(Etotal_wrt_w[i]))
        new_weights_layer_1.append(temp_weights)
    
    weights_layer1 = new_weights_layer_1
    weights_layer2 = new_weights_layer_2
    print(new_weights_layer_1)
    plt.plot(p,E_total,'rx')

plt.show()

# #print("Final Iteration")
# # ----- Feed Forward
# # First Layer
# for i in range(0,len(x)):
#     for j in range(0,len(weights_layer1[0])):
#         h_net[j] += x[i] * weights_layer1[j][i]
#     h_net[i] += bias[0] # Adding the first layer bias

# #print("Intermediate result is",h_net)
# h_out = sigmoid(h_net)
# #print("Output after activation is",h_out)

# y_net = [0,0]
# y_out = [0,0]
# # First Layer
# for i in range(0,len(x)):
#     for j in range(0,len(weights_layer1[0])):
#         y_net[j] += h_out[i] * weights_layer2[j][i]
#     y_net[i] += bias[1] # Adding the second layer bias

# ##print("Y Intermediate result is", y_net)
# y_out = sigmoid(y_net)
# print("Y Output after activation is",y_out)

# E_y = []
# for i in range(len(y_out)):
#     E_y.append((1/2)*(y_actual[i]-y_out[i])**2)
# #print(E_y)

# E_total = np.sum(E_y)
# print(E_total)