# Ammaar IftikharAAA
# 21901257
# eee443 - mini project fall22-23

# To run just execute code.py file using python code.py
# if the images are not displayed on pyplot, resize the image screen and it will be fixed.
# I tried to fix this issue but couldn't
# I have created a basic cli for interacting with the application



import h5py
import numpy as np
import math
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sb



def read_h5file(filename):
    dataset = {}
    data2 = h5py.File(filename, 'r')
    # retrieving keys from the file                                                                          
    keys = data2.keys()

    for key in keys:
        dataset[key] = (data2[key])[:]

    return keys, dataset

def plot_graph(EPOCHS, data):
    plt.plot(range(1, EPOCHS+1), data)
    plt.show()


################################################################## PARAMETER INITIALIZATION DEFINITIONS START ###################

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = xavier_distribution(shape=(layer_dims[l], layer_dims[l-1]))
        parameters["b" + str(l)] = xavier_distribution(shape=(layer_dims[l], 1)) 

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def init_lstm(parameters, dims):
    """
    initializes weights for the lstm structure
    """
    shape  = (dims[1], dims[0])
    shape2 = (dims[1], dims[1])
    parameters["z"] = xavier_distribution(shape)
    parameters["f"] = xavier_distribution(shape)
    parameters["o"] = xavier_distribution(shape)
    parameters["i"] = xavier_distribution(shape)
    
    parameters["Rz"] = xavier_distribution(shape2)
    parameters["Rf"] = xavier_distribution(shape2)
    parameters["Ro"] = xavier_distribution(shape2)
    parameters["Ri"] = xavier_distribution(shape2)
    
    parameters["bz"] =  xavier_distribution((dims[1], 1))
    parameters["bi"] =  xavier_distribution((dims[1], 1))
    parameters["bf"] =  xavier_distribution((dims[1], 1))
    parameters["bo"] =  xavier_distribution((dims[1], 1))

    return parameters
################################################################## PARAMETER INITIALIZATION DEFINITIONS END  ####################


def xavier_distribution(shape):
    """
    initializes with xavier distribution corresponding
    to the shape provided
    """
    scale = 6 / (shape[0] + shape[1])
    limit = np.sqrt( scale)
    weights = np.random.uniform(-limit, limit, size=shape)

    return weights

################################################################## ACTIVATION FUNCTION DEFINITIONS START  ###################
def softmax(Z):
    cache = Z

    return (np.exp(Z) / np.sum(np.exp(Z), axis = 0, keepdims=True)), cache

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

# Sigmoid function backward pass
def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * (s * (1-s))
    
    return dZ

# Hyperbolic Tangent function backward pass
def tanh_backward(dA, cache):
    
    Z = cache
    
    s = np.tanh(Z)**2
    dZ = dA * (1 - s)

    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dA, cache):
    Z = cache

    s = np.exp(Z) / np.sum(np.exp(Z), axis = 1)
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA, cache):
    
    Z = cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def cross_entropy_loss(AL, Y):
    m = Y.shape[0]
    cost = -1 * np.sum(np.multiply( np.log(AL.T), Y) + np.multiply( np.log(1 - AL.T), 1 - Y)) / m
    cost = np.squeeze(cost)

    return cost

def relu(Z):
    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)
    cache = Z 

    return A, cache

def tanh(Z):
    
    A = np.tanh(Z)
    cache = Z

    return A, cache

################################################################## ACTIVATION FUNCTION DEFINITIONS END  ###################

################################################################## PART 1 Autoencoder #####################################
def init_aeweights(dimens):
    """
    params
    dimens: the dimens of the encoder section of the autoencoder

    returns
    params: contains the weights for both the encoder and the
    decoder section of the autoencoder
    """
    params = {}
    
    for i in range(0, dimens.size - 1):
        w0 = np.sqrt(6 / (dimens[i] + dimens[i - 1]))
        
        params["W" + str(i)]  = np.random.uniform(-w0, w0, size=(dimens[i+1], dimens[i]))
        params["b" + str(i)] = np.random.uniform(-w0, w0, size=(dimens[i+1], 1))

    return params

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

# the function is responsible for converting
# an image into greyscale
def rgb_to_greyscale(img):
    """
    params
    img: the image to be converted

    return
    greyimg: grey scaled version of the input image
    """
    
    greyimg = np.zeros(shape=(16, 16), dtype=float)
    coefs = [0.2126, 0.7152, 0.0722]

    for i in range(3):
        greyimg += coefs[i] * img[i, :, :]

    return greyimg

def map_values(data, max):
    """
    maps to values between 0.1 - 0.9
    """
    data = (data + max) * (0.4 / max)
    data += 0.1 

    return data
    
def rgbs_to_grey(data):
    """
    params
    data: an array containing image parameters

    return
    greyimgs: a list of images after conversion to greyscale and flattened
    """
    greyimgs = []

    for i in range(data.shape[0]):
        greyimgs.append(rgb_to_greyscale(data[i]).flatten())

    return np.asarray(greyimgs)

def normalize_images(data):
    """
    """
    normal_imgs = np.zeros(shape=data.shape, dtype=float)
    arr_3stdev = 3*np.std(data.flatten()).squeeze()
   # print("std dev = ", arr_3stdev)
    for i in range(data.shape[0]):
        normal_imgs[i] = data[i] - (np.sum(data[i]) / data[i].size)
        normal_imgs[i] = np.clip(normal_imgs[i], -arr_3stdev, arr_3stdev)


    return map_values(normal_imgs, arr_3stdev)


def aeforward_pass(We, data):
    m = int(len(We) / 2)
    aecache = {}
    aecache["A0"] = data
    aecache["Z0"] = np.matmul(We["W0" ], data) + We["b0"]

    aecache["A1"], _ = sigmoid(aecache["Z0"])
    
    for i in range(1, m):
        #   print(i, "i")
        aecache["Z" + str(i)] = np.matmul(We["W" + str(i)], aecache["A" + str(i)]) + We["b" + str(i)]
        aecache["A" + str(i+1)], _ = sigmoid(aecache["Z" + str(i)])

    #print("A2----",aecache["A2"].shape)
    return aecache

def aebackpass(We, data, params, J_grad, aecache):

    m = int(len(We) / 2)
    dA_prev  = J_grad
    grads = {}
    dZ = dA_prev * aecache["A" + str(m)] * (1 - aecache["A"+ str(m)])
    
    grads["dW1"] = (np.matmul(dZ, aecache["A"+ str(m-1)].T) + params["lambda"]*We["W1"]) / 32
    grads["db1"] = np.sum(dZ, axis=1, keepdims=True) /32
    #print(grads["db1" ].shape)
    
    # for the hidden layers
    for i in reversed(range(0, m-1)):
        dA_prev = np.matmul(We["W" + str(i+1)].T, dZ)
        
        dZ =  dA_prev * aecache["A" + str(i+1)] * (1 - aecache["A"+ str(i+1)])
        kl_der = np.mean(aecache["A"+str(i+1)], axis=1, keepdims=True)
        
        dZ += params["beta"]*((-params["rho"]/kl_der) + ((1 - params["rho"]) / (1 - kl_der)))

        grads["dW" + str(i)] = (np.matmul(dZ, aecache["A"+ str(i)].T) + params["lambda"]*We["W" + str(i)] ) /32
        grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True) / 32

    return grads

def aeCost(We, data, params):
    """
    calculates the cost value and the gradient using the
    forward pass calculated.
    """
    L = int(len(We) / 2)
    activ_H = np.zeros(shape=(params["Lhid"]), dtype=float)
    J_grad = {}
    aecache = params["cache"]
    cost = 0
    
    for i in range(0, L):
        cost += params["lambda"] * (np.linalg.norm(We["W" + str(i)].flatten()) ** 2) / 2

    temp_avg = np.mean(aecache["A1"], axis=1, keepdims=True)
    
    cost +=  (np.linalg.norm(data.T - aecache["A2"]).squeeze() ** 2) / (2*data.shape[0])
    kl    =   np.sum(params["rho"] * np.log(params["rho"] / temp_avg) + (1 - params["rho"]) * np.log((1-params["rho"]) / (1 - temp_avg)))
    
    cost += params["beta"] * kl
    
    J_grad = (aecache["A2"] - data.T)
    
    return (cost, J_grad)

def solver(We, data, params, batch_size):

    for k in range(100):
        i = 0
        while i <= (data.shape[0] - batch_size):
#            print("delata is back")
            aecache = aeforward_pass(We, data[range(i, i + batch_size), :].T)
            #act_avg["A1"] = (act_avg["A1"] * i + (np.sum(aecache["A1"], axis=0) / batch_size)) / (i+1)
            #print(act_avg["A1"])
            params["cache"] = aecache
            J, J_grad = aeCost(We, data[range(i, i + batch_size), :], params)
            
            grads = aebackpass(We, data[range(i, i + batch_size), :], params, J_grad, aecache)
            grads["dW0"] = (grads["dW0"] + grads["dW1"].T) /2
            grads["dW1"] = grads["dW0"].T
            
            for n in range(2):
                We["W" + str(n)] -= 0.1*grads["dW" + str(n)] 
                We["b" + str(n)] -= 0.1*grads["db" + str(n)]
                
            i += batch_size
        print("Epoch ", k)
        print("loss --- ", J)
    return We

def run_aencoder(We, manup_data, params, batch_size):
    
    We = solver(We, manup_data, params, batch_size)
    img = normalize_images(We["W0"])
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    fig = plt.figure(figsize=(params["Lhid"]+2, params["Lhid"]+2))
    row =  col = int(params["Lhid"] ** .5)
    for i in range(params["Lhid"]):
        pic = img[i, :].reshape(16, 16)
        
        fig.add_subplot(row, col, i+1)
        plt.axis("off")
        plt.imshow(pic, cmap="gray", interpolation="nearest")

    plt.show()

def init_aesystem():
    # read file
    keys, data = read_h5file("data1.h5")
    params = {}
    params["Lin"], params["Lhid"], params["rho"] = 256, 49, 0.1
    params["beta"], params ["lambda"] = .01, 0.0001

    # convert to greyscale
    # sub mean pixel intensities
    # clip at (+/-) 3 std dev
    manup_data = rgbs_to_grey(data["data"])
    manup_data = normalize_images(manup_data)

    #fig = plt.figure(figsize=(20, 20))

    #for i in range(1, 201):
    #    fig.add_subplot(10, 20, i)
    #    plt.axis('off')
    #    plt.imshow(manup_data[i].reshape(16, 16).astype(float), cmap='gray', interpolation="nearest")
        
    #plt.show()
    
    #plt.imshow(manup_data[1, :].reshape((16, 16)).T, cmap="gray", interpolation="nearest")
    #plt.show()

    # initialize weights 
    We = init_aeweights(np.array([params["Lin"], params["Lhid"], params["Lin"]]))
    We["W0"] = (We["W0"] + We["W1"].T) /2
    We["W1"] = We["W0"].T

    return We, manup_data, params

########################################################################## PART 2   ######################################################

# Forward and backward pass for the 2nd question
def forward_propagation_2(X, W, b, batch_size):
    A, Z = {}, {}

    A["0"] = X
    Z["1"] = np.matmul(W["0"], A["0"])   # Dx250 * 250x3*batch_size

    A["1"] = Z["1"].T.reshape(batch_size, 3*W["0"].shape[0]).T

    Z["2"] = np.matmul(W["1"], A["1"]) + b["1"] # Px3D  * 3Dxbatch_size                                            
    A["2"], _ = sigmoid(Z["2"])  # Pxbatch_size

    Z["3"] = np.matmul(W["2"], A["2"])  + b["2"]# 250xP * Pxbatch_size
    A["3"], _ = softmax(Z["3"]) # 250xbatch_size

    return A, Z
    
def backpropagation2(A, Y, Z, b, W):
    dZ, dW, db = {}, {}, {}
    
    dA_prev = (Y - A["3"])                         # 250x200 
    
    dZ = dA_prev 
    
    dW["2"] = np.matmul(dZ, A["2"].T) / 200 # Px200  * 200x250 
    db["2"] = np.sum( dZ, axis=1, keepdims=True) / 200   

    dA_prev = np.matmul(W["2"].T, dZ)
    dZ = dA_prev * (A["2"] * (1-A["2"]))
    
    dW["1"] = np.matmul(dZ, A["1"].T) / 200     # 3Dx200  * 200xP
    db["1"] = np.sum( dZ, axis=1, keepdims=True) / 200

    dA_prev = np.matmul(W["1"].T, dZ)

    d, e = W["0"].shape[0], 250

    X = A["0"].T.reshape((-1, 3*e)).T
    
    dW["0"] = np.zeros(shape=(d, e), dtype=float)
    for i in range(3):
        dW["0"] = np.matmul(dA_prev[range(i*d, (i+1)*d), :], X[range(i*e, (i+1)*e), :].T) + dW["0"]

    dW["0"] = dW["0"] / 200    
    
    return dW, db

def update_parameters2(W, b, dW, db, learning_rate, rate, old_grad):

    old_grad["0"]  =  dW["0"] + rate*old_grad["0"]
    W["0"] += learning_rate*old_grad["0"]

    for i in range(1, 3):
        old_grad[str(i)]  = dW[str(i)]	+ rate*old_grad[str(i)]
        W[str(i)]  += learning_rate * old_grad[str(i)]
        b[str(i)]  += learning_rate * db[str(i)]

    return W, b, old_grad


def red_hot_encoding(data, labels):
    X = np.zeros(shape=(250, 3*data.shape[0]), dtype=float)
    Y = np.zeros(shape=(250, labels.size))

    for i in range(labels.size):
        Y[(labels[i] - 1), i] = 1

    j = 0
    while j < data.shape[0]:
        for i in range(3):
            X[(data[j,i] - 1), i + (3*j)] = 1
        j += 1
                
    return X, Y

def check_acc2(W, b, data):
    #ind_list = [i for i in range(data["d"].size)]
    #### shuffling the data                                                                                                                                 
    #shuffle(ind_list)
    inp, labels = red_hot_encoding(data["x"], data["d"])
    
    A, Z = forward_propagation_2(inp, W, b, labels.shape[1])
    cost = cross_entropy_loss(A["3"], labels.T)

    predictions = np.argmax(A["3"], axis=0) + 1
    count, j = 0, 0
    word_list = []
    
    #inp = inp[:, ind_list]
    #labels = labels[ind_list]
    
    while  j < labels.shape[1]:
        if predictions[j] == data["d"][j]:
            if len(word_list) < 11 and (j % 17 == 0):
                t = (j, predictions[j])
                word_list.append(t)
                
            count += 1
        j += 1

    print("correct prediction  = ", count)
    print("prediction percentage = ", count/data["d"].size)
    print("validation set loss = ", cost)

    return cost, count/data["d"].size, word_list

def train2(X, Y, W, b, val,  learning_rate, epochs, batch_size, mom_rate):
    cost_arr, acc_arr = [], []
    grad_mom = {}
    grad_mom["1"], grad_mom["2"], grad_mom["0"] = 0, 0, 0
    
    for l in range(epochs):
        print("epoch = ", l)

        j, k = 0, 0
        while j <= (Y.shape[1] - batch_size):            
            A, Z = forward_propagation_2(X[:, range(k, k + (3*batch_size))], W, b, batch_size)
            dW, db = backpropagation2(A, Y[:,range(j, j + batch_size)], Z, b, W)
            W, b, grad_mom = update_parameters2(W, b, dW, db, learning_rate, mom_rate, grad_mom)
            
            j += batch_size
            k += batch_size*3
    
        cost, acc, _ = check_acc2(W, b, val)
        
        cost_arr.append(cost)
        acc_arr.append(acc)
        
        ## stops when cost becomes less than 0.1
        if cost < 0.1:
            break

    ## displaying the graphs
    plt.title("Validation Cost Graph")
    plot_graph(len(cost_arr), cost_arr)
    plt.title("Validation Accuracy Graph")
    plot_graph(len(acc_arr), acc_arr)
    return W, b

def word_predicter(learning_rate, epochs, batch_size, mom_rate, dp):
    ## DECLARATION AND INITIALIZATION
    keys, data2 = read_h5file("data2.h5")
    D, P = dp
    W, b, val, test = {}, {}, {}, {}

    W["0"], W["1"] = np.random.normal(0, 0.01, D*250).reshape((D, 250)), np.random.normal(0, 0.01, D*3*P).reshape((P, 3*D))
    W["2"] = np.random.normal(0, 0.01, P*250).reshape((250, P))    
    b["1"], b["2"]= np.random.normal(0, 0.01, P).reshape((P, 1)), np.random.normal(0, 0.01, 250).reshape((250, 1))

    test["x"], test["d"] = data2["testx"], data2["testd"]
    val["x"], val["d"] = data2["valx"], data2["vald"]
    
    # word embedding processing
    # we will use one-hot vector for the input text
    X, Y = red_hot_encoding(data2["trainx"], data2["traind"])
    W, b = train2(X, Y, W, b, val, learning_rate, epochs, batch_size, mom_rate)

    ## test set
    print("\nTEST SET INFORMATION:")
    cost, acc, wordind = check_acc2(W, b, test)
   
    for i in wordind:
        ind, pred = i
        
        left = data2['words'][data2["testx"][ind] - 1]
        right = data2['words'][pred - 1]
        print(left, right)
        
################################################################# PART #3 ###################

# HELPER FUNCTIONS FOR FORWARD PROPAGATION
def linear_forward(A, W, b):
    Z = np.matmul(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache     = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "tanh":
        A, activation_cache = tanh(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



##################################### LSTM   DEFINITIONS ################################################
def lstm_forward(W, X, T):
    lstm_cache = {}
    
    z = np.zeros((W["z"].shape[0], T, X.shape[2]))
    inp = np.zeros((W["i"].shape[0], T, X.shape[2]))
    f = np.zeros((W["f"].shape[0], T, X.shape[2]))
    o = np.zeros((W["o"].shape[0], T, X.shape[2]))
    #c = np.zeros((W["z"].shape[0], T, X.shape[2]))
    c = [0]
    y = [np.zeros((W["o"].shape[0], X.shape[2]))]
    
    for i in range(T):
        ## linear activation forward for bloack input
        z[:, i, :], cache = tanh(np.matmul(W["z"], X[:, i,:]) + np.matmul(W["Rz"], y[-1]) + W["bz"])
        #z[:, i,:], cache = linear_activation_forward( X[:, i,:], W["z"], W["bz"], "sigmoid")
        #lstm_cache["z" +str(i)] = cache

        ## input gate
        inp[:, i, :], cache = sigmoid(np.matmul(W["i"], X[:, i,:])+ np.matmul(W["Ri"], y[-1]) + W["bi"])
        #inp[:, i,:], cache = linear_activation_forward(X[:, i,:], W["i"], W["bi"], "sigmoid")
        #lstm_cache["f"+str(i)] = cache

        ## forget gate
        f[:, i, :], cache = sigmoid(np.matmul(W["f"], X[:, i,:]) + np.matmul(W["Rf"], y[-1])+ W["bf"])
        #f[:, i,:], cache = linear_activation_forward(X[:, i,:], W["f"], W["bf"], "sigmoid")
        #lstm_cache["f"+str(i)] = cache

        ## cell
        c.append(z[:, i, :] * inp[:, i,:] + f[:, i, :] * c[-1])

        ## output gate
        o[:, i, :], cache = sigmoid(np.matmul(W["o"], X[:, i,:])+ np.matmul(W["Ro"], y[-1]) + W["bo"])
        #o[:, i,:], cache = linear_activation_forward(X[:, i, :], W["o"], W["bo"], "sigmoid")
        #lstm_cache["o"+str(i)] = cache
        
        ## block output
        y.append(np.tanh(c[-1]) * o[:, i, :])

    lstm_cache["x"] = X
    lstm_cache["z"] = z
    lstm_cache["f"] = f
    lstm_cache["o"] = o
    lstm_cache["i"] = inp
    lstm_cache["c"] = c
    lstm_cache["params"] = W
    lstm_cache["y"] = y
    
    return (y, lstm_cache)

def lstm_backward(dA, cache, T):
    """
    responsible for backpropagation in the lstm 
    """
    
    grads = {}
    W = cache["params"]
    ## initializing the gradients
    grads["Wf"] = np.zeros(W["f"].shape)
    grads["Wo"] = np.zeros(W["o"].shape)
    grads["Wi"] = np.zeros(W["i"].shape)
    grads["Wz"] = np.zeros(W["z"].shape)
    grads["bf"] = np.zeros(W["bf"].shape)
    grads["bo"] = np.zeros(W["bo"].shape)
    grads["bi"] = np.zeros(W["bi"].shape)
    grads["bz"] = np.zeros(W["bz"].shape)
    grads["Rf"] = np.zeros(W["Rf"].shape)
    grads["Ro"] = np.zeros(W["Ro"].shape)
    grads["Ri"] = np.zeros(W["Ri"].shape)
    grads["Rz"] = np.zeros(W["Rz"].shape)

    ## biases                                                                                                                                                                    
    for i in reversed(range(1, T)):
        #dc = dA * o[:, i, :]
        dc = dA * (1 - (np.tanh(cache["c"][i])**2))

        ## de / dh
        dWo = dA * cache["o"][:, i, :] * (1 - cache["o"][:, i, :]) * np.tanh(cache["c"][i])
        dWf = cache["o"][:, i, :] * dc * cache["f"][:, i, :] * (1 - cache["f"][:, i, :]) * cache["c"][i-1]
        dWi = cache["o"][:, i, :] * dc * cache["i"][:, i, :] * (1 - cache["i"][:, i, :]) * cache["z"][:, i, :]
        dWz = cache["o"][:, i, :] * dc * (1 - (cache["z"][:, i, :]**2)) * cache["i"][:, i, :]

        ## calculating changes in weights and biases
        grads["Wf"] += np.matmul(dWf, cache["x"][:, i, :].T) / cache["x"].shape[2]
        grads["Wo"] += np.matmul(dWo, cache["x"][:, i, :].T) / cache["x"].shape[2]
        grads["Wi"] += np.matmul(dWi, cache["x"][:, i, :].T)  / cache["x"].shape[2]
        grads["Wz"] += np.matmul(dWz, cache["x"][:, i, :].T)  / cache["x"].shape[2]

        grads["Rf"] += np.matmul(dWf, cache["y"][i-1].T)  / cache["x"].shape[2]
        grads["Ro"] += np.matmul(dWo, cache["y"][i-1].T)  / cache["x"].shape[2]
        grads["Ri"] += np.matmul(dWi, cache["y"][i-1].T)  / cache["x"].shape[2]
        grads["Rz"] += np.matmul(dWz, cache["y"][i-1].T)  / cache["x"].shape[2]
        
        ## biases
        grads["bf"] +=np.sum(dWf, axis=1, keepdims=True)  / cache["x"].shape[2]
        grads["bo"] +=np.sum(dWo, axis=1, keepdims=True)  / cache["x"].shape[2]
        grads["bi"] +=np.sum(dWi, axis=1, keepdims=True)  / cache["x"].shape[2]
        grads["bz"] +=np.sum(dWz, axis=1, keepdims=True)  / cache["x"].shape[2]

        dA = np.matmul(W["Ro"].T, dWo) + np.matmul(W["Rz"].T, dWz) + np.matmul(W["Rf"].T, dWf) + np.matmul(W["Ri"].T, dWi)
        
    return grads


################################################################## LSTM ENDS     ##############################################################


##################################################################        GRU    ##############################################################
def gru_forward(W, X, T):
    gru_cache = {}
   
    z = np.zeros((W["z"].shape[0], T, X.shape[2]))
    inp = np.zeros((W["i"].shape[0], T, X.shape[2]))
    f = np.zeros((W["f"].shape[0], T, X.shape[2]))
    o = np.zeros((W["o"].shape[0], T, X.shape[2]))
    #c = np.zeros((W["z"].shape[0], T, X.shape[2]))
    c = [0]
    y = [np.zeros((W["o"].shape[0], X.shape[2]))]
   
    for i in range(T):
        ## linear activation forward for bloack input
        z[:, i, :], cache = sigmoid(np.matmul(W["z"], X[:, i,:]) + np.matmul(W["Rz"], y[-1]) + W["bz"])
        # output gate
        o[:, i, :], cache = sigmoid(np.matmul(W["o"], X[:, i,:])+ np.matmul(W["Ro"], y[-1]) + W["bo"])
        ## input gate
        inp[:, i, :], cache = tanh(np.matmul(W["i"], X[:, i,:])+ np.matmul(W["Ri"], y[-1] * o[:, i, :]) + W["bi"])
        ## cell
        c.append((z[:, i, :]) * inp[:, i,:] + (1 - z[:, i, :]) * c[-1])
        ## block output
        y.append(c[-1])

    gru_cache["x"] = X
    gru_cache["z"] = z
    gru_cache["f"] = f
    gru_cache["o"] = o
    gru_cache["i"] = inp
    gru_cache["c"] = c
    gru_cache["params"] = W
    gru_cache["y"] = y
   
    return (y, gru_cache)

def gru_backward(dA, cache, T):
    """
    responsible for backpropagation in the gru
    """
   
    grads = {}
    W = cache["params"]
    ## initializing the gradients
    grads["Wf"] = np.zeros(W["f"].shape)
    grads["Wo"] = np.zeros(W["o"].shape)
    grads["Wi"] = np.zeros(W["i"].shape)
    grads["Wz"] = np.zeros(W["z"].shape)
    grads["bf"] = np.zeros(W["bf"].shape)
    grads["bo"] = np.zeros(W["bo"].shape)
    grads["bi"] = np.zeros(W["bi"].shape)
    grads["bz"] = np.zeros(W["bz"].shape)
    grads["Rf"] = np.zeros(W["Rf"].shape)
    grads["Ro"] = np.zeros(W["Ro"].shape)
    grads["Ri"] = np.zeros(W["Ri"].shape)
    grads["Rz"] = np.zeros(W["Rz"].shape)

    ## biases                                                                                                                                                                                          
    for i in reversed(range(2, T)):
        #dc = dA * o[:, i, :]
        dc = dA

        ## de / dh
        #print("c --",cache["c"][i-1].shape)
        #print("WRi", W["Ri"].shape)
        dWi = dA * cache["i"][:, i, :] * (1 - (cache["i"][:, i, :]**2)) * cache["z"][:, i, :]
        dWz = dA * (1 - (cache["z"][:, i, :])) * cache["z"][:, i, :] * (cache["i"][:, i, :] - cache["c"][i - 1])
        dWo = dWi * np.matmul(W["Ri"], cache["c"][i - 1]) * cache["o"][:, i, :] * (1 - cache["o"][:, i, :])
       
        ## calculating changes in weights and biases
        grads["Wo"] += np.matmul(dWo, cache["x"][:, i, :].T)  / cache["x"].shape[2]
        grads["Wi"] += np.matmul(dWi, cache["x"][:, i, :].T)  / cache["x"].shape[2]
        grads["Wz"] += np.matmul(dWz, cache["x"][:, i, :].T)  / cache["x"].shape[2]

        grads["Ro"] += np.matmul(dWo, cache["y"][i-1].T)  / cache["x"].shape[2]
        grads["Ri"] += np.matmul(dWi, cache["y"][i-1].T)   / cache["x"].shape[2]
        grads["Rz"] += np.matmul(dWz, cache["y"][i-1].T)  / cache["x"].shape[2]
       
        ## biases

        grads["bo"] +=np.sum(dWo, axis=1, keepdims=True)  / cache["x"].shape[2]
        grads["bi"] +=np.sum(dWi, axis=1, keepdims=True)  / cache["x"].shape[2]
        grads["bz"] +=np.sum(dWz, axis=1, keepdims=True)  / cache["x"].shape[2]

        #dA = np.matmul(W["Ro"].T, dWo) + np.matmul(W["Rz"].T, dWz) + np.matmul(W["Ri"].T, dWi)
        dA= dA * (1 - cache["z"][:, i, :]) + np.matmul(W["Rz"], dWz) + np.matmul(W["Ri"], dWi) * cache["o"][:, i, :] + np.matmul(W["Ro"].T, dWo)
    return grads

	
def truncated_forwardpass(W, V, b2, X, i, t):
    """
    uses recursion to compute the output of the 
    """
    if t == i:
        Xmod = X[:, 0, :]
        A, cache =  linear_activation_forward(Xmod, V, b2, "tanh")

        return A, (None, (cache, None))
    else:
        A, cache = truncated_forwardpass(W, V, b2, X[:, :, :], i + 1, t)
        Xmod = X[:, t - i, :]
        Y = (np.matmul(W, A) + np.matmul(V, Xmod) + b2)
        
        cacheW = ((A, W, b2), Y)                              # (activation_cache, linear_cache), assuming 0 bais
        cacheV = ((Xmod, V, b2), Y)

        A = np.tanh(Y)
        cache = (cache, (cacheV, cacheW))
        
        return A, cache                                       # (A, cache)   ---- will be used recursively in backpropagation

# FORWARD PROPAGATION ALGORITHM
def forward_propagation(X, parameters, activations, alg, T):
    """
     parameters will contain both parameters for all layers, i.e, bias and weights 
    """
    
    # DECLARATION AND INITIALIZATION 
    A = X
    L = len(parameters) 
    caches = []
 
    if alg == "RNN":
        A, trun_cache = truncated_forwardpass(parameters["W"], parameters["V"], parameters["Vb"], X, 0, T)
        caches.append(trun_cache)
        L -= 4
    elif alg == "lstm":
        A, lstm_cache = lstm_forward(parameters, X, T+1)
        caches.append(lstm_cache)
        A = A[-1]
        
        L -= 12
    elif alg == "gru":
        A, gru_cache = gru_forward(parameters, X, T+1)
        caches.append(gru_cache)
        A = A[-1]
	
        L -= 12
                                     
    L = int(L / 2)
    ## forward pass FFNN/MLP part of the network
    for l in range( 1, L):
        #print(l)
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    # output layer
    A, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)
    
    return A, caches



###########################################################################################################
# HELPER FUNCTIONS FOR BACKPROPAGATION
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.matmul( dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.matmul(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)         # the number of layers
    m = AL.shape[1]
    Y = Y.T                 # after this line, Y is the same shape as AL

    dAL = AL - Y      
    
    linear, nonlin = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dAL, linear)
    dA_prev = grads["dA" + str(L-1)]
    
    for l in reversed(range( L - 1)):
        dA_prev, dW_prev, db_prev = linear_activation_backward(dA_prev, caches[l], "relu")
        grads["dA" + str(l)]   = dA_prev
        grads["dW" + str(l+1)] = dW_prev
        grads["db" + str(l+1)] =  db_prev

    return grads, dA_prev


############################################################### Recursively implemets the rollout of input in time ###################################
def truncated_backpass(dA_prev, caches, t):
    """
    This function responsible for computing the delta for
    the recurrent layer weights and biases

    It uses recursion to accumulate the error through time...

    Inputs:
    dA_prev: The dE/dA of the layer following the recurrent layer
    caches:  Contains the information to compute the dw and dbs

    Outputs:
    grads: contains net 
    """
    
    next_cache, cur_cache = caches
 
    if t == 0:
        # evaluate for v only\
        cacheV, cacheW = cur_cache
        _, dV, db2       = linear_activation_backward(dA_prev, cacheV, "tanh")
        
        return dV, 0, 0, db2
    else:
        cacheV, cacheW   = cur_cache
        _,       dV, db2 = linear_activation_backward(dA_prev, cacheV, "tanh")
        dA_prev, dW, db1 = linear_activation_backward(dA_prev, cacheW, "tanh")
        dV_trun, dW_trun, db1_trun, db2_trun  = truncated_backpass(dA_prev, caches, t - 1)

        dW += dW_trun * 0.001
        dV += dV_trun * 0.001
        db2 += db2_trun * 0.001
        db1 += db1_trun * 0.001
        
        return dV, dW, db1, db2

# BACKPROPAGATION ALGORITHM
def backpropagation(A, Y, caches, t, alg):
    """
    calls MLP backpropagation + (rnn or lstm or gru) backpropagation to
    estimate the weights and biases changes
    Inputs:
    A:      the output of the entire network
    Y:      the desired output
    caches: the information desired for updates.
    t: T - 1. (Total time rollout - 1)
    alg: rnn or lstm or gru specification
    
    Outputs:
    grads: the estimated changes required
    """
    # t == 0 for non RNNetworks
    grads, dA_prev = L_model_backward(A, Y, caches[int(t!=0):])
    
    if t != 0 and alg == "RNN":
        grads["dV"], grads["dW"], grads["dWb"], grads["dVb"] = truncated_backpass(dA_prev, caches[0], t)
        grads["dV"], grads["dW"], grads["dVb"] = grads["dV"], grads["dW"], (grads["dWb"] + grads["dVb"])
    elif t!= 0 and alg == "lstm":
        grads["lstm"] = lstm_backward(dA_prev, caches[0], t+1)
    elif t != 0 and alg == "gru":
        grads["lstm"] = gru_backward(dA_prev, caches[0], t+1)
    return grads

############################################################################################################################################

# UPDATES THE PARAMETERS (Ws and bs)
def update_parameters(parameters, grads, learning_rate, rate, old_grad, alg):
    L = len(parameters)
            
    if alg == "RNN":
        L -= 4
        rnn_params = ["V", "W", "Vb"]
        
        for j in rnn_params:
            old_grad[j] = (rate * old_grad[j]) - (learning_rate * grads["d" + j])
            parameters[j] += old_grad[j] 
    elif alg == "lstm" or alg == "gru":
        lstm_params = ["o", "i", "z"]
        if alg == "lstm":
            lstm_params.append("f")
            
        L -= 12
        
        for j in lstm_params:
            old_grad[j] = (rate * old_grad[j]) - (grads["lstm"]["W" + j])
            old_grad["b"+j] = (rate * old_grad["b"+j]) - (grads["lstm"]["b" + j])
            old_grad["R"+j] = (rate * old_grad["R"+j]) - (grads["lstm"]["R" + j])
            parameters[j] += learning_rate*old_grad[j]
            parameters["b"+j] += learning_rate*old_grad["b"+j]
            parameters["R"+j] += learning_rate*old_grad["R"+j]


    L = int(L / 2)
    
    # operates on the Multi layer perceptron
    for i in range(1, L+1):
        old_grad["b" + str(i)] = (rate * old_grad["b" + str(i)]) - (grads["db" + str(i)])
        old_grad["W" + str(i)] = (rate * old_grad["W" + str(i)]) - (grads["dW" + str(i)])
        parameters["W" + str(i)]  += learning_rate * old_grad["W" + str(i)]
        parameters["b" + str(i)]  += learning_rate * old_grad["b" + str(i)]

    return parameters, old_grad


############################################################################################
# GIVES STATISTICS ABOUT THE PERFORMANCE OF THE ALGORITHM
def check_acc3(params, data, alg):
    inp, labels = data["x"], data["d"]
    count, j = 0, 0,
    conf_mat = np.zeros((6,6))
    A, caches = forward_propagation(inp[ :, :, :], params, {}, alg, 149)
            
    predictions = np.argmax(A, axis=0)
    correct = np.argmax(labels, axis=1)
    #print(correct)
    for j in range(labels.shape[0]):
        ## conf mat + acc
        conf_mat[correct[j], predictions[j]] += 1
        if predictions[j] == correct[j]:
            count += 1
              
    #print("A.shape", A.shape)
    #print("Y.shape", labels.shape)
    cost = cross_entropy_loss(A, labels)

    print("correct prediction count  = ", count)
    print("prediction percentage     = ", count/labels.shape[0])
    print("set loss                  = ", cost)    

    return cost, count/labels.shape[0], conf_mat
    
# RUNS THE FORWARD AND BACKPROPAGATION ALGORITHM IN ADDITION TO THE VALIDATION
# SCORING ALGORITHM TO TRAIN THE NETWORK
def train3(X, Y, params, val,  learning_rate, epochs, batch_size, mom_rate, t, alg):
    m = len(params)
    cost, accuracy = np.zeros( epochs, dtype=float), np.zeros( epochs, dtype=float)
    
    ## INITIALIZING GRADIENT MOMENTUM TO ZERO
    grad_mom = {}
    grad_mom["V"],  grad_mom["W"],  grad_mom["Vb"], grad_mom["Wb"] = 0, 0, 0, 0
    grad_mom["o"],  grad_mom["i"],  grad_mom["f"],  grad_mom["z"]  = 0, 0, 0, 0
    grad_mom["bo"], grad_mom["bf"], grad_mom["bi"], grad_mom["bz"] = 0, 0, 0, 0
    grad_mom["Ro"], grad_mom["Rf"], grad_mom["Ri"], grad_mom["Rz"] = 0, 0, 0, 0
    
    for i in range(1, int(m/2) - 1):
        grad_mom["W" + str(i)] = 0
        grad_mom["b" + str(i)] = 0 

    ## TRAINING THE NETWORK
    for k in range(epochs):
        print("\nepoch = ", k)
        
        j = 0
        while j <= (Y.shape[0] - batch_size):
            A, caches = forward_propagation(X[:, :,range(j, j+32)], params, {}, alg, t)
            grads = backpropagation(A, Y[range(j, j + 32), :], caches, t, alg)
            params, grad_mom = update_parameters(params, grads, learning_rate, mom_rate, grad_mom, alg)
                
            j += batch_size
        cost[k], accuracy[k], _ = check_acc3(params, val, alg)

        ### terminate training if loss is less than 0.1
        if cost[k] < 0.1:
            break
    
    plt.title("Cost Graph")
    plot_graph(epochs, cost)
    plt.title("Accuracy Graph")
    plot_graph(epochs, accuracy)
    return params

def run_lstm(trainset, valset, layer_dims):
    ## TRAINING HYPERPARAMETERs
    TAO, BATCH_SIZE, LEARNING_RATE, EPOCHS = 149, 32, 0.1, 50

    # INITIALIZATION OF THE NETWORK
    X, Y = trainset
    parameters = initialize_parameters_deep(layer_dims)
    parameters = init_lstm(parameters, [X.shape[0], layer_dims[0]])

    return train3(X, Y, parameters, valset, LEARNING_RATE, EPOCHS, BATCH_SIZE, 0.85, TAO, "lstm")

def run_gru(trainset, valset, layer_dims):
    ## TRAINING HYPERPARAMETERs                                                                                                                                                                             
    TAO, BATCH_SIZE, LEARNING_RATE, EPOCHS = 149, 32, 0.1, 50

    # INITIALIZATION OF THE NETWORK                                                                                                                                                                        
    X, Y = trainset
    parameters = initialize_parameters_deep(layer_dims)
    parameters = init_lstm(parameters, [X.shape[0], layer_dims[0]]) ## the gru has all the parameters the lstm has except the forget gate

    return train3(X, Y, parameters, valset, LEARNING_RATE, EPOCHS, BATCH_SIZE, 0.85, TAO, "gru")

def run_rnn(trainset, valset, layer_dims):
    ## TRAINING HYPERPARAMETERs
    TAO, BATCH_SIZE, LEARNING_RATE, EPOCHS = 149, 32, 0.01, 50
    
    ## INITIALIZING WEIGHTS AND BIASES FOR RNN
    X, Y = trainset
    parameters = initialize_parameters_deep(layer_dims)
    parameters["V"], parameters["W"]   = xavier_distribution((128, 3)), xavier_distribution((128, 128))
    parameters["Vb"], parameters["Wb"] = np.zeros(shape=(128, 1)), np.zeros(shape=(128, 1))
    
    return train3(X, Y, parameters, valset, LEARNING_RATE, EPOCHS, BATCH_SIZE, 0.85, TAO, "RNN")

def run3(alg):
    # DECLARATION AND INITIALIZATION
    layer_dims = [ 128, 32, 16, 6]
    keys, dataset = read_h5file("data3.h5")

    X, Y = dataset["trX"].transpose(2, 1, 0), dataset["trY"]
    ind_list = [i for i in range(3000)]

    #### shuffling the data
    shuffle(ind_list)
    X = X[:, :, ind_list]
    Y = Y[ind_list]

    val_size = int(X.shape[2] *0.1)
    val, test = {}, {}
    val["x"], val["d"] = X[:, :, range(val_size)], Y[range(val_size), :]
    X, Y = X[:, :, range(val_size, X.shape[2])], Y[range(val_size, X.shape[2]), :]
    test["x"], test["d"] = dataset["tstX"].transpose(2, 1, 0), dataset["tstY"]

    params = {}
    
    if alg == "RNN":
        params = run_rnn((X, Y), val, [ 128, 32, 10, 6])
    elif alg == "lstm":
        params = run_lstm((X, Y), val, layer_dims)
    elif alg == "gru":
        params = run_gru((X, Y), val, [ 128, 70, 36, 20, 6])

    cost, acc, confmat = check_acc3(params, test, alg)

    sb.heatmap(confmat, annot=True)
    plt.title("Test Set Confusion Matrix")
    plt.show()
    print(confmat)

    temp = {}
    temp["x"], temp["d"] = X, Y
    
    cost, acc, confmat = check_acc3(params, temp, alg)
    sb.heatmap(confmat, annot=True)
    plt.title("Train Set Confusion Matrix")
    plt.show()
    print(confmat)
    
#run3("gru")
#word_predicter(0.15, 50, 200, .85, (32, 256))
#We, manup_data, params = init_aesystem()
#run_aencoder(We, manup_data, params, 32)

j = ""
while True:
    print("select one of the following:\n1 ---> part 1\n2 -----> part 2\nrnn ----> to run rnn\nlstm ----> to run lstm\ngru ----> to run gru\nQ or q to quit")
    j = input()

    if j == "1":
        We, manup_data, params = init_aesystem()                                                                                                                                                         
        run_aencoder(We, manup_data, params, 32)
    elif j == "2":
        word_predicter(0.15, 50, 200, .85, (32, 256))
    elif j == "rnn":
        run3("RNN")
    elif j == "lstm":
        run3("lstm")
    elif j == "gru":
        run3("gru")
    elif j  != "Q" or j != "q":
        break
