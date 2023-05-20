import tensorflow as tf
from keras.datasets import mnist,cifar10
import numpy as np
import random
import statsmodels.api as sm
# import matplotlib.pyplot as plt
from tensorflow import keras
# from datetime import datetime
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import sys, os
import warnings
import numpy
# import itertools
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import statsmodels.api as sm
from torchvision import datasets, transforms, utils, models
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
import torch.utils.data
from scipy.stats import gaussian_kde
numpy.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
### import from our files
# ! pip install statsmodels
from Net import Discriminator, Net_X, ResNet50
from adeepci import ADeepCI

# define functions
def get_function(func_str):
    if func_str == "abs":
        return (lambda x: (-1+0.4*np.abs(x)).flatten(), 
                lambda x: (-1+0.4*torch.abs(x)).flatten())
    elif func_str == "log":
        return (lambda x: 2*np.log(np.abs(x)).flatten(), 
                lambda x: 2*torch.log(torch.abs(x)).flatten())
    elif func_str == "sin":
        return (lambda x: (0.5+0.5*np.sin(x)).flatten(), 
                lambda x: (0.5+0.5*torch.sin(x)).flatten())
    elif func_str == "none":
        return (lambda x: 0.2*x.flatten(), 
                lambda x: 0.2*torch.Tensor(x).flatten())
    else:
        return (lambda x: np.sign(np.abs(np.abs(x)-5)-2).flatten(), 
                lambda x: torch.sign(torch.abs(torch.abs(x)-5)-2).flatten())

def convert_to_gray(images):
    image_gray_array = []
    for image in images:
        import numpy as np
        from PIL import Image
        print(image.shape)
        # 将彩色图像转换为灰度图像
        image_gray = np.dot(image, [0.2989, 0.5870, 0.1140])
        # 将灰度图像转换为NumPy数组
        image_gray_array.append(image_gray)
    return np.array(image_gray_array)

class ToGray(object):
    def __call__(self, image):
        gray_image = transforms.functional.to_grayscale(image)
        return gray_image


def Simdata(NUM_I,seed,func,rho): #a = y_train/test, b = X_train/test
    
    np.random.seed(seed)    
    X_1_a_j = [] 
    X_2_a_j = [] 
    X_1_a_k = [] 
    X_2_a_k = [] 
    X_1_b_j = [] 
    X_2_b_j = [] 
    X_1_b_k = [] 
    X_2_b_k = []
    Z       = []

    X_2_a_j_t = [] 
    X_2_a_k_t = [] 
    X_2_b_j_t = [] 
    X_2_b_k_t = []

    for i in tqdm(range(0,NUM_I)):
        J = np.random.randint(4,10) # number of choice
        samplea = np.array(random.sample(list(np.arange(a.shape[0])),J)) # a list of index
        samplea = np.expand_dims(samplea, axis=1)
        ej = np.concatenate([a[i] for i in samplea],axis = None) # a list of number on images
        ej = [i for i in ej.tolist()]
        ej = np.float_(ej)

        X_1_a = np.random.uniform(0,2,J) #customer a
        X_2_a = ej
        X_2_a_pic = [torch.Tensor(b[i]) for i in samplea]
        
        
        
        
        sampleb = np.array(random.sample(list(np.arange(a.shape[0])),J)) # a list of index
        sampleb = np.expand_dims(sampleb, axis=1)
        ej = np.concatenate([a[i] for i in sampleb],axis = None) # a list of number on images
        ej = [i for i in ej.tolist()]
        ej = np.float_(ej)
        
        X_1_b = np.random.uniform(0,2,J) #customer b
        X_2_b = ej
        X_2_b_pic = [torch.Tensor(b[i]) for i in sampleb]
        
        
        
        xi  = np.random.normal(0,0.5,J)    # same across all customers
        
        # X_2_a = X_2_a + rho*xi  #customer a endogeneity remove for now
        # X_2_b = X_2_b + rho*xi  #customer b endogeneity remove for now
        
        u_a   = X_1_a + 2*func(X_2_a) + xi + np.random.normal(0,3,J) # \epsilon_{a} # 3 for score
        u_b   = X_1_b + 2*func(X_2_b) + xi + np.random.normal(0,3,J) # \epsilon_{b}
        
        choice_j = np.argmax(u_a) # return the index of product in the sample that customer a chose, we assume customer a as choose j
        choice_k = np.argmax(u_b) # return the index of product in the sample that customer b chose, we assume customer b as choose k
               
        if choice_j == choice_k:
            continue
        else:  

            X_1_a_j.append(X_1_a[choice_j])
            X_2_a_j.append(X_2_a_pic[choice_j])
            X_1_a_k.append(X_1_a[choice_k])
            X_2_a_k.append(X_2_a_pic[choice_k])
            
            X_1_b_j.append(X_1_b[choice_j]) 
            X_2_b_j.append(X_2_b_pic[choice_j]) 
            X_1_b_k.append(X_1_b[choice_k]) 
            X_2_b_k.append(X_2_b_pic[choice_k])
            
            
            
            X_2_a_j_t.append(X_2_a[choice_j])
            X_2_a_k_t.append(X_2_a[choice_k])             
            X_2_b_j_t.append(X_2_b[choice_j]) 
            X_2_b_k_t.append(X_2_b[choice_k])
            #Z.append(np.array([X_1_a[choice_j], X_2_a[choice_j], X_1_a[choice_k], X_2_a[choice_k],X_1_b[choice_j], X_2_b[choice_j], X_1_b[choice_k], X_2_b[choice_k]]))
            Z.append(np.array([X_1_a[choice_j],X_1_a[choice_k],X_1_b[choice_j],X_1_b[choice_k]]))
            
    X_2_a_j = torch.cat(X_2_a_j, out=torch.Tensor(len(X_2_a_j), 28, 28))
    X_2_a_k = torch.cat(X_2_a_k, out=torch.Tensor(len(X_2_a_k), 28, 28))
    X_2_b_j = torch.cat(X_2_b_j, out=torch.Tensor(len(X_2_b_j), 28, 28))
    X_2_b_k = torch.cat(X_2_b_k, out=torch.Tensor(len(X_2_b_k), 28, 28))
    
            
    print("simdata:X_2_a_k:",X_2_a_k.shape)
    return torch.Tensor(X_1_a_j).reshape((-1,1)).double(), X_2_a_j.unsqueeze(1), \
    torch.Tensor(X_1_a_k).reshape((-1,1)).double(), X_2_a_k.unsqueeze(1), torch.Tensor(X_1_b_j).reshape((-1,1)).double(), \
    X_2_b_j.unsqueeze(1), torch.Tensor(X_1_b_k).reshape((-1,1)).double(), X_2_b_k.unsqueeze(1), \
    torch.tensor(Z, dtype=torch.float64),\
    torch.Tensor(X_2_a_j_t).reshape((-1,1)).double(), torch.Tensor(X_2_a_k_t).reshape((-1,1)).double(),\
    torch.Tensor(X_2_b_j_t).reshape((-1,1)).double(), torch.Tensor(X_2_b_k_t).reshape((-1,1)).double()

import torch
from torch.optim import optimizer
def train(epoch):
    all_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        data, target = data.to(device), target.to(device)
        # print(data.shape)
        data = data.float()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # mnist 784 = 28*28 cifar-10 = 3*32*32

        output = model(data)
        cls = output[:, :-1].to(device)
        confident = output[:, -1].to(device)
        # print(cls.shape)
        # print(target.shape)
        loss_0 = F.cross_entropy(cls, target)
        loss_1 = torch.mean(torch.abs(confident - 0.5))# 0.5 is the threshold

        if(loss_1 > 0.5): # means the confident is too high and the model is not confident
            loss = loss_0 + loss_1
        else:# means the confident is too low and the model is confident
            loss = loss_0
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        all_loss += loss.item()
    all_loss /= args.bs # ？
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, all_loss))

def test(model):
    test_loss = 0
    correct = 0
    # print('test_loader: ', len(test_loader))
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        data = torch.flatten(data, start_dim=1)
        # print('shape of target: ', target.shape)

        output = model(data)

        cls = output[:,:-1].to(device) # shape is (batch, 10)
        real = output[:,-1].to(device)# shape is (batch, 1)
        loss_0 = F.cross_entropy(cls, target)
        loss_1 = torch.mean(torch.abs(real - 0.5))
        test_loss += torch.mean(loss_0 + loss_1)
        
        pred = torch.argmax(cls, dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= args.bs
    accuracy = correct / args.bs
    
    print("test_loss: ", test_loss.item(), "accuracy:", accuracy)
if __name__ == '__main__':
    seedNum = 888
    tf.random.set_seed(seedNum)
    np.random.seed(seedNum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(os.getcwd())
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    #  ---------------------
    # parameters
    #  -------------------
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--test_batch_size", type=int, default=200, help="testing batch size")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--cuda", type=bool, default=True, help="use GPU computation")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--log_interval", type=int, default=600, help="interval between image sampling")
    parser.add_argument("--direction", type=str, default='/root/DeepCI-master/notebooks', help="the file you want to run")
    parser.add_argument("--seed", type=int, default=888, help="the seed you want to use")

    # ---------------------
    # parameters for adeepci
    # ---------------------
    
    parser.add_argument("--learner_l2", type=float, default=1e-3, help="learner_l2")
    parser.add_argument("--adversary_l2", type=float, default=1e-3, help="adversary_l2")    
    parser.add_argument("--adversary_norm_reg", type=float, default=1e-3, help="adversary_norm_reg")
    parser.add_argument("--learner_lr", type=float, default=0.0004, help="learner_lr")
    parser.add_argument("--adversary_lr", type=float, default=0.0001, help="adversary_lr")
    parser.add_argument("--n_epoch", type=int, default=2, help="n_epochs")
    parser.add_argument("--bs", type=int, default=100, help="bs")
    parser.add_argument("--train_learner_every", type=int, default=1, help="train_learner_every")
    parser.add_argument("--train_adversary_every", type=int, default=8, help="train_adversary_every")   
    parser.add_argument("--ols_weight", type=float, default=0.02, help="ols_weight")
    parser.add_argument("--warm_start", type=bool, default=True, help="warm_start")
    parser.add_argument("--logger",type=str, default=None, help="logger" )
    parser.add_argument("--model_dir", type=str, default='.', help="model_dir")
    parser.add_argument("--device", type=str, default='cuda', help="device")
    parser.add_argument("--verbose", type=bool, default=False, help="verbose")
    parser.add_argument("--k",type=int, default=256, help="k for net" )
    print(os.getcwd())
    args = parser.parse_args()
    print(args)
    os.chdir(args.direction)
    
    # ---------------------
    # load data mnist or cifar-10
    # ---------------------
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # shape (60000, 28, 28)
    # if gray
    # X_train = convert_to_gray(X_train)
    print(X_train.shape)
    X_train = X_train/255
    # shape (10000, 28, 28)
    X_test = X_test/255
    (example_X_train,example_y_train) = (X_train[:2000], y_train[:2000])
    (example_X_test,example_y_test) = (X_test[:2000], y_test[:2000])

    # get simdata
    a = y_train
    b = X_train
    func_run = "sin"
    kwargs={}
    aa, bb = get_function(func_run)
    X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, X_1_b_j, \
        X_2_b_j, X_1_b_k, X_2_b_k, Z,  X_2_a_j_t, X_2_a_k_t,\
            X_2_b_j_t,  X_2_b_k_t= Simdata(2000, 2, aa, 0.2)
    
    image = X_2_a_k[6]
    # plot the sample
    fig = plt.figure
    plt.imshow(image.squeeze(0), cmap='gray')
    # plt.show()
    print(X_2_a_k.shape)

    # import sys
    # import os
    # sys.path
    # print(sys.path)
    # sys.path.append('/notebooks/AdversarialGMM/')
    
    # define the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    print(torch.cuda.is_available())
    print(device)
    
    
    net_adversary = torch.nn.Sequential(
            # 256
            torch.nn.Linear(4, args.k),
            torch.nn.BatchNorm1d(args.k),
            torch.nn.LeakyReLU(0.2),
            # 256
            torch.nn.Linear(args.k, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            nn.Dropout(p=0.1),                        
            torch.nn.Linear(256, 784),
            torch.nn.Tanh()
            )# input shape (batch, 1, 28, 28), output shape (batch, 1, 28, 28)
    
    # load the data set and create data loader instance
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        # if 1 channel
                        # transforms.Grayscale()
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # if 1 channel
                        # transforms.Grayscale()
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # # 定义灰度转换器
    # gray_transform = transforms.Grayscale()

    # 假设你已经使用DataLoader加载了CIFAR-10数据集并存储在名为 "data_loader" 的变量中

    # 遍历数据集并转换为灰度图像
    # gray_images = []
    # for batch in train_loader:
    #     # 将张量转换为PIL图像
    #
    #     batch = torch.Tensor(np.array(batch))
    #     pil_batch = transforms.ToPILImage()(batch)
    #     # 将彩色图像转换为灰度图像
    #     gray_batch = gray_transform(pil_batch)
    #     gray_images.append(gray_batch)
    # train_loader = gray_images
    import torchvision.transforms as T
    transform = T.Resize((32,32))
    # define the model
    d = torch.Tensor(X_train[:100]).unsqueeze(1)
    # learner = Discriminator()
    learner = ResNet50()
    adversary = net_adversary
    model = learner.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)

    # pretrain net
    for epoch in range(1, 2):
        train(epoch)
        test(model)
        
    # ---------------------
    # log the results
    # ---------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    model_temp = learner #torch.load(os.path.join(res.model_dir,"epoch{}".format(res.n_epochs - 1)))
    X_2_a_k = torch.flatten(X_2_a_k, start_dim=1)
    print("shape of X_2_a_k: ", X_2_a_k.shape)
    test(model_temp)
    pred = model_temp(X_2_a_k.cuda()).cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    print("pred: ", pred.shape)
    print("X_2_a_k_t: ", X_2_a_k_t.shape)
    x = X_2_a_k_t.T.squeeze().cpu().data.numpy()
    # calculate the density
    xy = np.vstack([x,pred])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, pred, c=z, s=100)
    plt.show()
    
    
    # ---------------------
    # the main training 
    # ---------------------
  
    res = ADeepCI(learner, adversary).fit(X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, 
                    X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t,
                    learner_l2=args.learner_l2, adversary_l2=args.adversary_l2, adversary_norm_reg=args.adversary_norm_reg,
                    learner_lr=args.learner_lr, adversary_lr=args.adversary_lr, 
                    n_epochs=args.n_epoch, bs=args.bs, train_learner_every=args.train_learner_every, 
                    train_adversary_every=args.train_adversary_every,
                    ols_weight=args.ols_weight, warm_start=args.warm_start, 
                    logger=args.logger, model_dir=args.model_dir, device = args.device, verbose=args.verbose)
    
    # ---------------------
    # show the results
    # ---------------------

    model_temp = learner #torch.load(os.path.join(res.model_dir,"epoch{}".format(res.n_epochs - 1)))
    X_2_a_k = torch.flatten(X_2_a_k, start_dim=1)
    print("shape of X_2_a_k: ", X_2_a_k.shape)
    test(model_temp)
    pred = model_temp(X_2_a_k.cuda()).cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    print("pred: ", pred.shape)
    print("X_2_a_k_t: ", X_2_a_k_t.shape)
    x = X_2_a_k_t.T.squeeze().cpu().data.numpy()
    # calculate the density
    xy = np.vstack([x,pred])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(x, pred, c = z, s = 100)
    plt.show()
    plt.scatter(X_2_a_k_t.T.squeeze().cpu().data.numpy(), aa(X_2_a_k_t).cpu().data.numpy(), cmap='Spectral_r')# plot the true function
    plt.show()
    

    y = learner(X_2_a_k.cuda()).cpu().data.numpy()
    x = aa(X_2_a_k_t).cpu().data.numpy()
    print('shape of x: ', x.shape)
    print('shape of X_2_a_k_t: ', X_2_a_k_t.shape)
    
    # Fit and summarize OLS model
    pred = np.argmax(y, axis=1)
    mod = sm.OLS(pred,X_2_a_k_t.to(device).cpu().data.numpy())
    print('shape of y: ', y.shape)
    res = mod.fit()
    print(res.summary())
