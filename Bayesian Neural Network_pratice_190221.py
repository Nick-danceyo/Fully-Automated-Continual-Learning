# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:56:26 2019

@author: park
"""
import numpy as np
from copy import deepcopy

import torch
from torch.utils import data

dtype = torch.float


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")





"""(1)
사용자 정의 nn 모듈: 기존 모듈의 순차적 구성보다 더 복잡한 모델을 구성해야 할 떄, 
nn.Module의 서브클래스로 새 모듈을 정의하고, 입력 텐서를 받아 다른 모듈 또는 텐서의 
autograd의 연산을 사용하여 출력 텐서를 생성하는 foraward를 정의
"""
class NN(torch.nn.Module):
    
    def __init__(self,D_in, H, D_out):
        
        """
        생성자에게 2개의 nn.Linear 모듈을 생성 (instantiate) 멤버 변수로 지정
        
        이것은 미리 설정해 놓은 함수와도 같은 것
        """
        super(NN, self).__init__()
        #self.linear1 = torch.nn.Linear(D_in,H).to(device,dtype=dtype)#GPU로 해결하도록
        #self.linear2 = torch.nn.Linear(H,D_out).to(device,dtype=dtype)#GPU로 해결하도록
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,D_out)
    
    def forward(self,x):

        """
        순전파 함수에서 입력 데이터의 텐서를 받아서 출력 데이터의 텐서를 반환
        텐서 상의 임의의 연산자 뿐만 아니라 생성자에서 정의한 모듈을 사용 할 수 있음
        """
        
        output_linear1 = self.linear1(x)
        
        output_H_relu = torch.nn.functional.relu(output_linear1)
        #h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(output_H_relu)
        
        return y_pred
    
    
    
"""
N은 배치 크기
D_in은 인풋 차원
H는 은닉 계층의 차원
D_out은 아웃풋 차원
"""
N, D_in, H, D_out = 64, 1000, 100, 10

"""
입력과 출력 텐서 값을 무작위 생성을 통해 얻음
"""
        
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)


"""
앞에서 정의한 클래스를 생성(Instantiat)해서 모델을 구성
"""
model = NN(D_in, H, D_out).cuda()


"""
손실함수와 Optimizer를 만듬. SGD 생성자에서 model.parameters()를 호출 하면
모델의 멤버인 2개의 nnLinear 모듈의 학습 가능한 매개변수들이 포함 됨
"""

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)


"""
epoch 500 학습
"""

for t in range(500):
        
    """
    순전파 단계: 모델에 x를 전달하여 예상하는 yr값을 계산
    """
    y_pred = model(x)
    
    """
    손실을 계산하고 출력
    """

    loss = criterion(y_pred,y)
    print(t,loss.item())
    
    """
    그래디언트 를 0으로 만들고, 역적파 수행, 가중치 업데이트
    """
    optimizer.zero_grad()#clears the gradients of all optimized torch.Tensor
    loss.backward()
    optimizer.step()#Performs a single optimization step (parameter update).


    
"""#########################################################################"""    
"""#########################################################################"""    
"""#########################################################################"""    
"""#########################################################################"""    
"""#########################################################################"""    




"""(2)

위에 기본 사용자 정의 nn 모듈을 방식을 응용하여 Bayesian Neural Network 모델 구축

"""



"""
사용자 정의 nn 모듈: 기존 모듈의 순차적 구성보다 더 복잡한 모델을 구성해야 할 떄, 
nn.Module의 서브클래스로 새 모듈을 정의하고, 입력 텐서를 받아 다른 모듈 또는 텐서의 
autograd의 연산을 사용하여 출력 텐서를 생성하는 foraward를 정의
"""
class NN(torch.nn.Module):
    
    def __init__(self,D_in, H, D_out):
        
        """
        생성자에게 2개의 nn.Linear 모듈을 생성 (instantiate) 멤버 변수로 지정
        
        이것은 미리 설정해 놓은 함수와도 같은 것
        """
        super(NN, self).__init__()
        #self.linear1 = torch.nn.Linear(D_in,H).to(device,dtype=dtype)#GPU로 해결하도록
        #self.linear2 = torch.nn.Linear(H,D_out).to(device,dtype=dtype)#GPU로 해결하도록
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,D_out)
    
    def forward(self,x):

        """
        순전파 함수에서 입력 데이터의 텐서를 받아서 출력 데이터의 텐서를 반환
        텐서 상의 임의의 연산자 뿐만 아니라 생성자에서 정의한 모듈을 사용 할 수 있음
        """
        
        output_linear1 = self.linear1(x)
        
        output_H_relu = torch.nn.functional.relu(output_linear1)
        #h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(output_H_relu)
        
        return y_pred
    
    
    
"""
utilit 모듈을 활용하여 MNIST 데이터 불러오기 및 batch 등을 설정
"""

import torchvision



train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist-data/', train=True, download=True,
                       transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])),
        batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist-data/', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
                       ),
        batch_size=128, shuffle=True)
"""
앞에서 정의한 클래스를 생성(Instantiat)해서 모델을 구성

nn.Module을 활용한 NN 클래스는 자동적으로 모델의 초기값들을 지정해준다.
"""

#We define a standard feedforward NN oif one hidden layer of 1024 units
D_in, H, D_out = 28*28, 1024, 10


"""
뒷 부분에 cuda를 붙여 줌으로써 이를 활용한 아래 부분은 모두 gpu를 사용 할 수 있음!!!
"""
NN_model = NN(D_in, H, D_out).cuda()



"""
Bayesian 모델 구성
In order to make our NN model as a Bayesiaan, we need to put priors on the
parameters and bias. These are distributions thaat represent our prior belief
about resonable valuese for model parameters (before observing)
"""

import pyro
#from pyro.distributions import Normal


"""
In Pyro, the model() function defines how the output data is generated. 
"""
def Bayesian_model(x_data, y_data):
    
    
    # we put priors following independent multivariate normal distribution on the NN's model parameters 
    #Normal은 independent multivariate normal distribution으로 간주하면 된다.
    linear1_weights_prior = pyro.distributions.Normal(loc=torch.zeros_like(NN_model.linear1.weight), scale=torch.ones_like(NN_model.linear1.weight))
    linear1_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(NN_model.linear1.bias), scale=torch.ones_like(NN_model.linear1.bias))

    linear2_weights_prior = pyro.distributions.Normal(loc=torch.zeros_like(NN_model.linear2.weight), scale=torch.ones_like(NN_model.linear2.weight))
    linear2_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(NN_model.linear2.bias), scale=torch.ones_like(NN_model.linear2.bias))

    priors = {'linear1.weights.prior': linear1_weights_prior, 'linear1.bias.prior': linear1_bias_prior, 
              'linear2.weights.prior': linear2_weights_prior, 'linear2.bias.prior': linear2_bias_prior}
    
    
    """
    In order to put the prior on the model parameters, we'll 'lift' the parameters of our NN model to random variables
    We can do this in pyro via random_module(), which effectively takes a given NN.module and turns it into aa distribution
    over the same module;
    Specifically, each parameter in the original NN_model is saampled from form the provided prior.
    """
    
    
    # lilft module parameterse to random variables sampled form the priors
    lifted_module = pyro.random_module("module",NN_model,priors)
    
    #sample a parameter
    lifted_NN_model = lifted_module()
  
    # run the nn forward on data
    # transform    
    lhat = torch.nn.functional.log_softmax(lifted_NN_model(x_data),dim=1)

    
    # condition on the observed data
    #It is equivalent to the distribution that torch.multinomial() samples from.
    #pyro.sample의 역활
    pyro.sample("obs",pyro.distributions.Categorical(logits=lhat),obs=y_data)
    



def guide(x_data, y_data):
    
    
    softplus = torch.nn.Softplus()
    # First layer weight distribution priors
    linear1_weights_mu = torch.randn_like(NN_model.linear1.weight)
    linear1_weights_sigma = torch.randn_like(NN_model.linear1.weight).to(device)
    linear1_weights_mu_param = pyro.param("linear1_weight_mu", linear1_weights_mu).to(device)
    linear1_weights_sigma_param = softplus(pyro.param("linear1_weight_sigma", linear1_weights_sigma))
    linear1_weights_prior = pyro.distributions.Normal(loc=linear1_weights_mu_param, scale=linear1_weights_sigma_param)
    
    # First layer bias distribution priors
    linear1_bias_mu = torch.randn_like(NN_model.linear1.bias)
    linear1_bias_sigma = torch.randn_like(NN_model.linear1.bias)
    linear1_bias_mu_param = pyro.param("lineaar1_bias_mu", linear1_bias_mu)
    linear1_bias_sigma_param = softplus(pyro.param("lineaar1_bias_sigma", linear1_bias_sigma))
    linear1_bias_prior = pyro.distributions.Normal(loc=linear1_bias_mu_param, scale=linear1_bias_sigma_param)
    
    # Output layer weight distribution priors
    linear2_weights_mu = torch.randn_like(NN_model.linear2.weight)
    linear2_weights_sigma = torch.randn_like(NN_model.linear2.weight)
    linear2_weights_mu_param = pyro.param("linear2_weight_mu", linear2_weights_mu)
    linear2_weights_sigma_param = softplus(pyro.param("linear2_weight_sigma", linear2_weights_sigma))
    linear2_weights_prior = pyro.distributions.Normal(loc=linear2_weights_mu_param, scale=linear2_weights_sigma_param).independent(1)
    
    # Output layer bias distribution priors
    linear2_bias_mu = torch.randn_like(NN_model.linear2.bias)
    linear2_bias_sigma = torch.randn_like(NN_model.linear2.bias)
    linear2_bias_mu_param = pyro.param("outb_mu", linear2_bias_mu)
    linear2_bias_sigma_param = softplus(pyro.param("outb_sigma", linear2_bias_sigma))
    linear2_bias_prior = pyro.distributions.Normal(loc=linear2_bias_mu_param, scale=linear2_bias_sigma_param)
    priors = {'linear1.weights.prior': linear1_weights_prior, 'linear1.bias.prior': linear1_bias_prior, 'linear2.weights.prior': linear2_weights_prior, 'linear2.weight.bias': linear2_bias_prior}
    
    lifted_module = pyro.random_module("module", NN_model, priors)
    
    return lifted_module()


from pyro.optim import Adam
optim =  Adam({"lr": 0.01})
svi = pyro.infer.SVI(Bayesian_model, guide, optim, loss=pyro.infer.Trace_ELBO())    

num_iterations = 5
loss = 0


for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        # calculate the loss and take a gradient step
        #아래 데이터 마다 gpu를 붙여줌
        loss += svi.step(data[0].view(-1,28*28).to(device), data[1].to(device))
        print("batch_id ", batch_id)
        
        
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)
