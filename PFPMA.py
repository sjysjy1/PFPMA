import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

def sign_fun(x):
    cond=x<0
    y=torch.full_like(x,1,device=x.device)
    y[cond]=0
    return y


def calculating_grad(model,images,labels):
    images.requires_grad_()
    model.zero_grad()
    outs = model(images)
    one_hot_labels = torch.eye(outs.shape[1]).to(images.device)[labels]
    other = torch.max((1 - one_hot_labels) * outs, dim=1)[0]
    real = torch.max(one_hot_labels * outs, dim=1)[0]
    outs_diff = (real-other)
    outs_diff.sum().backward()
    return outs_diff

def PFPMA_L2(model:nn.Module,
             images:Tensor,
             labels:Tensor,
             penalty_type: str = 'MaxSquare',
             outloop_num:int =4,
             innerloop_num:int=250,
             StepSize:float=0.01,
             rho:float=0.99,
             beta:float=0,
             UseRMS:bool=True,
             lam:float=1,
             decay:float=0.5,
):
    device=images.device
    images_ori = images.clone().detach().to(device)
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    #initializing variables
    lower_bound=torch.full((len(images),),0,device=device)
    best_adv = images.clone().detach().to(device)
    best_adv_norm = torch.full((len(images), ), float('inf'), device=device)
    RMS_avg=torch.zeros_like(images)
    grad_momentum=torch.zeros_like(images)
    #a step to find a initial point that is not the original, otherwise attack is stuck as gradient is zero
    outs_diff=calculating_grad(model,images,labels)
    images = torch.clamp(images - StepSize * images.grad, min=0, max=1).detach()
    # pre-calculating the gradient before outerloop
    outs_diff=calculating_grad(model,images,labels)
    #outer loop
    for i in range(outloop_num):
        #inner loop
        for j in range(innerloop_num):
             with torch.no_grad():
                 obj_minus_bound=torch.norm((images-images_ori).view(len(images),-1),p=2,dim=1)**2-lower_bound
                 if penalty_type=='MaxSquare':
                    grad_obj=2*obj_minus_bound[:,None,None,None] * 2*(images - images_ori) * sign_fun(obj_minus_bound)[:,None,None,None]
                 elif penalty_type=='Max':
                    grad_obj=torch.ones_like(images)* 2*(images - images_ori)  * sign_fun(obj_minus_bound)[:,None,None,None]
                 if penalty_type=='MaxSquare':
                    grad_penalty=lam*2 *outs_diff[:,None,None,None] * images.grad * sign_fun(outs_diff)[:,None,None,None]
                 elif penalty_type=='Max':
                    grad_penalty=lam*torch.ones_like(images) * images.grad * sign_fun(outs_diff)[:,None,None,None]
                 grad=grad_obj+grad_penalty
                 grad_momentum = beta * grad_momentum + grad
                 if UseRMS:
                    RMS_avg = rho * RMS_avg + (1 - rho) * grad_momentum ** 2
                    step_size=StepSize/(RMS_avg**0.5+1e-8)
                 else:
                    step_size=StepSize
                 images = torch.clamp(images - step_size * grad_momentum, min=0.0, max=1.0)
             outs_diff=calculating_grad(model,images,labels)
             with torch.no_grad():
                 norm = torch.norm((images - images_ori).view(len(images), -1), p=2, dim=1)
                 is_adv = (outs_diff < 0)
                 is_better_adv = is_adv & (norm < best_adv_norm)
                 #best_adv = torch.where(is_better_adv[:,None,None,None], images.detach(), best_adv)
                 best_adv = torch.where(is_better_adv[:, None, None, None], images, best_adv)
                 best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
        lower_bound=decay*torch.norm((images - images_ori).view(len(images), -1), p=2, dim=1)**2
    return best_adv

def ISTA_fun(x,x_ori,lamb):
    idx1=torch.ge(x-x_ori,lamb)
    idx2 = torch.le(x-x_ori, -lamb)
    idx3=torch.ge(x-x_ori, -lamb) & torch.le(x-x_ori, lamb)
    x[idx1] = x[idx1]-lamb[idx1]
    x[idx2] = x[idx2]+lamb[idx2]
    x[idx3] = x_ori[idx3]
    return x

# L1 attack using ISTA
def PFPMA_L1_ISTA(model:nn.Module,
             images: Tensor,
             labels: Tensor,
             penalty_type: str = 'MaxSquare',
             outloop_num: int = 4,
             innerloop_num: int = 250,
             StepSize: float = 0.01,
             beta: float = 0,
             UseRMS: bool = True,
             rho: float = 0.99,
             lam:float=1,
             decay:float=0.5,
):
    device=images.device
    images_ori = images.clone().detach().to(device)
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    #initializing variables
    bound=torch.full((len(images),),0.0,device=device)
    best_adv = images.clone().detach().to(device)
    best_adv_norm = torch.full((len(images), ), float('inf'), device=device)
    RMS_avg=torch.zeros_like(images)
    grad_momentum=torch.zeros_like(images)
    #a step to find a initial point that is not the original, otherwise attack is stuck as gradient is zero
    outs_diff=calculating_grad(model,images,labels)
    images = torch.clamp(images - StepSize * images.grad, min=0, max=1).detach()
    # pre-calculating the gradient before outerloop
    outs_diff=calculating_grad(model,images,labels)

    #outer loop
    for i in range(outloop_num):
        #inner loop
        for j in range(innerloop_num):
             with torch.no_grad():
                 obj_minus_bound=torch.norm((images-images_ori).view(len(images),-1),p=1,dim=1)-bound
                 if penalty_type=='MaxSquare':
                    grad_penalty=lam*2 *outs_diff[:,None,None,None] * images.grad * sign_fun(outs_diff)[:,None,None,None]
                 elif penalty_type=='Max':
                    grad_penalty=lam*torch.ones_like(images) * images.grad * sign_fun(outs_diff)[:,None,None,None]

                 grad_momentum = beta * grad_momentum + (1-beta)*grad_penalty

                 if UseRMS:
                    RMS_avg = rho * RMS_avg + (1 - rho) * grad_momentum ** 2
                    step_size=StepSize/(RMS_avg**0.5+1e-8)
                    select = (sign_fun(obj_minus_bound) == 0)
                    images[select] = images[select] - step_size[select] * grad_momentum[select]   #for those max{0,||x||-L}=0
                 else:
                    step_size = torch.full_like(images, StepSize, device=device)
                    select = (sign_fun(obj_minus_bound) == 0)
                    images[select] = images[select] - step_size[select] * grad_momentum[select]
                 if torch.any(~select):
                    images[~select]  = (images[~select]  - step_size[~select]  * grad_momentum[~select] )   #for those max{0,||x||} != 0
                    images[~select] = torch.clamp(ISTA_fun(images[~select], images_ori[~select], step_size[~select]), min=0.0, max=1.0).detach()
             outs_diff=calculating_grad(model,images,labels)
             with torch.no_grad():
                 norm = torch.norm((images - images_ori).view(len(images), -1), p=1, dim=1)
                 is_adv = (outs_diff < 0)
                 is_better_adv = is_adv & (norm < best_adv_norm)
                 best_adv = torch.where(is_better_adv[:, None, None, None], images, best_adv)
                 best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
        bound=decay*torch.norm((images - images_ori).view(len(images), -1), p=1, dim=1)
    return best_adv










