import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_MNIST import model_MNIST
from LeNet5 import Model_LeNet5
import time
import matplotlib.pyplot as plt
import torchattacks
from functools import partial
from adv_lib.attacks import alma, ddn,fmn,fab
from adv_lib.utils.attack_utils import run_attack
import random
from PFPMA import PFPMA_L2,PFPMA_L1_ISTA

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

random_seed = 1
torch.manual_seed(random_seed)
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

seed_torch()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
model=model_MNIST()
model.to(device)
criterion=nn.CrossEntropyLoss()
batch_size=512
list_para=[
{'model':'LeNet5','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.999,'lam':1,'decay':0.3},
{'model':'LeNet5','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.2,'UseRMS':True,'rho':0.99,'lam':1,'decay':0.3},
{'model':'Standard','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.999,'lam':1,'decay':0.3},
{'model':'Standard','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':1,'decay':0.3},
{'model':'ddn','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.999,'lam':10,'decay':0.3},
{'model':'ddn','attack':'PFPMA','p_norm':'L2','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':100,'decay':0.8},

{'model':'LeNet5','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':0.1,'decay':0.3},
{'model':'LeNet5','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':0.1,'decay':0.3},
{'model':'Standard','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.999,'lam':0.1,'decay':0.3},
{'model':'Standard','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.999,'lam':0.1,'decay':0.3},
{'model':'ddn','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'Max','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':0.1,'decay':0.3},
{'model':'ddn','attack':'PFPMA','p_norm':'L1','outloop_num':3,'innerloop_num':150,'stepsize':0.01,'penalty_type':'MaxSquare','batch_size':batch_size,'beta':0.0,'UseRMS':True,'rho':0.99,'lam':1,'decay':0.3},

{'model':'LeNet5','attack':'DeepFool','iter_max':100,'batch_size':batch_size},
{'model':'Standard','attack':'DeepFool','iter_max':100,'batch_size':batch_size},
{'model':'ddn','attack':'DeepFool','iter_max':100,'batch_size':batch_size},

{'model':'LeNet5','attack':'CW','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},
{'model':'Standard','attack':'CW','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},
{'model':'ddn','attack':'CW','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},

{'model': 'LeNet5','attack':'DDN','batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'DDN','batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'DDN','batch_size':batch_size ,'iter_num':1000},

## fab_L2 in torch attack(its FAB_L1 consistently fails)
{'model':'LeNet5','attack':'FAB_torchattack','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},
{'model':'Standard','attack':'FAB_torchattack','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},
{'model':'ddn','attack':'FAB_torchattack','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},

{'model': 'LeNet5','attack':'EAD','batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'EAD','batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'EAD','batch_size':batch_size ,'iter_num':1000},

{'model':'LeNet5','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.01},
{'model':'Standard','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},
{'model':'ddn','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},

#fab L1 in adversarial library of rony
{'model':'LeNet5','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},
{'model':'Standard','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},
{'model':'ddn','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},
    ]
for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    if item['model']=='LeNet5':
        model = Model_LeNet5()
        model.load_state_dict(torch.load('./models/mnist/MNIST_LeNet_onlyweight.pth'), False)
    else:
        model = model_MNIST()
        if item['model'] == 'Standard':
            model.load_state_dict(torch.load('./models/mnist/mnist_regular.pth'), False)
        elif item['model'] == 'ddn':
            model.load_state_dict(torch.load('../models/mnist/mnist_robust_ddn.pth'), False)  # ALM paper github : l2 adversarially trained
    model.to(device)
    model.eval()  # turn off the dropout
    test_data = torch.unsqueeze(test_dataset.data, dim=1)
    test_labels = test_dataset.test_labels.to(device)
    test_data_normalized = test_data / 255.0
    test_data_normalized = test_data_normalized.to(device)
    outputs = model(test_data_normalized)
    _, labels_predict = torch.max(outputs, 1)
    correct = torch.eq(labels_predict, test_labels)
    correct_sum = correct.sum()
    correct_index=[]
    for i in range(10000):
        if correct[i]:
            correct_index.append(i)
    print('clean accuracy is:', correct_sum / 10000.0)
    start_time = time.time()
    if item['attack'] == 'PFPMA':
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            if item['p_norm']=='L2':
               adv_images=PFPMA_L2(model,images, labels,outloop_num=item['outloop_num'],innerloop_num=item['innerloop_num'],StepSize=item['stepsize'],penalty_type=item['penalty_type'],beta=item['beta'],UseRMS=item['UseRMS'],rho=item['rho'],lam=item['lam'],decay=item['decay'])
               perturbation = torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1)
            elif item['p_norm']=='L1':
               adv_images=PFPMA_L1_ISTA(model,images, labels,outloop_num=item['outloop_num'],innerloop_num=item['innerloop_num'],StepSize=item['stepsize'],penalty_type=item['penalty_type'],beta=item['beta'],UseRMS=item['UseRMS'],rho=item['rho'],lam=item['lam'],decay=item['decay'])
               perturbation = torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1)
            outs=model(adv_images)
            _, labels_predict = torch.max(outs, 1)
            success = (labels_predict != labels)
            list_success_fail=list_success_fail+success.tolist()
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'DDN':
            method = partial(ddn, steps=item['iter_num'])
            attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(),
                                     labels=test_labels[correct_index].cpu(), attack=method,
                                     batch_size=item['batch_size'])
            outs = model(attack_data['adv_inputs'].to(device))
            _, labels_predict = torch.max(outs, dim=1)
            success = (labels_predict != test_labels[correct_index])
            list_success_fail = list_success_fail + success.tolist()
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=2, dim=1)
            list_pert = list_pert + perturbation.tolist()
            #print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'FMN':
        method = partial(fmn, norm=item['p_norm'], steps=item['steps'], γ_init=item['γ_init'], α_init=item['α_init'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(),labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs = model(attack_data['adv_inputs'].to(device))
        _, labels_predict = torch.max(outs, dim=1)
        list_success_fail = ~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 0:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=0, dim=1)
        elif item['p_norm'] == 1:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 2:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=2, dim=1)
        elif item['p_norm'] == float('inf'):
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=float('inf'), dim=1)
        list_pert = list_pert + perturbation.tolist()
        print('perturbation is: ', perturbation)
        print('avg_pert is: ', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'FAB_advlib':
        method = partial(fab, norm=item['p_norm'], n_iter=item['steps'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(),labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        outs = model(attack_data['adv_inputs'].to(device))
        _, labels_predict = torch.max(outs, dim=1)
        list_success_fail = ~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 1:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 2:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=2, dim=1)
        elif item['p_norm'] == float('inf'):
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=float('inf'), dim=1)
        list_pert = list_pert + perturbation.tolist()
        print('perturbation is: ', perturbation)
        print('avg_pert is: ', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'EAD':
        list_const = []
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            adv_images = atk(images, labels)
            outs = model(adv_images)
            _, labels_predict = torch.max(outs, 1)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            perturbation = torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert = list_pert + perturbation.tolist()
            #print('perturbation is: ', torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'DeepFool':
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            atk = torchattacks.DeepFool(model, steps=item['iter_max'])  # torchattack
            adv_images = atk(images, labels)
            outs = model(adv_images)
            _, labels_predict = torch.max(outs, 1)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            perturbation = torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert = list_pert + perturbation.tolist()
            print('perturbation is: ', torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'FAB_torchattack':  #FAB_L2
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            atk = torchattacks.FAB(model, norm=item['p_norm'], steps=item['steps'], eps=item['eps'],
                                   n_classes=10)  # torchattack
            adv_images= atk(images, labels)
            outs = model(adv_images)
            _, predict_labels = torch.max(outs, dim=1)
            success_fail = ~torch.eq(predict_labels, labels)
            list_success_fail = list_success_fail + success_fail.tolist()
            if item['p_norm'] == 'L2':
                perturbation = torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            elif item['p_norm'] == 'Linf':
                perturbation = torch.norm((images - adv_images).view(len(images), -1), p=float('inf'), dim=1)
            list_pert = list_pert + perturbation.tolist()
            #print('perturbation is: ', torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'CW':
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            atk = torchattacks.CW(model, steps=item['iter_max'], lr=item['lr'], c=item['c'])  # torchattack
            adv_images = atk(images, labels)
            #adv_images, c, predict_labels = atk(images, labels)
            outs = model(adv_images)
            _, labels_predict = torch.max(outs, 1)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            perturbation = torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert = list_pert + perturbation.tolist()
            #print('perturbation is: ', torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
    end_time = time.time()
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    list_pert_success=[ pert  for pert,result in zip(list_pert,list_success_fail) if result]
    avg_pert=sum(list_pert_success)/len(list_pert_success)
    print('avg_pert is',avg_pert)
    dict_save={'device':device,'para':item,'time_used':time_used,'list_success_fail':list_success_fail,'attack_success_rate':attack_success_rate,'list_pert':list_pert,'avg_pert':avg_pert}

    if 'PFPMA' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_PenalyType{}_lam{}_decay{}_stepsize{}_beta{}_UseRMS{}_rho{}.pt'.format(item['model'],item['attack'],item['p_norm'],item['penalty_type'],item['lam'],item['decay'],item['stepsize'],item['beta'],item['UseRMS'],item['rho']))
    elif 'DDN' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'EAD' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'DeepFool' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_itermax{}.pt'.format(item['model'], item['attack'], item['iter_max']))
    elif 'FAB_torchattack' == item['attack']:
        torch.save(dict_save, './result/mnist/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps']))
    elif 'FAB_advlib' == item['attack']:
        torch.save(dict_save, './result/mnist/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps']))
    elif 'CW' in item['attack']:
        torch.save(dict_save, './result/mnist/{}_attack_{}_IterMax{}_lr{}.pt'.format(item['model'], item['attack'],item['iter_max'], item['lr']))
    elif 'FMN' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_steps{}_gammaini{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps'],item['γ_init']))


