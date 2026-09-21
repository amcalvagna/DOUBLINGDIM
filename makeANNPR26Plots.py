

import torch
from matplotlib import pyplot as plt
from globals import *
from utils import tree, init_axes, colors, style
end = epochs

alltrains = {} #{i: tree() for i in range(n_kfolds)} #tree() # [{}]*n_kfolds
alltests = {} #{i: tree() for i in range(n_kfolds)}
avgtrains = {}

nnet.fc_name = 'LINEAR[512, 10]'
#trains_file = f'{save_path}{nnet.fc_name}_{ds.name}_trains_{epochs}.pt'

alltrains = torch.load(trains_file())   
#alltests = torch.load(tests_file())    


def AvgTrains(folds, n_kfolds=0):
    tmp = tree()
    # for key in folds[0]['base']['accu']: 
    #     llist = np.array([[folds[k]['base']['accu'][key]] for k in range(n_kfolds)])[0]
    #     tmp['base']['accu'][key] = llist.mean(axis=0)   #llist.std(axis=0)
    #     llist = np.array([[folds[k]['base']['loss'][key]] for k in range(n_kfolds)])[0]
    #     tmp['base']['loss'][key] = llist.mean(axis=0)   #llist.std(axis=0
    for setting in folds[0]['coresets']: 
        for key in folds[0]['coresets'][setting]['accu']:
            llist = np.array([[folds[k]['coresets'][setting]['accu'][key]] for k in range(n_kfolds)])[0]
            tmp['coresets'][setting]['accu'][key] = llist.mean(axis=0)   #llist.std(axis=0)
            llist = np.array([[folds[k]['coresets'][setting]['loss'][key]] for k in range(n_kfolds)])[0]
            tmp['coresets'][setting]['loss'][key] = llist.mean(axis=0)  #llist.std(axis=0
    return tmp  

def draw_trains_(data, setting, end, title):   
    # recall that axes[0] is for accuracies subplot, axes[1] for losses subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    init_axes(axes)   
    # for key in data['base']['accu']: 
    #     s = style()
    #     axes[0].plot(x, data['base']['accu'][key][0:end], 'black', linestyle = s, label = key); 
    #     axes[1].plot(x, data['base']['loss'][key][0:end], 'black', linestyle = s)  
   
    setsize, count, c = 2, 0, colors([0])
    for key in data['coresets'][setting]['loss']: 
        s = style(); c = c if count%setsize else colors()
        n = len(data['coresets'][setting]['loss'][key]) #train and val arrays may differ in length
        x = range(1, end+1) if n==end else np.arange(1, end+1, validation_frequency)
        axes[0].plot(x, data['coresets'][setting]['accu'][key][0:end], color = c, linestyle = s, label = key); 
        axes[1].plot(x, data['coresets'][setting]['loss'][key][0:end], color = c, linestyle = s)      
        count+=1
    fig.legend(loc='lower center', ncol=1+count//2) # bbox_to_anchor=(0.5, -0.05)            
    plt.suptitle(title, fontsize=14) #, y=0.97, fontsize=14)  # Use suptitle for the whole figure
    plt.subplots_adjust(left=.08, right=.955, top=.901, bottom=0.11)  # Increase top and bottom margins 
    return plt

def draw_trains(data, setting, end, title):
    plt = draw_trains_(data, setting, end, title)
    m,s = '-','-' #setting   
    plt.show()
    #plt.savefig(image_file(m, s, end), bbox_inches='tight')
    #plt.close() #fig.clf()

def draw_tests(data, setting, title="K-Fold Test Performance", n_kfolds=n_kfolds):
    """
    folds: list/array of arrays
        folds[k][i] = test performance at step i of fold k
        All folds must have same length.
    """
    labels = [key for key in data[0]['coresets'][setting]['accu']]
    
    foldsacc=[[] for i in range(n_kfolds)]    
    for k in range(n_kfolds):
        for key in data[k]['coresets'][setting]['accu']: 
            foldsacc[k].append(data[k]['coresets'][setting]['accu'][key])
          
    foldsloss=[[] for i in range(n_kfolds)]
    for k in range(n_kfolds):
        for key in data[k]['coresets'][setting]['loss']: 
            foldsloss[k].append(data[k]['coresets'][setting]['loss'][key])
            
    foldsa = np.array(foldsacc)    
    foldsl = np.array(foldsloss)                 # shape: (K, T)
    K, T = foldsa.shape
    #print(foldsa.shape)
    #print(labels)
    # Compute mean and std across folds
    mean_curvea = foldsa.mean(axis=0)
    std_curvea  = foldsa.std(axis=0)
    mean_curvel = foldsl.mean(axis=0)
    std_curvel  = foldsl.std(axis=0)
    #x = np.arange(1,T+1)
    #plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    init_axes(axes, TLIMACCU, TLIMLOSS)
    x = labels 
    # Plot each fold
    for k in range(K):
        col = colors()
        axes[1].scatter(x, foldsl[k], s=60, color=col) 
        axes[0].scatter(x, foldsa[k], s=60, color=col, label=f'k-fold# {k}') 

    m,s = setting
    #plt.savefig(image2_file(m, s), bbox_inches='tight')
    #plt.show()
    axes[0].set_xlabel('ratio')
    axes[1].set_xlabel('ratio')
    fig.legend(loc='lower center', ncol=K) # bbox_to_anchor=(0.5, -0.05)            
    plt.suptitle(title, y=0.97, fontsize=14)  # Use suptitle for the whole figure
    #plt.subplots_adjust(left=.05, right=.97, top=.9, bottom=0.16)  # Increase top and bottom margins
    #plt.show()
    plt.savefig(image2_file(m, s), bbox_inches='tight')
    plt.close() #fig.clf()



# get a single average plot from many k-fold coresets plots
avgtrains = AvgTrains(alltrains, n_kfolds) if n_kfolds>1 else alltrains[0]

for setting in ((m, s) for m in metrics for s in samplers):
    m, s = setting
    logging.info(f"Drawing {image_file(m, s, end)}")
    title = f'Training of {nnet.fc_name} classifier head on {ds.name} coresets'
    draw_trains(avgtrains, setting, end, title)                    
    #draw_tests(alltests, setting, title, n_kfolds)

