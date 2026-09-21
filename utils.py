
import os
import gc
import copy
import math
import datetime as dt
#from huggingface_hub import list_user_following
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
#import torch
#from torchinfo import summary
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from data import ddConfig
from globals import *


def tree():
    return defaultdict(tree)

def setup_logging(log_level, log_filename):
    logging.basicConfig(
        level = log_level,
        #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        ]
    )  


def apply(model, dataset):
    """Extract features from the dataset using the given model as pretrained feature extractor.""" 
    model = model.to(device)
    model.eval()
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) #era applicato a batch!!
    features = []
    labels = []
    with torch.no_grad():
        for image, label in tqdm(dataLoader, desc="Extracting embeddings"):
            image, label = image.to(device), label.to(device)
            featurembs = model(image)
            features.append(featurembs.cpu())
            labels.append(label.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0) 
    return  features, labels 

def kfold(dataset:Dataset, n_kfolds):
    outer_folds = [] # train/test balanced k-splits
    targets = np.array(dataset.targets) # type: ignore #tensors[1].detach().cpu().numpy() #labels
    if n_kfolds == 1:
        # single stratified split with constant 1/6 test quota
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_quota, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(targets)), targets))
        outer_folds.append((Subset(dataset, train_idx), Subset(dataset, test_idx)))
    else:
        skf = StratifiedKFold(n_splits=n_kfolds, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):
            outer_folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx)))
    return outer_folds

def nestedSplit(dataset:Dataset, n_kfolds, val_quota=0.1):
    labels = np.array(dataset.targets) # type: ignore #tensors[1].detach().cpu().numpy()
    outer_folds = kfold(dataset, n_kfolds) #train/test folds
    nested_folds = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_quota, random_state=seed) #train/val split
    for fold_idx, (train_outer, test_outer) in enumerate(outer_folds):
        logging.info(f"\nProcessing inner split for outer fold {fold_idx+1}")
        # Extract labels for the train_outer subset
        outer_indices = train_outer.indices
        outer_labels = labels[outer_indices]

        # Perform ONE stratified shuffle split
        (inner_train_idx, inner_val_idx), = sss.split(np.zeros(len(outer_labels)), outer_labels)

        # Map inner indices back to original dataset
        inner_train_indices = np.array(outer_indices)[inner_train_idx]
        inner_val_indices   = np.array(outer_indices)[inner_val_idx]

        inner_train = Subset(dataset, inner_train_indices)
        inner_val   = Subset(dataset, inner_val_indices)

        nested_folds.append((inner_train, inner_val, test_outer))

        logging.info(f"Inner train = {len(inner_train)}, "
            f"inner val = {len(inner_val)}, "
            f"test = {len(test_outer)}")
    return nested_folds
# def split(dataset:Subset, ratio=0.1):
#     data = dataset.indices
#     targets = np.array(dataset.indices) #labels
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
#     (train_idx, val_idx), = sss.split(np.zeros(len(targets)), targets)
#     train_subset = Subset(dataset, train_idx)
#     val_subset  = Subset(dataset, val_idx)  
#     return train_subset, val_subset



def extractFeatures(model, fold):
    train, val, test = fold
    feats, labels = apply(model, train)
    trainf = TensorDataset(feats, labels)
    feats, labels = apply(model, val)
    valf = TensorDataset(feats, labels)
    feats, labels = apply(model, test)
    testf = TensorDataset(feats, labels)
    return (trainf, valf, testf)

def coreset(dataset, subset): # returns the desired data coreset by their indexes
    samples = dataset.tensors[0]
    labels = dataset.tensors[1]
    core_samples, core_labels =  torch.tensor(samples[subset], dtype=torch.float32), torch.tensor(labels[subset], dtype=torch.long)
    coredata = TensorDataset(core_samples, core_labels)
    return coredata
           
from typing import Any, Dict, Literal, Optional
Task = Literal["train", "val", "test"]  
def run(mode:Task, model, dataset, optimizer=None, valset=None, sched=None, keep=False): 
    model.to(device)
    criterion=criterion_lambda()
    Loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0) 
    best_acc, best_loss = .0 , .0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        model.train() if mode=='train' else model.eval()    
        for inputs, labels in tqdm(Loader, desc=f"[Epoch {epoch+1}/{epochs}] {mode}"):
            inputs, labels = inputs.to(device), labels.to(device)              
            with torch.set_grad_enabled(mode == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)    # type: ignore
                if mode=='train': 
                    optimizer.zero_grad(set_to_none=True)           # type: ignore      
                    loss.backward()
                    optimizer.step()                  # type: ignore
                #if mode == 'val' and sched: sched.step() # type: ignore
                epoch_loss += loss.item() * labels.size(0) 
                epoch_correct += outputs.argmax(dim=1).eq(labels).sum().item() 
                epoch_total += labels.size(0)
        epoch_accu = epoch_correct * 100.0 / epoch_total
        epoch_loss /= epoch_total        
        logging.info(f" {mode} Loss: {epoch_loss:.4f} | {mode} Acc: {epoch_accu:.2f}%") 
        if mode != 'train': return epoch_accu, epoch_loss #test or validation needed just one epoch
        # in-train validation mode
        model.train_accuracies.append(epoch_accu); model.train_losses.append(epoch_loss) 
        if sched: sched.step() # advance scheduler at each train epoch
        if valset is not None and epoch%validation_frequency==0: #validate
            val_accu, val_loss = run('val', model, dataset=valset, sched=sched) #Validate with other testset                        
            model.val_accuracies.append(val_accu); model.val_losses.append(val_loss)
            if val_loss > best_loss: 
                best_acc, best_loss = val_accu, val_loss
                if keep is True: model.best = copy.deepcopy(model.state_dict())
    logging.info(f"Training complete. Best acc/loss: {best_acc:.2f}%/{best_loss:.2f}")
    return model

def train(model, trainset, valset=None, save=False):     
    #train and (recursively) validate: returns four lists
    model.train_accuracies, model.train_losses = [], []
    model.val_accuracies, model.val_losses = [], []
    optimizer = optimizer_lambda(model, optimizer_kwargs)
    scheduler = scheduler_lambda(optimizer, scheduler_kwargs)
    model = run('train', model, trainset, optimizer, valset, scheduler, save) 
    return model

def test(model, testset): 
    #test: returns two scalars
    return run('test', model, testset)
        
# NON FUNZIONA SE NON TARI I PARAMETRI ADEGUATAMENTE, E COME????
def momentum(f, target, init = np.random.randn(), eta = 1e-2, mu = 1e-6, max = 1000): 
    # ---- Momentum Gradient Descent ----
    v = 0.0    # initial velocity
    x = init   # initial position
    eps = 1e-6 
    for t in range(max):
        fx = f(x)  # current outcome of the function to be optimized
        loss = (fx - target)**2   # loss
        df_dx = f(x+eps) - f(x-eps) / (2*eps)  # momentum
        g = 2 * (fx -target) * df_dx    # gradient of squared loss
        v = mu * v - eta * g   # update velocity
        x = x + v              # update position        
        logging.info(f"step {t:4d} | x = {x:8.4f} | f(x) = {f(x):8.4f} | loss = {loss:.6f}")
        if loss < eps : break

    logging.info("Converged:")
    logging.info(f"x ≈ {x:.6f}, f(x) ≈ {f(x):.6f}, target = {target}\n")
    return x

# def runModelOnBase(fold, weights_file): # k-th fold
#     train_set, val_set, test_set = fold
#     trains = {'base':{'accu':{},'loss':{}},'coresets':{}}   
#     tests = {'base':{'accu':{},'loss':{}},'coresets':{}}   
#     # FULLY TRAIN/val the fc on full features and test it, as reference
#     #==============================
#     logging.info(f"Training model on FULL DATASET: {nnet.fc_name} on {ds.name} with lr={lr} and batch={batch} for {epochs} epochs")
#     model = nnet().fc(ds.num_classes)  # empty fc model for next use
#     model = train(model, train_set, val_set)
#     trains['base']['accu'] |= {'train 100%': model.train_accuracies}    # type: ignore
#     trains['base']['loss'] |= {'train 100%': model.train_losses}        # type: ignore
#     trains['base']['accu'] |= {'val 100%': model.val_accuracies}       # type: ignore
#     trains['base']['loss'] |= {'val 100%': model.val_losses}            # type: ignore
#     #==============================
#     test_acc, test_loss = test(model, test_set)
#     logging.info(f"--> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
#     tests['base']['accu'] |= {'test 100%': test_acc} #[test_acc for i in range(epochs)]
#     tests['base']['loss'] |= {'test 100%': test_loss} #[ for i in range(epochs)]
#     # logging.info(f"Testing {nnet.fc_name} on {ds.name} with PRETRAINED weights")
#     # model = nnet().fc(ds.num_classes, weights_file)  # pretrained fc model for next use
#     # test_acc, test_loss = test(model, test_set) # test only. returns two scalars: test accuracy and test loss 
#     # logging.info(f"--> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
#     # tests['base']['accu'] |= {'pretrained test': test_acc} #[test_acc for i in range(epochs)]
#     # tests['base']['loss'] |= {'pretrained test': test_loss} #[ for i in range(epochs)]
#     model = None  # free memory
#     return trains, tests

def runModelonCoresets(fold, coresets_file):
    train_set, val_set, test_set = fold
    #trains, tests = tree(), tree()
    trains = {'coresets':{}}; tests = {'coresets':{}}   
    coresets = torch.load(coresets_file)
    for m, s in ((m, s) for m in metrics for s in samplers):
        setting = (m, s) #f'{m} {s}'
        trains['coresets'][setting] = {'accu':{}, 'loss':{}} #add keys for the new setting's trains[k][k]
        tests['coresets'][setting] = {'accu':{}, 'loss':{}} #add keys for the new setting's trains[k][k]
        for r in ratios:                
            coreset_idx = coresets[r][m][s]
            coreset_train = coreset(train_set, coreset_idx)    # prendi solo il coreset selezionato
            #==============================                  
            logging.info(f"Train/Val for model: {nnet.fc_name} on {ds.name} coreset {setting} with compression:{r:.0%}")  
            model = nnet(ds.num_classes).fc(empty=True) # new instance of linear classifier for the current dataset
            #stats = summary(model, input_size=(batch, ds.shape[0], ds.shape[1], ds.shape[2]), verbose=0, device=device) #input_size=(batch, 1, 512)
            #logging.info(stats)
            model = train(model, coreset_train, val_set)                          
            #==============================              
            trains['coresets'][setting]['accu'] |= {f'train {r:.2%}': model.train_accuracies} # type: ignore
            trains['coresets'][setting]['loss'] |= {f'train {r:.2%}': model.train_losses}     # type: ignore
            trains['coresets'][setting]['accu'] |= {f'val {r:.2%}': model.val_accuracies}     # type: ignore
            trains['coresets'][setting]['loss'] |= {f'val {r:.2%}': model.val_losses}         # type: ignore                     
            logging.info(f'Testing of model: {nnet.fc_name} at ratio {r}')
            #==============================
            test_acc, test_loss = test(model, test_set)     #GET THE LAST OR THE BEST?? 
            #==============================
            tests['coresets'][setting]['accu'] |= {f'test {r:.2%}': test_acc} #[test_acc for i in range(epochs)]}
            tests['coresets'][setting]['loss'] |= {f'test {r:.2%}': test_loss} #[test_loss for i in range(epochs)]}
            model = None  # free memory  
    return trains, tests


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



def plot_hystogram(data, bins, title):
    show_distribution(data, bins=512, title="")

def show_distribution(data, bins=512, title=""):
    timestamp = dt.datetime.now().strftime("%y%m%d-%H%M")
    logging.debug(data)
    logging.info(f"{title}: size={len(data)}, mean={np.mean(data)}, std={np.std(data)}, min={np.min(data)}, max={np.max(data)}")   
    # Plot histogram    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(title)
    if log_level == logging.DEBUG : plt.show()
    plt.close()

def print_key_paths(d, root=""):
    for key, value in d.items():
        path = f"{root}.{key}" if root else key
        print(path)
        if isinstance(value, dict):
            print_key_paths(value, path)

def clean_percentage_keys(input_file, output_file):
    import re
    # Load dictionary (change method if not using torch)
    data = torch.load(input_file)
    print_key_paths(data)
    input()   
    cleaned = {'coresets':{}}

    # regex: find "<number>.<number>%", capture only the % without decimals
    pattern = re.compile(r"(\d+)\.\d+%")

    for key1 in data['coresets']: 
        cleaned['coresets']|={key1:{'accu':{}, 'loss':{}}}

        d = data['coresets'][key1]['accu']
        for key, value in d.items():
            # Replace 92.34% → 92%
            new_key = pattern.sub(r"\1%", key)
            cleaned['coresets'][key1]['accu'][new_key] = value

        d = data['coresets'][key1]['loss']
        for key, value in d.items():
            # Replace 92.34% → 92%
            new_key = pattern.sub(r"\1%", key)
            cleaned['coresets'][key1]['loss'][new_key] = value
        
    print_key_paths(cleaned)
    input()    
    # Save cleaned dictionary
    torch.save(cleaned, output_file)

    print("Done. Cleaned keys saved to:", output_file)

def init_axes(axes,r0=YLIMACCU,r1=YLIMLOSS):
    """prepare to Plot training and test accuracies with different colors."""
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(r0)
    axes[0].grid(True)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('avg cross-entropy loss')
    axes[1].set_ylim(r1)
    axes[1].grid(True)
    return axes 

# iterate four line styles 
def style(index = [0], reset=False):                 
    styles = ['-', '--' ]#, '-.', ':']
    if reset : index[0] = 0 #restart cycle
    s = styles[index[0]]     
    index[0] = (index[0]+1) % len(styles)
    return s

def colors(index = [0]):  
    cmap = plt.get_cmap("tab10") #plt.cm.tab10.colors  # This works: tab10 is a ListedColormap with a .colors attribute
    color =  cmap(index[0])
    index[0] = (index[0]+1) % cmap.N
    return color
    
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
    plt.suptitle(title, fontsize=12) #, y=0.97, fontsize=14)  # Use suptitle for the whole figure
    plt.subplots_adjust(left=.05, right=.97, top=.9, bottom=0.16)  # Increase top and bottom margins 
    return plt

def draw_trains(data, setting, end, title):
    plt = draw_trains_(data, setting, end, title)
    m,s = setting   
    plt.savefig(image_file(m, s, end), bbox_inches='tight')
    plt.close() #fig.clf()

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
    plt.suptitle(title, fontsize=12) #, y=0.97, fontsize=14)  # Use suptitle for the whole figure
    #plt.subplots_adjust(left=.05, right=.97, top=.9, bottom=0.16)  # Increase top and bottom margins    
    plt.savefig(image2_file(m, s), bbox_inches='tight')
    plt.close() #fig.clf()

def draw_dds(data, setting, title="K-Fold Doubling Distances Estimates"):
    
    labels = [key for key in data[0][setting]]
    
    foldsDD=[[] for i in range(n_kfolds)]
    
    for k in range(n_kfolds):
        for ratio in data[k][setting]: 
            foldsDD[k].append(data[k][setting][ratio])
                 
    foldsDD = np.array(foldsDD)    
    K, T = foldsDD.shape
    #print(foldsa.shape)
    print(labels)
    x = np.arange(T)
    plt.figure(figsize=(10, 6))
    #fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    plt.title('Accuracy')
    plt.ylabel('Doubling Dim')
    plt.ylim(1, np.max(foldsDD)*1.5)
    plt.grid(True)      
    # Plot each fold
    for k in range(K):
        col = colors()
        plt.scatter(labels, foldsDD[k], s=60, color=col, label=f'k-fold# {k}') #"blue") #alpha=0.3, lw=1.5, label=xlabelsa[k])
    m,s = setting
    #plt.savefig(image2_file(m, s), bbox_inches='tight')
    #plt.show()
    plt.xlabel('ratio')
    plt.legend(loc='lower center', ncol=K) # bbox_to_anchor=(0.5, -0.05)            
    plt.suptitle(title, fontsize=12) #, y=0.97, fontsize=14)  # Use suptitle for the whole figure
    #plt.subplots_adjust(left=.05, right=.97, top=.9, bottom=0.16)  # Increase top and bottom margins    
    plt.savefig(image3_file(m, s), bbox_inches='tight')
    plt.close() #fig.clf()


def computeDoublingDim(dataset, coresets_file, ddEstimator, config:ddConfig=ddConfig()): #normP=math.inf, profile_dd=None):
    train_set, _, _ = dataset
    # if profile_dd is None:
    #     profile_dd = os.getenv("DD_PROFILE", "0").lower() in {"1", "true", "yes", "on"}
    # profiling = lambda: "ON" if profile_dd else "OFF"
    # logging.info(f"RUNNING WITH PROFILING {profiling}")
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"Doubling dimension computations running on {device.type.upper()}.")
    
    #-------------------#          
    if config is None:      #default config
        config.min_centers = 128
        config.max_centers = 4096
        nb_cap = 2024
        max_radii = 192
        #-------------------#
        fps_block = 4096
        cdist_qblock=64       # small query blocks (centers to points)
        cdist_rblock=2048      # larger result blocks (safe for MPS)
        cover_block=1024
        radii=None
        anchor_samples = 256
        radius_quantiles = (0.50, 0.70, 0.85, 0.93)
        seed = 0 
    
    logging.info(f"Configuring Estimator computation based on setup for {config.name.upper()}.")
    
    #------------------------------------------------------------------------------------------
    # Always load metadata on CPU and move tensors lazily per coreset.
    coresets = torch.load(coresets_file, map_location="cpu")  # METTERE DIRETTAMENTE SU MPS???
    #------------------------------------------------------------------------------------------
    dds = {}
    g = torch.Generator(device='cpu')

    for m, s in ((m, s) for m in metrics for s in samplers):
        setting = (m, s)
        dds[setting] = {}

        for r in ratios:
            coreset_idx = coresets[r][m][s]
            coreset_points = coreset(train_set, coreset_idx).tensors[0]
            #------------------------------------------------
            #coreset_points = coreset_points.to(device, dtype=torch.float32, non_blocking=True) #### O VERO????? ???????~~~~########
            #__________________________________________________
            num = len(coreset_idx)
            center_samples = int(min(config.min_centers, num)+r*config.max_centers) #96
            logging.info(f"Computing DD for model: {nnet.fc_name} on {ds.name} coreset {setting} with compression:{r:.00%}")
            logging.info(f"Using {center_samples} random points as centers, {config.max_radii} spaced radiuses per center, early stop at max {config.nb_cap}")
            #logging.info(f"Using max {max_centers, num} random points, {max_radii} spaced radiuses per center, early stop at max {num}")

            coreset_points = coreset_points.detach().to(device=device, dtype=torch.float32)
            n = coreset_points.shape[0]

            if device == 'mps' and hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()

            centers_idx = _farthest_point_sample_streaming(coreset_points, min(center_samples, n), seed=config.seed, rblock=config.fps_block)
            centers = coreset_points[centers_idx]

            g.manual_seed(config.seed + 1)
            # # alternative (random) strategy for centers sampling
            # center_idx = torch.randperm(n, generator=g)[:min(num_centers, n)].to(device)
            # centers = coreset_points[center_idx]

            #compute radii for radiuses sampling
            if config.radii is None:
                anchor_idx = torch.randperm(n, generator=g)[:min(config.anchor_samples, n)].to(device)
                anchors = coreset_points[anchor_idx]
                center_anchor_d = _linf_cdist_blocked(centers, anchors, qblock=config.cdist_qblock, rblock=config.cdist_rblock)
                q = torch.tensor(config.radius_quantiles, device=device, dtype=torch.float32)
                radii_t = torch.quantile(center_anchor_d.flatten(), q).clamp_min(1e-12)
                radii_t = torch.unique(radii_t)
            else:
                radii_t = torch.tensor(list(config.radii), device=device, dtype=torch.float32)

            start = datetime.datetime.now()
         
            res = ddEstimator(
                    coreset = coreset_points,  # your [50000, 512] tensor
                    center_samples=center_samples,        # 96 centers → ~10s runtime on M1 Ultra 32-core
                    anchor_samples=config.anchor_samples,
                    centers_idx=centers_idx,
                    radii= radii_t,
                    max_radii=config.max_radii,       # for radii selection
                    radius_quantiles = config.radius_quantiles,
                    neighborhood_cap=config.nb_cap,     # cap per neighborhood to avoid long covers
                    fps_block=config.fps_block,           # FPS distance chunks
                    cdist_qblock=config.cdist_qblock,         # small query blocks (centers to points)
                    cdist_rblock=config.cdist_rblock,        # larger result blocks (safe for MPS)
                    cover_block=config.cover_block,         # greedy cover chunks
                    device="mps",
                    seed = config.seed,
                    sync_each_center=False,   # disable unless you see memory creep
                )
            
            elapsed = datetime.datetime.now() - start 
            print({'estimate': res.estimate, 'radii': res.radii.tolist() if res.radii is not None else "", 'per_scale_max': res.per_scale_max.tolist() if res.per_scale_max is not None else "", 'device': res.config['device'], 'sec': elapsed})

            dds[setting][r] = res.estimate
            logging.info(f"Processed points amount: {num}")
            logging.info(f"Approx doubling constant λ: {res.per_scale_max}")
            logging.info(f"Approx doubling dimension  : {res.estimate}")
            logging.info(f"Elapsed time: {elapsed}.")

            # Free GPU/MPS memory before moving to the next coreset.
            del coreset_points  #, pairwise_dist            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device.type == "cpu":
                gc.collect()
    return dds

def pick_device(prefer: Optional[str] = None) -> str:
    if prefer is not None:
        return prefer
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def _linf_cdist_blocked(a: torch.Tensor, b: torch.Tensor, qblock: int, rblock: int) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    for i in range(0, a.shape[0], qblock):
        ai = a[i:i+qblock]
        for j in range(0, b.shape[0], rblock):
            bj = b[j:j+rblock]
            out[i:i+qblock, j:j+rblock] = (ai[:, None, :] - bj[None, :, :]).abs().amax(dim=-1)
    return out


def _linf_dist_one_to_many(x: torch.Tensor, Y: torch.Tensor, rblock: int = 2048) -> torch.Tensor:
    out = torch.empty((Y.shape[0],), device=Y.device, dtype=torch.float32)
    for j in range(0, Y.shape[0], rblock):
        yj = Y[j:j+rblock]
        out[j:j+rblock] = (yj - x).abs().amax(dim=-1)
    return out


def _farthest_point_sample_streaming(X: torch.Tensor, m: int, seed: int = 0, rblock: int = 2048) -> torch.Tensor:
    n = X.shape[0]
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=g).item())
    chosen = [first]
    min_dist = _linf_dist_one_to_many(X[first], X, rblock=rblock)
    for _ in range(1, min(m, n)):
        nxt = int(torch.argmax(min_dist).item())
        chosen.append(nxt)
        d = _linf_dist_one_to_many(X[nxt], X, rblock=rblock)
        min_dist = torch.minimum(min_dist, d)
    return torch.tensor(chosen, device=X.device, dtype=torch.long)

#import sys

def show_bar(x, max_value, width=40):
    filled = int(width * x / max_value)
    bar = "█" * filled + " " * (width - filled)
    pct = 100 * x / max_value
    sys.stdout.write(f"\r|{bar}| {pct:6.2f}%")
    sys.stdout.flush()