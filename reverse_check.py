#!python3
from copy import deepcopy
import time
from sys import argv
from genericpath import exists
from coresets import compute_scales, make_coresets
from dataset import CoreDataset
from utils import *



# -----------------------------------------------
# -----------------------------------------------
if __name__ == '__main__':
    np.random.seed(seed) # for reproducibility
    torch.manual_seed(seed) # for reproducibility
    end = int(argv[2]) if len(argv)>2 else epochs #stop epoch for drawing accuracies
    if end > epochs: epochs = end
    
    # -------------------------------
    # prepare logging
    # -------------------------------

    setup_logging(logging.INFO, logfile)  
    logging.info(f"Running: {argv[0]}...")
    logging.info(f"Hyper parameters:\n {hypm}.")

    #=================================================================
    # 
    #=================================================================
    #datasetKfolds = nestedSplit(ds.combined(), n_kfolds, val_quota) # internal kfold   
    #train_dataset, val_dataset, test_dataset = datasetKfolds[0] #ds.dataset()     
    train_dataset, test_dataset = ds.dataset()     
    net = nnet(out = ds.num_classes, pretrain = pretrained_weights)
    
    #train the entire model (backbone+fc)
    if not exists(f"{root}save/pretr_mdl.pt"): 
        model = net.fullnet()    
        model = train(model, train_dataset, test_dataset, save=True)
        #print("COMPUTED \n", pretr_mdl.fc.state_dict())
        torch.save(model.best, f"{root}save/pretr_mdl.pt") # type: ignore

    pretr_mdl = deepcopy(net.fullnet(f"{root}save/pretr_mdl.pt")).to(device) # load backbone
    pretr_fc = deepcopy(net.fc())  #print("LOADED \n", pretr_mdl.fc.state_dict())
    
    #extract the features
    pretr_back = deepcopy(net.backbone()).to(device) # return current trained backbone
    if not exists(f"{root}save/features.pt"):        
        fdata, flabels = apply(pretr_back, train_dataset) #flabels fasulle, copia di quelle originali
        features  = TensorDataset(fdata, flabels)
        torch.save(features, f"{root}save/features.pt")
    else: features = torch.load(f"{root}save/features.pt") #, weights_only=True)
           
    #make coresets (make sure only one metric and sampler are set in hyperparams)
    features_np = features.tensors[0].numpy()
    if not exists(f"{root}data/scales.pt"):
        scales = compute_scales(features_np)
        torch.save(scales, f"{root}data/scales.pt")
    else: scales = torch.load(f"{root}data/scales.pt")

    if not exists(f"{root}data/coresets.pt"):
        coresets = make_coresets(features_np, scales)
        torch.save(coresets, f"{root}data/coresets.pt")
    else: coresets = torch.load(f"{root}data/coresets.pt")

    # # train the classifier
    # if not exists(f"{root}fc.pt"): 
    #     train(fc, features)
    #     torch.save(fc.state_dict(), f"{root}fc.pt")
    # else: fc.load_state_dict(torch.load(f"{root}fc.pt"))

    m, s = metrics.pop(), samplers.pop()
    # one data consistency test per coreset
    
    np.random.seed()
    for r in coresets: 
        cs  = coresets[r][m][s]
        index = 35#np.random.randint(0, len(cs))
        
        #labels are saved correctly
        my_feat, my_label = features.tensors[0][cs[index]], features.tensors[1][cs[index]]
        img_data, img_label = train_dataset.__getitem__(cs[index])
        assert img_label == my_label.item(), f"Expected {img_label}, got {my_label}"  #the data label correspond the feature label            
        
        pretr_back.eval()
        my_feat = my_feat.to(device)
        img = img_data.unsqueeze(0).to(device)    
        with torch.no_grad(): 
            your_feat = pretr_back(img).squeeze(0) #flatten?
        assert all(your_feat == my_feat), f"Expected {my_feat} == {your_feat} \n {pretr_fc.state_dict()}"   #the img correspond to the right feature
        
        pretr_fc.eval()
        with torch.no_grad():
            pred1 = pretr_fc(pretr_back(img))
            pred2 = pretr_mdl(img) 
        assert all(pred1 == pred2), f"Expected {pred1}, got {pred2}, \n {pretr_fc.state_dict()}"   #the predicted label correspond to the original label            

        
        
        
        
    logging.info("Looks good!")

    # check test and training performance on coresets
    setting = (m, s)
    testdata = [{"coresets": {(m,s): {"accu": {},"loss": {}}}}]
    traindata = [{"coresets": {(m,s): {"accu": {},"loss": {}}}}]
    testdata.append({"coresets": {(m,s): {"accu": {},"loss": {}}}}) 
    traindata.append({"coresets": {(m,s): {"accu": {},"loss": {}}}})
        
    if not exists(f"{root}save/testdata.pt"):
        for r in coresets: 
            cs  = coresets[r][m][s]
            ctrain = Subset(train_dataset, cs) #CoreDataset(train_dataset, cs)
            net = nnet(ds.num_classes).fullnet() #.to(device)    
            model = train(net, ctrain, test_dataset)                                        ###--------->TRAIN
            traindata[0]['coresets'][setting]['accu'] |= {f'train {r:.2%}': model.train_accuracies} # type: ignore
            traindata[0]['coresets'][setting]['loss'] |= {f'train {r:.2%}': model.train_losses}     # type: ignore
            traindata[0]['coresets'][setting]['accu'] |= {f'val {r:.2%}': model.val_accuracies}     # type: ignore
            traindata[0]['coresets'][setting]['loss'] |= {f'val {r:.2%}': model.val_losses}    # type: ignore
            test_acc, test_loss = test(net, test_dataset)                          
            testdata[0]['coresets'][(m,s)]['accu'] |= {f'test {r:.2%}': test_acc} 
            testdata[0]['coresets'][(m,s)]['loss'] |= {f'test {r:.2%}': test_loss} 
            logging.info(f"Training complete for ratio {r:.2%}. Test acc/loss: {test_acc:.2f}%/{test_loss:.2f}")
 
            # check post training performance of a random core set rcs
             # add space to host next run for random coresets data
        
        for r in coresets: 
            rcs  = np.random.randint(0, len(train_dataset), size=len(coresets[r][m][s]))
            ctrain = CoreDataset(train_dataset, rcs)
            net = nnet(ds.num_classes).fullnet()    
            model = train(net, ctrain, test_dataset)   #added validation dataset...                                   ###--------->TRAIN
            traindata[1]['coresets'][setting]['accu'] |= {f'train {r:.2%}': model.train_accuracies} # type: ignore
            traindata[1]['coresets'][setting]['loss'] |= {f'train {r:.2%}': model.train_losses}     # type: ignore
            traindata[1]['coresets'][setting]['accu'] |= {f'val {r:.2%}': model.val_accuracies}     # type: ignore
            traindata[1]['coresets'][setting]['loss'] |= {f'val {r:.2%}': model.val_losses}    # type: ignore
            test_acc, test_loss = test(net, test_dataset)
            testdata[1]['coresets'][(m,s)]['accu'] |= {f'test {r:.2%}': test_acc} 
            testdata[1]['coresets'][(m,s)]['loss'] |= {f'test {r:.2%}': test_loss} 
            logging.info(f"Random complete for ratio {r:.2%}. Test acc/loss: {test_acc:.2f}%/{test_loss:.2f}")
            
        torch.save(testdata, f"{root}save/testdata.pt")
        torch.save(traindata, f"{root}save/traindata.pt")
    else: 
        testdata = torch.load(f"{root}save/testdata.pt") 
        traindata = torch.load(f"{root}save/traindata.pt")    
   
    # take a coreset of features and get back their corresponding original data in dataset
    # as features are saved in the feature dataset in the same order of extraction 
    # their index correspond to the index of their originating image. 
    # ckeck that reapplying the image with that index originates a feature identical with the one with that index
    
    # take the original dataset and select the image with same index (datasets are never shuffled at reading)
    # instantiate the pretrained model and apply the data to obtain a feature
    # compare obtained feature with extracted feature 

    elapsed = datetime.datetime.now() - startuptime
    
    logging.info(f"processing took:  {elapsed.total_seconds():.2f} seconds ({elapsed.total_seconds()/60:.2f}) minutes).")
    logging.info(f"Drawing {image_file(m, s, end)}")
    title = f'{ds.name} {nnet.fc_name} {m} {s}'
    n_kfolds = 2 # the second is the random case
    plt = draw_trains_(traindata[0], (m,s), end, title)
    logging.info(f"Drawing {image_file(m, s, end)}epsilon.png")
    plt.savefig(image_file(m, s, end)+"epsilon.png", bbox_inches='tight')
    plt.close() #fig.clf()
    plt = draw_trains_(traindata[1], (m,s), end, title)
    logging.info(f"Drawing {image_file(m, s, end)}random.png")
    plt.savefig(image_file(m, s, end)+"random.png", bbox_inches='tight')
    plt.close() #fig.clf()
    draw_tests(testdata, (m,s), title, n_kfolds)
