#!python3
import json
from sys import argv
from genericpath import exists
from torchinfo import summary

from coresets import compute_scales, make_coresets
#from doublingGreedyMLX import computeDoublingDim
#from doubling.doubling_dim_linf_mps import ddEstimator, config
from doubling.ddLinfMixed import ddEstimator, config
from utils import *


# -----------------------------------------------
# TRAIN AND TEST ONLY THE FC LAYER OF RESNET18 ON CIFAR-10 FEATURES
# want to test three FC versions:  pretrained on CIFAR-10, trained on coreset features, trained on fullSet features 
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
    # kfolding on train/test data, and nested split on train/val 
    #=================================================================
    #train_dataset, test_dataset = ds.dataset() 
    datasetKfolds = nestedSplit(ds.combined(), n_kfolds, val_quota) # internal kfold   
    #datasetKfolds = [(train_dataset, test_dataset, test_dataset)]
    #returns a list of tuples of Subsets objects
    #=================================================================


    #=================================================================
    # TRAIN the nnet o dataset train/val(test) and save best weights:
    #=================================================================   
    for f, (train_set, val_set, test_set) in enumerate(datasetKfolds):
        if not exists(pretrained_models(f)): 
            model = nnet(ds.num_classes, pretrained_weights).fullnet() #train the entire model (backbone+fc)
            #logging.debug(model) 
            stats = summary(model, input_size=(batch, ds.shape[0], ds.shape[1], ds.shape[2]), verbose=0, device=device)
            logging.info(stats)
            logging.info(f"TRAINING: Training {nnet.name} full net {nnet.num_classes} classes on {ds.name} dataset fold {f}...")        
            train(model, train_set, val_set, save=True)          
            torch.save(model.best, pretrained_models(f)) #weights_only=True 
            logging.info(f"Saved pretrained weights of {nnet.name} in {pretrained_models(f)}...")
            
            # SANITY CHECK : re-test on dataset with the saved weights to check it works
            if log_level == logging.DEBUG: 
                logging.debug(f"testing saved/loaded model: {pretrained_models(f)}")
                net = nnet(ds.num_classes).fullnet(pretrained_models(f)) # reload on empty
                test(net, test_set)   
            logging.info(f"TRAINING {f+1}/{n_kfolds} Done.")
    #=================================================================


    # now each dataset fold has its own pretrained model
    #=================================================================
    # GET FEATURES DATASET KFOLDS from corresponding pretrained model
    #=================================================================   
    featureKfolds = []
    model = nnet(ds.num_classes)  #always run to initialize name fields
    for f, fold in enumerate(datasetKfolds): 
        if exists(features_file(f)): featureKfolds.append(torch.load(features_file(f)))
        else:
            model = model.backbone(pretrained_models(f)) #take backbone only
            #model = nnet(ds.num_classes).backbone(pretrained_models(f))
            features = extractFeatures(model, fold)
            featureKfolds.append(features)
            torch.save(features, features_file(f))  
            logging.info(f"Saved feature embeddings of {ds.name} dataset fold {f}/{n_kfolds} in {features_file(f)}...")
    logging.info(f"FEATURE EXTRACTION Done.")
    #=================================================================

    #========================COMPUTE SCALES FOR DD SAMPLING=========================================
    scales = []
    for f, (train_set, _, _) in enumerate(featureKfolds): 
        if not exists(metafile(f)): 
            data_np = train_set.tensors[0].squeeze().numpy() 
            scales.append(compute_scales(data_np))  # also  checks and, if required, creates metadata metafile       
            metadata={"dataset":ds.name, "model":nnet.name, "ratios":ratios, 'scales':scales[f]}
            with open(metafile(f),'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)            
                logging.info(f"Saved updated Metadata...in {metafile(f)}")       
        else :
            with open(metafile(f),'r', encoding='utf-8') as f: 
                metadata = json.load(f) 
                scales.append(metadata["scales"])
                logging.info(f"Loaded fold {f}/{n_kfolds} precomputed grid scales {scales} for target ratios {ratios} ")
    #=================================================================

    # # SOME SANITY CHECKS
    # assert ds.name == metadata["dataset"]
    # assert metrics == metadata["metrics"]
    # assert samplers == metadata["samplers"]
    # assert ratios == metadata["ratios"]
    # assert scales == metadata['scales']
    
    #=================================================================
    # COMPUTE AND SAVE CORESETS INDICES for each kfold: ma ATTENZIONE: sono mappati su quelli del subset
    #=================================================================
    origin = Origins['origin'] #origins[o_name] #get corresponding data

    for f, (train_set, _, _) in enumerate(featureKfolds): 
        if exists(coresets_file(f)): continue 
        logging.info(f"Computing Coresets of fold {f}.")
        selected = make_coresets(train_set.tensors[0].numpy(), scales[f])  # requires nparray
        coreset_name = coresets_file(f) 
        torch.save(selected, coreset_name)                         
        logging.info(f"Saved {ds.name} dataset coreset version {coreset_name}")   
        if log_level == logging.DEBUG : plot_hystogram(selected, bins = 100, title=coreset_name)
        logging.info(f"CORESETS (fold = {f}/{n_kfolds}) Done.")   
    #=================================================================

    # ------------------------------
    # MAIN LOOP: train and test on core datasets
    # ------------------------------    
    alltrains = {} #{i: tree() for i in range(n_kfolds)} #tree() # [{}]*n_kfolds
    alltests = {} #{i: tree() for i in range(n_kfolds)}
    avgtrains = {}
    dds = {}
    if exists(trains_file()): 
        logging.info(f"Loading Full Dataset Trained model data: {trains_file()} ")    
        alltrains = torch.load(trains_file())   
        alltests = torch.load(tests_file())     
    else:     
        lr=.1 # <<<<<<<<<<<<<!!!!!!!!!!!AAAAAAATTENZIONEEEEEE!!!!!!!!!!!!!
        logging.info(f">>>>>>>>>>>>>>>>>>>>{trains_file()} not found")
        logging.info(f"Starting training/testing for {nnet.fc_name} on {ds.name} at {timestamp}")
        logging.info(f"Using hyperparameters: lr={lr}, batch={batch}, epochs={end} out of {epochs}, device={device}")
        for f, fold in enumerate(featureKfolds):    # fold is a tuple: (train_set, val_set, test_set)
            #======================================   BASE RUNS
            #alltrains[f], alltests[f] = runModelOnBase(fold, pretrained_models(f))
            train_data, test_data = runModelonCoresets(fold, coresets_file(f))
            alltrains[f] = train_data
            alltests[f] = test_data
            #=========================================
        logging.info(f"Saving Trained model plots data: {nnet.fc_name} for {epochs} epochs")
        torch.save(alltrains, trains_file()) 
        torch.save(alltests, tests_file()) 
    
    if exists(dds_file()): 
        logging.info(f"Loading doubling estimates data from: {dds_file()} ")    
        dds = torch.load(dds_file())   
    else:
         for f, fold in enumerate(featureKfolds):  
            dds[f] = computeDoublingDim(fold, coresets_file(f), ddEstimator, config) #dds for all settings and ratios of each fold
            torch.save(dds, dds_file()) 
            
    #=========================================
    # PLOT images of train and test model runs
    #=========================================
    avgtrains = AvgTrains(alltrains, n_kfolds) if n_kfolds>1 else alltrains[0]# get a single average plot from many k-fold coresets plots
    for setting in ((m, s) for m in metrics for s in samplers):
        m, s = setting
        logging.info(f"Drawing {image_file(m, s, end)}")
        title = f'{ds.name} {nnet.fc_name} Training' #{setting}'
        draw_trains(avgtrains, setting, end, title)                    
        #draw_tests(alltests, setting, title, n_kfolds)
        #if exists(dds_file()): draw_dds(dds, setting, title)
    logging.info("TRAIN AND PLOT Done.")
    #=========================================
    
    elapsed = datetime.datetime.now() - startuptime
    logging.info(f"processing took:  {elapsed.total_seconds():.2f} seconds ({elapsed.total_seconds()/60:.2f}) minutes).")

