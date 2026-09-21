# ------------------------------
# set global hyperparameters
# ------------------------------
import sys, json, datetime, logging
from custom_samplers import *
from metrics import *
import dataset, model

#----------------HYPER PARAMETERS LAMBDAS and CLASS REGISTRY

# pretrained weigths for used models
Pretrained_file = {
	'IMAGENET1K_v1' : './models/resnet18/resnet18-f37072fd.pth',  # for resnet18 only
	'IMAGENET1K_v2' : './models/resnet50/resnet50-11ad3fa6.pth'	  # for resnet50 only
}

Lambdas = { 
	"Adam": lambda model, args: torch.optim.Adam(model.parameters(), lr, **args),
	"SGD" : lambda model, args: torch.optim.SGD(model.parameters(), lr, **args),
	"CrossEntropyLoss" : lambda : torch.nn.CrossEntropyLoss(),
	"F_cross_entropy" : lambda x,y : F.cross_entropy(x,y),   
	"CosineAnnealingLR" : lambda opt, args: torch.optim.lr_scheduler.CosineAnnealingLR(opt, **args) 
}
#o_name = sys.argv[1] if len(sys.argv)>2 else 'origin'

Origins = {
    'dirac': dirac_delta,
    'gaussian': gaussian,
    'uniform': uniform, 
    'origin': None
  }

Metrics = {
	'Euclidean': euclidean,
	'Manhattan': manhattan,
	'Chebyshev': chebyshev,
	'Cosine': cosine,
	'W_Euclidean': weuclidean,
	'Mahalanobis': mahalanobis,  
	'Wasserstein_fast': wasserstein_fast1D,  #computationally less expensive but equivalent to wasserstein_1D
	'Wasserstein_1D': wasserstein_1D, 
	'Wasserstein': wasserstein, 
	'Sinkhorn': sinkhorn,
  }

Samplers = {
	'First': (first, {}),
	'Last': (last, {}),
	'Random': (random, {'seed': 42}),
	'Norm_Min': (norm_min, {}),
	'Norm_Max': (norm_max, {}),
	'Centroid': (centroid, {}),
	'Middle_Point': (lambda indices, data_np, metric: indices[len(indices)//2], {}),
	'Max_Dim[0]': (lambda indices, data_np, metric: indices[np.argmax(data_np[indices, 0])], {}),
}

#------------- BASIC
log_level = logging.INFO
startuptime = datetime.datetime.now()
timestamp = startuptime.strftime("%y%m%d-%H%M")
device = torch.device("mps" if torch.mps.is_available() else "cpu")  

#-------------- PATHS
root = f'./{sys.argv[1]}/'
data_path = f"{root}data/"
save_path = f"{root}save/"    # train/test data files and metafile 
core_path=f"{root}coresets/" # coreset subsets versions of the original dataset. was tmp/
img_path = f"{root}img/"    # plot files 
log_path = f"{root}log/"
#dist_path = f".{root}/dist/" # metric distance data (if used) 

#===========================================================================
# Read hyperparameters configuration
#===========================================================================n
hyperfile = lambda: f'{root}hyper_parameters.json' 
with open(hyperfile(),'r', encoding='utf-8') as f: 
	hypm = json.load(f) 
	ds = getattr(dataset, hypm['dataset'])
	nnet = getattr(model, hypm['model']); 
	nnet.name=hypm['model']  #get network as class type and string
	pretrained_weights = Pretrained_file[hypm['pretrained_weights']] if hypm['pretrained_weights'] != "" else None
	metrics = set(hypm['metrics'])
	samplers = set(hypm['samplers'])
	origins = set(hypm['origins'])
	ratios = hypm['ratios']
	seed = hypm['seed']
	batch = hypm['batch_size']
	epochs = hypm['epochs']
	lr = hypm["lr"]
	validation_frequency = hypm['validation_frequency']
	criterion_lambda = Lambdas[hypm['criterion']]
	criterion_kwargs  = hypm["criterion_kwargs"]
	optimizer_lambda = Lambdas[hypm['optimizer']]
	optimizer_kwargs = hypm["optimizer_kwargs"]
	scheduler_lambda = Lambdas[hypm["scheduler"]]
	scheduler_kwargs = hypm["scheduler_kwargs"]
	n_kfolds = hypm["n_kfolds"]     
	val_quota= hypm["val_quota"]    #val/train ratio
	test_quota= hypm["test_quota"] #test quota if not k-folding
	YLIMACCU = hypm["YLIMACCU"][0], hypm["YLIMACCU"][1]
	YLIMLOSS = hypm["YLIMLOSS"][0], hypm["YLIMLOSS"][1]
	TLIMACCU = hypm["TLIMACCU"][0], hypm["TLIMACCU"][1]
	TLIMLOSS = hypm["TLIMLOSS"][0], hypm["TLIMLOSS"][1]

#===========================================================================

	#o_name = sys.argv[1] if len(sys.argv)>2 else 'origin'

#---------------- FILES & NAMES
logfile = f"{log_path}{ds.name}-{nnet.name}_{timestamp}.log"
features_file = lambda fold: f"{data_path}{ds.name}_features_{fold}.pt"
pretrained_models = lambda fold: f"{data_path}{nnet.name}_{ds.name}_weights_{fold}.pt" # last is from 15 epochs
metafile = lambda fold: f'{save_path}{ds.name}_metadata_{fold}.json' #if len(sys.argv)>1 else f'{save_path}_metadata.json'
coresets_file = lambda fold : f'{core_path}{ds.name}_coreset_{fold}.pt'
trains_file = lambda :f'{save_path}{nnet.fc_name}_{ds.name}_trains_{epochs}.pt'
tests_file = lambda : f'{save_path}{nnet.fc_name}_{ds.name}_tests_{epochs}.pt' 
dds_file = lambda: f'{save_path}{nnet.fc_name}_{ds.name}_dds.pt' 
image_file = lambda m,s,end : f'{img_path}{nnet.fc_name}-train-{m}-{s}-{end}.png' 
image2_file = lambda m,s: f'{img_path}{nnet.fc_name}-test-{m}-{s}.png' 
image3_file = lambda m,s: f'{img_path}{nnet.fc_name}-dds-{m}-{s}.png' 

#distances_file = lambda m,o : f"{dist_path}{ds.name}_{m}_distances_from_{o}.pt"
#hystogram_file = lambda m,o : f"{img_path}{ds.name}_{m}_distances_from_{o}.png"





