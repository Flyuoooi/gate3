from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# Name of backbone
_C.MODEL.TYPE = ''
# Model name
_C.MODEL.NAME = ''
_C.MODEL.PRETRAIN_PATH=''


# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Dimension of the attribute list
_C.MODEL.TEXT_DIMS = []
_C.MODEL.CLOTH_XISHU = 3
# Add cloth embedding only, options: 'True', 'False'
_C.MODEL.CLOTH_ONLY = False
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'

_C.MODEL.ADD_TEXT = False
_C.MODEL.TEXT = CN()
_C.MODEL.TEXT.CLIP_NAME = "ViT-B/16"
_C.MODEL.TEXT.N_CTX = 4

_C.MODEL.TEXT.METHOD = "cocoop"      # "coop" or "cocoop"
_C.MODEL.TEXT.ENABLE_MASK = False    # 是否对 ctx 做 mask（mask 后再 pool 成 [B,512]）

_C.MODEL.TEXT.ATTN_MASK_RATIO = 0.5
_C.MODEL.TEXT.TEMP = 0.07
_C.MODEL.TEXT.ITC_W = 0.0

# --- Proxy-only feature-wise keep-top + soft gate (identity-evidence preserving) ---
_C.MODEL.TEXT.PROXY_GATE_ON = True
_C.MODEL.TEXT.PROXY_GATE_WARMUP= 20
_C.MODEL.TEXT.PROXY_GATE_TEMP = 1.0       # sigmoid temperature for non-top-k dims
_C.MODEL.TEXT.PROXY_GATE_MIN = 0.0        # lower bound for non-top-k gates
_C.MODEL.TEXT.PROXY_GATE_APPLY = "t1,t2"  # apply to injected tokens: t1, t2, or "t1"
_C.MODEL.TEXT.PROXY_KEEP_RATIO = 0.5      # keep top-K dims (K = ratio * embed_dim)
_C.MODEL.TEXT.PROXY_DETACH_PROXY = True   # detach classifier proxy when gating
_C.MODEL.TEXT.PROXY_DETACH_SCORE = False  # detach score (gate becomes constant wrt token)


_C.MODEL.TEXT.PROXY_LAMBDA_PRE = 0.1  # Design A: lambda reaches this at end of warmup
_C.MODEL.TEXT.PROXY_LAMBDA_RAMP_EPOCHS = 30  # <=0 means auto (total_epoch - warmup)

_C.MODEL.TEXT.PROXY_PULL_APPLY= "t1,t2"
_C.MODEL.TEXT.PROXY_SIM_ON = True  # Design B: similarity matching loss
_C.MODEL.TEXT.PROXY_SIM_W = 0.75

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1



# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE =16
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_HEIGHT = 224
_C.DATA.IMG_WIDTH = 224
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Data root
_C.DATA.ROOT = '/home/jin/code/MADE/Data'
# Number of instances
_C.DATA.NUM_INSTANCES = 4 #8
# Batch size during testing
_C.DATA.TEST_BATCH = 128
# Data sampling strategy
_C.DATA.SAMPLER = 'softmax_triplet'
_C.DATA.RANDOM_NOISE = False
_C.DATA.RANDOM_PROP = 0.05


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()

# Random crop prob
_C.AUG.RC_PROB = 1.0
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.WARMUP_LR = 7.8125e-07
# _C.SOLVER.WARMUP_LR = 1e-06
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 60)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 15
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1


# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Test using images only
_C.TEST.TYPE = 'image_only'
_C.TEST.RERANKING = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
