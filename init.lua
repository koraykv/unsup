require 'torch'
require 'nn'

require 'kex'
require 'optim'

unsup = {}

-- classes that implement algorithms
torch.include('unsup', 'UnsupModule.lua')
torch.include('unsup', 'AutoEncoder.lua')
torch.include('unsup', 'SparseAutoEncoder.lua')
torch.include('unsup', 'LinearFistaL1.lua')
torch.include('unsup', 'SpatialConvFistaL1.lua')
torch.include('unsup', 'psd.lua')
torch.include('unsup', 'LinearPsd.lua')
torch.include('unsup', 'ConvPsd.lua')
torch.include('unsup', 'UnsupTrainer.lua')
torch.include('unsup', 'pca.lua')
