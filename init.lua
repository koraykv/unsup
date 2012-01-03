require 'torch'
require 'random'
require 'lab'
require 'nn'

require 'kex'
require 'optim'
require 'libunsup'

-- extra modules that we need
torch.include('unsup', 'SpatialFullConvolution.lua')
torch.include('unsup', 'WeightedMSECriterion.lua')
torch.include('unsup', 'L1Cost.lua')
torch.include('unsup', 'CriterionModule.lua')

-- classes that implement algorithms
torch.include('unsup', 'LinearFistaL1.lua')
torch.include('unsup', 'SpatialConvFistaL1.lua')

torch.include('unsup', 'test.lua')

