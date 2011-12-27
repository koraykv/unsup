require 'torch'
require 'random'
require 'lab'
require 'nn'
require 'libunsup'

-- extra modules that we need
torch.include('unsup', 'SpatialFullConvolution.lua')
torch.include('unsup', 'WeightedMSECriterion.lua')

-- classes that implement algorithms
torch.include('unsup', 'L1Cost.lua')
torch.include('unsup', 'Kmeans.lua')
torch.include('unsup', 'Fista.lua')
torch.include('unsup', 'LinearFistaL1.lua')
torch.include('unsup', 'SpatialConvFistaL1.lua')

torch.include('unsup', 'test.lua')

