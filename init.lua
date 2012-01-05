require 'lab'
require 'nn'

require 'kex'
require 'optim'

unsup = {}

-- classes that implement algorithms
torch.include('unsup', 'LinearFistaL1.lua')
torch.include('unsup', 'SpatialConvFistaL1.lua')

