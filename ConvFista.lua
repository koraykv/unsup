local ConvFista = torch.class('unsup.ConvFista')

function ConvFista:__init(nin, nout, ki, kj)
   local dictionary = nn.SpatialBackConvolution(nin, nout, kj, ki, 1, 1)
   local recons = MSECriterion()
   self.decoder = FunctionCost(dictionary, recons)