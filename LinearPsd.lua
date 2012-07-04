local LinearPSD, parent = torch.class('unsup.LinearPSD','unsup.PSD')

-- inputSize   : size of input
-- outputSize  : size of code
-- lambda      : sparsity coefficient
-- beta	       : prediction coefficient
-- params      : optim.FistaLS parameters
function LinearPSD:__init(inputSize, outputSize, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   -- decoder is L1 solution
   self.decoder = unsup.LinearFistaL1(inputSize, outputSize, lambda, params)

   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = params.encoderType or 'linear'

   if params.encoderType == 'linear' then
      self.encoder = nn.Linear(inputSize,outputSize)
   elseif params.encoderType == 'tanh' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Linear(inputSize,outputSize))
      self.encoder:add(nn.Tanh())
      self.encoder:add(nn.Diag(outputSize))
   elseif params.encoderType == 'tanh_shrink' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Linear(inputSize,outputSize))
      self.encoder:add(nn.TanhShrink())
      self.encoder:add(nn.Diag(outputSize))
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, self.beta, self.params)

end
