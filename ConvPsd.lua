local ConvPSD, parent = torch.class('unsup.ConvPSD','unsup.PSD')

-- inputFeatures   : number of input features
-- outputFeatures  : size of code (feature maps)
-- kw, kh          : width, height of convolutional kernel
-- iw, ih          : width, height of input patches
-- lambda          : sparsity coefficient
-- beta            : prediction coefficient
-- params          : optim.FistaLS parameters
function ConvPSD:__init(inputFeatures, outputFeatures, kw, kh, iw, ih, lambda, beta, params)
   

   -- prediction weight
   self.beta = beta

   -- decoder is L1 solution
   self.decoder = unsup.SpatialConvFistaL1(inputFeatures, outputFeatures, kw, kh, iw, ih, lambda, params)

   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = params.encoderType or 'linear'

   if params.encoderType == 'linear' then
      self.encoder = nn.SpatialConvolution(inputFeatures, outputFeatures, kw, kh, 1, 1)
   elseif params.encoderType == 'tanh' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.SpatialConvolution(inputFeatures, outputFeatures, kw, kh, 1, 1))
      self.encoder:add(nn.Tanh())
      self.encoder:add(nn.Diag())
   elseif params.encoderType == 'tanh_shrink' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.SpatialConvolution(inputFeatures, outputFeatures, kw, kh, 1, 1))
      self.encoder:add(nn.TanhShrink())
      self.encoder:add(nn.Diag())
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, beta, params)
end

