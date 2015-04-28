local ConvPSD, parent = torch.class('unsup.ConvPSD','unsup.PSD')

-- conntable       : A connection table (ref nn.SpatialConvolutionMap)
-- kw, kh          : width, height of convolutional kernel
-- iw, ih          : width, height of input patches
-- lambda          : sparsity coefficient
-- beta            : prediction coefficient
-- params          : optim.FistaLS parameters
function ConvPSD:__init(conntable, kw, kh, iw, ih, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   local decodertable = conntable:clone()
   decodertable:select(2,1):copy(conntable:select(2,2))
   decodertable:select(2,2):copy(conntable:select(2,1))
   local outputFeatures = conntable:select(2,2):max()

   -- decoder is L1 solution
   self.decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, lambda, params)


   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = params.encoderType or 'linear'

   if params.encoderType == 'linear' then
      self.encoder = nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1)
   elseif params.encoderType == 'tanh' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
      self.encoder:add(nn.Tanh())
      self.encoder:add(nn.Diag(outputFeatures))
   elseif params.encoderType == 'tanh_shrink' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
      self.encoder:add(nn.TanhShrink())
      self.encoder:add(nn.Diag(outputFeatures))
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, beta, params)
end

