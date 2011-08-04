
SET(src init.c)

SET(luasrc init.lua 
  CriterionModule.lua 
  L1Cost.lua
  FunctionCost.lua 
  LinearFista.lua 
  Fista.lua 
  Kmeans.lua 
  SpatialBackConvolution.lua 
  WeightedMSECriterion.lua
  )

ADD_TORCH_PACKAGE(unsup "${src}"  "${luasrc}" "Unsupervised Learning")

TARGET_LINK_LIBRARIES(unsup luaT TH)
