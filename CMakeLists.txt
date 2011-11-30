
SET(src init.c)

SET(luasrc init.lua 
  CriterionModule.lua 
  L1Cost.lua
  FunctionCost.lua 
  LinearFista.lua 
  FistaDrivers.lua 
  Fista.lua 
  Fista2.lua 
  Kmeans.lua 
  SpatialBackConvolution.lua 
  WeightedMSECriterion.lua
  test/test.lua
  test/test_conv.lua
  test/test_km.lua
  test/test_fista.lua
  test/test_fista2.lua
  )

ADD_TORCH_PACKAGE(unsup "${src}"  "${luasrc}" "Unsupervised Learning")

TARGET_LINK_LIBRARIES(unsup luaT TH)
