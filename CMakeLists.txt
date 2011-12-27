
SET(src init.c)

SET(luasrc init.lua 
  SpatialFullConvolution.lua 
  WeightedMSECriterion.lua
  L1Cost.lua
  LinearFistaL1.lua 
  SpatialConvFistaL1.lua 
  Kmeans.lua 
  test/test.lua
  test/test_conv.lua
  test/test_km.lua
  test/test_fista.lua
  )

ADD_TORCH_PACKAGE(unsup "${src}"  "${luasrc}" "Unsupervised Learning")

TARGET_LINK_LIBRARIES(unsup luaT TH)
