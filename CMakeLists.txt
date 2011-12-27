
SET(src init.c)

SET(luasrc init.lua 
  CriterionModule.lua
  SpatialFullConvolution.lua 
  WeightedMSECriterion.lua
  L1Cost.lua
  LinearFistaL1.lua 
  SpatialConvFistaL1.lua 
  test/test.lua
  test/test_module.lua
  test/test_fista.lua
  )

ADD_TORCH_PACKAGE(unsup "${src}"  "${luasrc}" "Unsupervised Learning")

TARGET_LINK_LIBRARIES(unsup luaT TH)
