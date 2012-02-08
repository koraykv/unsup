
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src)# init.c)

SET(luasrc init.lua 
  LinearFistaL1.lua 
  SpatialConvFistaL1.lua 
  psd.lua
  LinearPsd.lua
  ConvPsd.lua
  test/test_fista.lua
  )

ADD_TORCH_PACKAGE(unsup "${src}"  "${luasrc}" "Unsupervised Learning")

#TARGET_LINK_LIBRARIES(unsup luaT TH)
