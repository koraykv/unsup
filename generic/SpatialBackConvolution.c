#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialBackConvolution.c"
#else


static int nn_(SpatialBackConvolution_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  /* do convolutions */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);
  THLab_(conv2Dmv)(output, 0.0, input, tweight, dH, dW, "fc");
  THTensor_(free)(tweight);
  
  return 1;
}


static int nn_(SpatialBackConvolution_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  
  long nOutputPlane = weight->size[1];
  THArgCheck( nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane" );

  /* gradient to kernels */
  THLab_(conv2DRevger)(gradWeight, 1.0, gradOutput, input, dH, dW);

  /* gradient to input */
  THLab_(conv2Dmv)(gradInput, 0.0, gradOutput, weight, dH, dW, "vx");

  return 1;
}

static const struct luaL_Reg nn_(SpatialBackConvolution__) [] = {
  {"SpatialBackConvolution_forward", nn_(SpatialBackConvolution_forward)},
  {"SpatialBackConvolution_backward", nn_(SpatialBackConvolution_backward)},
  {NULL, NULL}
};

static void nn_(SpatialBackConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialBackConvolution__), "nn");
  lua_pop(L,1);
}

#endif
