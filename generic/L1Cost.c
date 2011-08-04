#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/L1Cost.c"
#else

static int nn_(L1Cost_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  accreal sum;

  sum = 0;
  TH_TENSOR_APPLY(real, input, 
		  sum += fabs(*input_data););

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(L1Cost_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY2(real, gradInput, real, input,
                   *gradInput_data = ( *input_data >= 0 ? 1 : -1););
    
  return 1;
}

static const struct luaL_Reg nn_(L1Cost__) [] = {
  {"L1Cost_forward", nn_(L1Cost_forward)},
  {"L1Cost_backward", nn_(L1Cost_backward)},
  {NULL, NULL}
};

static void nn_(L1Cost_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(L1Cost__), "nn");
  lua_pop(L,1);
}

#endif
