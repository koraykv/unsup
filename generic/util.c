#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/util.c"
#else

static int unsup_(shrinkage)(lua_State *L)
{
  real lambda = luaL_checknumber(L,2);
  THTensor *tensor = luaT_checkudata(L,1, torch_Tensor);
  luaL_argcheck(L, lambda >=0, 2, "Lambda should be non-negative");

  if (lambda == 0) return 1;

  TH_TENSOR_APPLY(real, tensor,
		  if (*tensor_data > lambda)
		  {
		    *tensor_data -= lambda;
		  }
		  else if (*tensor_data < -lambda)
		  {
		    *tensor_data += lambda;
		  }
		  else
		  {
		    *tensor_data = 0;
		  });
  return 1;
}

static const struct luaL_Reg unsup_(util__) [] = {
  {"shrinkage", unsup_(shrinkage)},
  {NULL, NULL}
};

static void unsup_(util_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaL_setfuncs(L, unsup_(util__), 0);
  lua_pop(L,1);
}

#endif
