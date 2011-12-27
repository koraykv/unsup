#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/util.c"
#else

static int unsup_(shrinkage)(lua_State *L)
{
  real lambda = luaL_checknumber(L,2);
  THTensor *tensor = luaT_checkudata(L,1, torch_(Tensor_id));
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

static int unsup_(sign)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *tensor = NULL;
  THTensor *r = luaT_checkudata(L,1,torch_(Tensor_id));
  if (narg == 1)
  {
    tensor = r;
  }
  else if (narg == 2)
  {
    tensor = luaT_checkudata(L,2,torch_(Tensor_id));
  }
  else
  {
    luaL_error(L,"1 or 2 input tensors expected");
  }
  TH_TENSOR_APPLY2(real, r, real, tensor,
		   if (*tensor_data > 0)
		     *r_data = 1;
		   else if (*tensor_data < 0)
		     *r_data = -1;
		   else
		     *r_data = 0;);
  return 1;
}

static const struct luaL_Reg unsup_(util__) [] = {
  {"shrinkage", unsup_(shrinkage)},
  {"sign", unsup_(sign)},
  {NULL, NULL}
};

static void unsup_(util_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaL_register(L, NULL, unsup_(util__));
  lua_pop(L,1);
}

#endif
