#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define unsup_(NAME) TH_CONCAT_3(unsup_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;
static const void* torch_LongTensor_id = NULL;

#include "generic/Kmeans.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialBackConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/L1Cost.c"
#include "THGenerateFloatTypes.h"

#include "generic/util.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libunsup(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");
  torch_LongTensor_id = luaT_checktypename2id(L, "torch.LongTensor");

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "unsup");

  unsup_FloatKmeans_init(L);
  unsup_DoubleKmeans_init(L);

  nn_FloatSpatialBackConvolution_init(L);
  nn_DoubleSpatialBackConvolution_init(L);

  nn_FloatL1Cost_init(L);
  nn_DoubleL1Cost_init(L);

  unsup_Floatutil_init(L);
  unsup_Doubleutil_init(L);
  
  return 1;
}


