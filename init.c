#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define unsup_(NAME) TH_CONCAT_3(unsup_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#include "generic/util.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libunsup(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "unsup");

  unsup_Floatutil_init(L);
  unsup_Doubleutil_init(L);

  return 1;
}
