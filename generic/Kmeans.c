#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Kmeans.c"
#else

static int unsup_(Kmeans_update)(lua_State *L)
{
  THTensor *data = luaT_checkudata(L,2, torch_(Tensor_id));
  THLongTensor *oldind = luaT_checkudata(L,3, torch_LongTensor_id);
  THLongTensor *newind = luaT_checkudata(L,4, torch_LongTensor_id);
  THLongTensor *counts = luaT_checkudata(L,5, torch_LongTensor_id);

  
  THTensor *dic = luaT_getfieldcheckudata(L, 1, "dictionary", torch_(Tensor_id));
  
  long nmove = 0;
  THTensor_(fill)(dic,0.0);
  THLongTensor_fill(counts,0);

  long ndata = data->size[0];
  long i;

  THTensor* dataslice = THTensor_(new)();
  THTensor* dicelement = THTensor_(new)();
  long* countdata = THLongTensor_data(counts);
  long* oldinddata = THLongTensor_data(oldind);
  long* newinddata = THLongTensor_data(newind);

  for (i=0; i<ndata; i++) 
  {
    THTensor_(select)(dataslice,data,0,i);
    long olddicid = oldinddata[i]-1;//THTensor_(get1d)(newind,i);
    long newdicid = newinddata[i]-1;//THTensor_(get1d)(oldind,i);
    //printf("%ld %ld %ld\n",i,olddicid,dicid);
    THTensor_(select)(dicelement,dic,1,newdicid);
    THTensor_(cadd)(dicelement,1,dataslice);
    if (newdicid != olddicid)
      nmove++;
    countdata[newdicid] = countdata[newdicid]+1;
  }
  THTensor_(free)(dataslice);
  THTensor_(free)(dicelement);
  lua_pushnumber(L,nmove);
  return 1;
}

/* static void unsup_(print_vec)(THTensor* v) */
/* { */
/*   long i; */
/*   for(i=0;i<v->size[0];i++){ */
/*     printf("%g\n",THTensor_(get1d)(v,i)); */
/*   } */
/*   printf("\n"); */
/* } */
/* static void unsup_(print_vec2i)(THTensor* v,long j) */
/* { */
/*   long i; */
/*   for(i=0;i<v->size[1];i++){ */
/*     printf("%g\n",THTensor_(get2d)(v,j,i)); */
/*   } */
/*   printf("\n"); */
/* } */
/* static void unsup_(print_vec2)(THTensor* v) */
/* { */
/*   long i,j; */
/*   for(i=0;i<v->size[0];i++){ */
/*     for(j=0;j<v->size[1];j++) { */
/*       printf("%g ",THTensor_(get2d)(v,i,j)); */
/*     } */
/*     printf("\n");     */
/*   } */
/*   printf("\n"); */
/* } */
static int unsup_(Kmeans_getdist)(lua_State *L)
{
  THTensor *data = luaT_checkudata(L,2, torch_(Tensor_id));
  THTensor *dic = luaT_getfieldcheckudata(L, 1, "dictionary", torch_(Tensor_id));
  
  THTensor *alldist = THTensor_(new)();
  THTensor* dici = THTensor_(new)();
  THTensor* dist = THTensor_(new)();
  THTensor* distsum = THTensor_(new)();
  THTensor* alldisti = THTensor_(new)();
  THTensor* diciclone;
  THTensor* dicirep = THTensor_(new)();

  THTensor_(resizeAs)(dist,data);
  THTensor_(resize2d)(alldist,data->size[0],dic->size[1]);
  
  long i;
  for(i=0; i<dic->size[1]; i++)
  {
    THTensor_(select)(dici,dic,1,i);
    THTensor_(select)(alldisti, alldist, 1, i);

    //unsup_(print_vec)(dici);

    diciclone = THTensor_(newClone)(dici);
    THTensor_(setStorage2d)(dicirep, diciclone->storage, 0,  
			    data->size[0], 0,
			    dici->size[0], 1);
    
    THTensor_(copy)(dist,data);
    THTensor_(cadd)(dist, -1, dicirep);
    THTensor_(cmul)(dist,dist);
    //unsup_(print_vec2(dist));
    THLab_(sum)(distsum,dist,1);
    THTensor_(copy)(alldisti,distsum);
    //unsup_(print_vec2(alldist));
    THTensor_(free)(diciclone);
  }
  //THTensor_(free)(alldist); return this
  THTensor_(free)(dici);
  THTensor_(free)(dist);
  THTensor_(free)(distsum);
  THTensor_(free)(alldisti);
  THTensor_(free)(dicirep);
  luaT_pushudata(L,alldist,torch_(Tensor_id));
  return 1;
}
  

static const struct luaL_Reg unsup_(Kmeans__) [] = {
  {"Kmeans_update", unsup_(Kmeans_update)},
  {"Kmeans_getdist", unsup_(Kmeans_getdist)},
  {NULL, NULL}
};

static void unsup_(Kmeans_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, unsup_(Kmeans__), "unsup");
  lua_pop(L,1);
}

#endif
