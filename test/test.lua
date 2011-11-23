

require 'paths'
dofile(paths.concat(paths.dirname(paths.thisfile()),'test_conv.lua'))

function unsup.test()
   unsup.conv_test()
   print()
   print()
   --dofile(paths.concat(paths.dirname(paths.thisfile()),'test_km.lua'))
   --dofile(paths.concat(paths.dirname(paths.thisfile()),'test_fista.lua'))
end
