

require 'paths'
dofile(paths.concat(paths.dirname(paths.thisfile()),'test_module.lua'))

function unsup.test()
   unsup.module_test()
   print()
   print()
   --dofile(paths.concat(paths.dirname(paths.thisfile()),'test_fista.lua'))
end
