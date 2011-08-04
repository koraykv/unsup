local L1Cost, parent = torch.class('nn.L1Cost','nn.Criterion')

function L1Cost:__init()
   parent.__init(self)
end

function L1Cost:forward(input)
   return input.nn.L1Cost_forward(self,input)
end

function L1Cost:backward(input)
   return input.nn.L1Cost_backward(self,input)
end

function L1Cost:write(file)
   parent.write(self, file)
end

function L1Cost:read(file)
   parent.read(self, file)
end
