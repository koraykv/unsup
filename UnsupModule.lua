local UnsupModule,parent = torch.class('unsup.UnsupModule','nn.Module')

function UnsupModule:__init()
	parent.__init(self)
end

function UnsupModule:normalize()
	error('Every unsupervised module has to implement normalize function')
end

