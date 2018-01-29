local WeightCombineTable, parent = torch.class('nn.WeightCombineTable', 'nn.Module')


function WeightCombineTable:__init(maxT)
    parent.__init(self)
    self.output    = {}
    self.gradInput = {}
    self.maxT = maxT
end


function WeightCombineTable:updateOutput(input)
    assert(type(input) == 'table', 'input must be table')

    local T = #input
    self.output = input[1]
    for t = 2, T do
        self.output = self.output + input[t]
    end
    self.output = self.output / T
    return self.output
end


function WeightCombineTable:updateGradInput(input, gradOutput)
    local T = #input
    self.gradInput = {}
    for t = 1, T do
        self.gradInput[t] = gradOutput * 1 / T
    end
    return self.gradInput
end


function WeightCombineTable:accGradParameters(input, gradOutput, scale)
end
