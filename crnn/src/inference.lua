function naiveDecoding(input)
    --[[ Naive, lexicon-free decoding
    ARGS:
      - `input`   : float tensor [nFrame x inputLength x nClasses]
    RETURNS:
      - `pred`    : int tensor [nFrame x inputLength]
      - `predRaw` : int tensor [nFrame x inputLength]
    ]]

    assert(input:dim() == 3)
    local nFrame, inputLength = input:size(1), input:size(2)
    local pred, predRaw = input.nn.CTC_naiveDecoding(input)
    return pred, predRaw
end


function decodingWithLexicon(input, lexicon, topN)
    --[[ Decoding by selecting the lexicon word with the highest probability
    ARGS:
      - `input`   : float tensor [nFrame x inputLength x nClasses], model feed forward output
    RETURNS:
      - `pred`    : int tensor [nFrame x inputLength]
      - `predRaw` : int tensor [nFrame x inputLength]
    ]]

    assert(input:dim() == 3 and input:size(1) == 1)
    assert(type(lexicon) == 'table')
    local lexSize = #lexicon

    local target = str2label(lexicon, 30) -- FIXME
    local inputN = torch.repeatTensor(input, lexSize, 1, 1)
    -- loss = -logProb
    local loss, prob = inputN.nn.CTC_forwardBackward(inputN, target, true, inputN.new())
    local resTopN, probTopN = {}, {}
    local _, idx = torch.sort(prob, 1, true)
    local topNIdx = idx:narrow(1,1,topN)
    for i = 1, topN do
      idx = topNIdx[i]
      resTopN[i]  = lexicon[idx]
      probTopN[i] = prob[idx]
    end
    -- local _, idx = torch.max(prob, 1)
    -- idx = idx[1]
    return resTopN, probTopN
end

function decodingWithLexiconBK(input, lexiconTable, topN)
    --[[ Decoding by selecting the lexicon word with the highest probability
    ARGS:
      - `input`   : float tensor [nFrame x inputLength x nClasses], model feed forward output
    RETURNS:
      - `pred`    : int tensor [nFrame x inputLength]
      - `predRaw` : int tensor [nFrame x inputLength]
    ]]
    assert(input:dim() == 3 and input:size(1) == 1)
    -- print(input)
    local pred, predRaw = input.nn.CTC_naiveDecoding(input)
    local str = label2str(pred)[1]
    -- print(str)
    -- print(lexiconTable)
    local lexicon = Lexicon(lexiconTable)
    local candidate, edists = lexicon:searchWithin(str, 4, 100)

    local nNeighbor = #candidate
    if nNeighbor < 1 then
      local resTopN, probTopN = {}, {}
      resTopN[1] = str
      probTopN[1] = 0
      return resTopN, probTopN
    end
    -- print(nNeighbor)
    local target = str2label(candidate, 30) -- FIXME
    local inputN = torch.repeatTensor(input, nNeighbor, 1, 1)
    -- loss = -logProb
    local loss, prob = inputN.nn.CTC_forwardBackward(inputN, target, true, inputN.new())
    local resTopN, probTopN = {}, {}
    local _, idx = torch.sort(prob, 1, true)
    local topNIdx = idx:narrow(1,1,topN)
    for i = 1, topN do
      idx = topNIdx[i]
      resTopN[i]  = candidate[idx]
      probTopN[i] = prob[idx]
    end
    -- local _, idx = torch.max(prob, 1)
    -- idx = idx[1]
    return resTopN, probTopN
end
