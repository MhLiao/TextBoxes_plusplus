local Lexicon, parent = torch.class('Lexicon')

function Lexicon:__init(lexicon)
    if type(lexicon) == 'string' then
        assert(paths.filep(lexicon), string.format('%s does not exist', lexicon))
        self.lex = torch.Tensor().nn.LEX_createLexiconFromFile(lexicon)
    elseif type(lexicon) == 'table' then
        self.lex = torch.Tensor().nn.LEX_createLexiconFromStringList(lexicon)
    else
        assert(false)
    end
end

function Lexicon:manualDestroy()
    torch.Tensor().nn.LEX_destroyLexicon(self.lex)
end

function Lexicon:searchWithin(query, dist, maxK)
    dist = dist or 2
    maxK = maxK or 64
    local neighbours, dists = torch.Tensor().nn.LEX_searchWithin(self.lex, query, dist, maxK)
    return neighbours, dists
end
