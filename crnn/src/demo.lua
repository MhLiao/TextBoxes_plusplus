require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('crnn/src/libcrnn')
require('crnn/src/utilities')
require('crnn/src/inference')
require('crnn/src/CtcCriterion')
require('crnn/src/LstmLayer')
require('crnn/src/BiRnnJoin')
require('crnn/src/SharedParallelTable')
require('crnn/src/WeightCombineTable')

local Image = require('image')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('-modelDir','./crnn/model/','crnn model directory')
cmd:option('-bboxDir','./demo_images/detection_result/','bounding box directory')
cmd:option('-imgDir','./demo_images/','image directory')
cmd:option('-imgName','demo.jpg','image name')
cmd:option('-cropDir','./demo_images/crops/','crop images directory')
cmd:option('-resultDir','./demo_images/recognition_result/','recognition result directory')
cmd:option('-dicPath','./crnn/data/icdar_generic_lexicon.txt','lexicon path')
cmd:option('-useLexicon',false,'use lexicon or not')
cmd:text()

config = cmd:parse(arg)

-- cutorch.setDevice(0)
torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = config.modelDir
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'model_crnn.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

function readAllLines(fpath)
	local lines = {}
	local idx = 1
	for line in io.lines(fpath) do
		lines[idx] = line
		idx = idx + 1
	end
	return lines
end

function split(s, delimiter)
    result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end

-- remove the file's extension
function stripextension(filename)  
	filename = string.match(filename, ".+/([^/]*%.%w+)$")
    local idx = filename:match(".+()%.%w+$")  
    if(idx) then  
        return filename:sub(1, idx-1)  
    else  
        return filename  
    end  
end

function file_exists(path)
  local file = io.open(path, "rb")
  if file then file:close() end
  return file ~= nil
end


local image_name = config.imgName
local imgPath = config.imgDir .. image_name
local bbox_recognitionPath = config.resultDir .. split(image_name,"%.")[1] .. '.txt'
local sub_dir = split(image_name,"%.")[1]
local bboxPath = config.bboxDir .. split(image_name,"%.")[1] .. '.txt'
local bboxes = readAllLines(bboxPath)

if config.useLexicon then
	local lexicon = readAllLines(dicPath)
end
if #bboxes == 1 and string.len(bboxes[1]) == 1 then
else
	local bbox_recognition = io.open(bbox_recognitionPath, 'w+')
	for ii = 1, #bboxes do
		local bbox = bboxes[ii]
		local coordinates = split(bbox, ',')
		local x1 = tonumber(coordinates[1])
		local y1 = tonumber(coordinates[2])
		local x2 = tonumber(coordinates[3])
		local y2 = tonumber(coordinates[4])
		local x3 = tonumber(coordinates[5])
		local y3 = tonumber(coordinates[6])
		local x4 = tonumber(coordinates[7])
		local y4 = tonumber(coordinates[8])
		local det_score = tonumber(coordinates[9])
		bbox = string.format('%d,%d,%d,%d,%d,%d,%d,%d,%f', x1,y1,x2,y2,x3,y3,x4,y4,det_score)
		crop_path = string.format('%s%s/%d.jpg',config.cropDir,sub_dir,ii)
		if file_exists(crop_path) then
			local imgCrop = Image.load(crop_path, 3, 'byte')
			imgCrop = Image.rgb2y(imgCrop)
			local height, width = imgCrop:size(2), imgCrop:size(3)
			imgCrop = Image.scale(imgCrop, 100, 32)[1]
			if config.useLexicon then
				local resTopN, probTopN = recognizeImageWithLexicion(model, imgCrop, lexicon, 1)
				local text = resTopN[1]
				local prob = probTopN[1]
				bbox_recognition:write(string.format('%s,%s,%f', bbox, text, prob) .. '\n')
			else
				local text, raw, prob = recognizeImageLexiconFree(model, criterion, imgCrop)
				bbox_recognition:write(string.format('%s,%s,%f', bbox, text, prob) .. '\n')
			end
		end	
	end
	bbox_recognition:close()
end
