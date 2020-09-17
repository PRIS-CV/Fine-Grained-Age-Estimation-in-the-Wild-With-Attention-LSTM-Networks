--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The full pre-activation ResNet variation from the technical report
-- "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027)
--

local nn = require 'nn'
local nninit=require 'nninit'
require 'cunn'
require 'ResidualDropIpretrain'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 1,1, stride,stride)
                                             :init('weight', nninit.kaiming, {gain = 'relu'}))
      --      :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n

      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))

      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n * 4

      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))

      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride, type)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      s:add(block(features, stride,
                  type == 'first' and 'no_preact' or 'both_preact'))
      for i=2,count do
         s:add(block(features, 1))
      end
      return s
   end


   local modelPath = opt.retrain
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
   local   modelpre = torch.load(modelPath)
   print(modelpre)
 
   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [82]  = {{7, 8, 14, 7}, 512, basicblock},
         [58]  = {{5, 6, 12, 5}, 512, basicblock},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
         [200] = {{3, 24, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
--      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(modelpre:get(1))
      model:add(Max(3,3,2,2,1,1))
      model1=nn.Sequential()
      model21=nn.Sequential()
    --  for i=1, def[1] do addResidualDrop(model21, nil, 64, next(modelpre:findModules('cudnn.SpatialConvolution'),2*i), next(modelpre:findModules('cudnn.SpatialConvolution'),2*i+1)) end
      for i=1, def[1] do addResidualDrop(model21, nil,64, 64,1,modelpre:get(5):get(i):get(1):get(1):get(1),modelpre:get(5):get(i):get(1):get(1):get(4)) end
      model1:add(nn.ConcatTable()
               :add(model21)
               :add(shortcut(64,64)))
            :add(nn.CAddTable(true))
      model22=nn.Sequential()    
      addResidualDrop(model22, nil, 64, 128,2,modelpre:get(6):get(1):get(1):get(1):get(1),modelpre:get(6):get(1):get(1):get(1):get(4),modelpre:get(6):get(1):get(1):get(2)) 
      for i=1, def[2]-1 do addResidualDrop(model22, nil, 128, 128,1,modelpre:get(6):get(i+1):get(1):get(1):get(1),modelpre:get(6):get(i+1):get(1):get(1):get(4)) end
      model1:add(nn.ConcatTable()
               :add(model22)
               :add(shortcut(64,128,2)))
            :add(nn.CAddTable(true))
      model23=nn.Sequential()
      addResidualDrop(model23, nil, 128, 256,2,modelpre:get(7):get(1):get(1):get(1):get(1),modelpre:get(7):get(1):get(1):get(1):get(4),modelpre:get(7):get(1):get(1):get(2)) 
      for i=1, def[3]-1 do addResidualDrop(model23, nil, 256, 256,1,modelpre:get(7):get(i+1):get(1):get(1):get(1),modelpre:get(7):get(i+1):get(1):get(1):get(4)) end
      model1:add(nn.ConcatTable()
               :add(model23)
               :add(shortcut(128,256,2)))
            :add(nn.CAddTable(true))
      model24=nn.Sequential()
      addResidualDrop(model24, nil, 256, 512,2,modelpre:get(8):get(1):get(1):get(1):get(1),modelpre:get(8):get(1):get(1):get(1):get(4),modelpre:get(8):get(1):get(1):get(2)) 
      for i=1, def[4]-1 do addResidualDrop(model24, nil, 512, 512,1,modelpre:get(8):get(i+1):get(1):get(1):get(1),modelpre:get(8):get(i+1):get(1):get(1):get(4)) end
      model1:add(nn.ConcatTable()
               :add(model24)
               :add(shortcut(256,512,2)))
            :add(nn.CAddTable(true))
       model:add(nn.ConcatTable()
            :add(model1)
            :add(shortcut(64, 512, 8)))
         :add(nn.CAddTable(true))

      model:add(cudnn.SpatialBatchNormalization(512))
      model:add(cudnn.ReLU(true)) 
      if opt.Dropout=='Dropout' then
      model:add(nn.Dropout(0.5))     
      end
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures,opt.nClasses))
print(model)
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

     model:add(cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1))
------> 16, 32,32   First Group
      for i=1,n do   addResidualDrop(model, nil, 16) model:add(cudnn.ReLU(true))  end
------> 32, 16,16   Second Group
      addResidualDrop(model, nil, 16, 32, 2) model:add(cudnn.ReLU(true))
      for i=1,n-1 do   addResidualDrop(model, nil, 32)  model:add(cudnn.ReLU(true))  end
------> 64, 8,8     Third Group
      addResidualDrop(model, nil, 32, 64, 2) model:add(cudnn.ReLU(true))
      for i=1,n-1 do   addResidualDrop(model, nil, 64)  model:add(cudnn.ReLU(true))  end
------> 10, 8,8     Pooling, Linear, Softmax
      model:add(cudnn.SpatialBatchNormalization(64))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialAveragePooling(8,8)):add(nn.Reshape(64))
      model:add(nn.Linear(64, 10))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   -------------------suijishendu------------------------------------------------
   --[[addtables = {}
for i=1,model:size() do
    if tostring(model:get(i)) == 'layer' then addtables[#addtables+1] = i end
--print(addtables)
end
for i,block in ipairs(addtables) do
  if opt.deathMode == 'uniform' then
    model:get(block).deathRate = opt.deathRate
  elseif opt.deathMode == 'lin_decay' then
    model:get(block).deathRate = i / #addtables * opt.deathRate
  else
    print('Invalid argument for deathMode!')
  end
end
function openAllGates()
  for i,block in ipairs(addtables) do model:get(block).gate = true end
end--]]

-------------------------------------------------------------
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end
--function addResidualDrop(deathRate,block, features, count, stride, type)
   --model:add(layer(block, features, count, stride, type))
   --add(layer(basicblock, 16, n, 1))
  -- model:add(cudnn.ReLU(true))
 -- return model
--end
return createModel
