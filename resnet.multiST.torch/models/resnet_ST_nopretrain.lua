--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'image'
require 'cunn'
require 'torch'

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
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
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

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

local networks = {}

	-- These are the basic modules used when creating any macro-module
	-- Can be modified to use for example cudnn
	networks.modules = {}
	networks.modules.convolutionModule = nn.SpatialConvolutionMM
	networks.modules.poolingModule = nn.SpatialMaxPooling
	networks.modules.nonLinearityModule = nn.ReLU

	-- Size of the input image. This comes from the dataset loader
	networks.base_input_size = 48

	-- Number of output classes. This comes from the dataset.
	networks.nbr_classes = 43

	-- Creates a conv module with the specified number of channels in input and output
	-- If multiscale is true, the total number of output channels will be:
	-- nbr_input_channels + nbr_output_channels
	-- Using no_cnorm removes the spatial contrastive normalization module
	-- The filter size for the convolution can be specified (default 5)
	-- The stride of the convolutions is fixed at 1
	function networks.new_conv(nbr_input_channels,nbr_output_channels,
							   multiscale, no_cnorm, filter_size)
	  multiscale = multiscale or false
	  no_cnorm = no_cnorm or false
	  filter_size = filter_size or 5
	  local padding_size = 2
	  local pooling_size = 2
	  local normkernel = image.gaussian1D(7)

	  local conv

	  local first = nn.Sequential()
	  first:add(networks.modules.convolutionModule(nbr_input_channels,
										  nbr_output_channels,
										  filter_size, filter_size,
										  1,1,
										  padding_size, padding_size))
	  first:add(networks.modules.nonLinearityModule())
	  first:add(networks.modules.poolingModule(pooling_size, pooling_size,
											   pooling_size, pooling_size))
	  if not no_cnorm then
		first:add(nn.SpatialContrastiveNormalization(nbr_output_channels,
													 norm_kernel))
	  end

	  if multiscale then
		conv = nn.Sequential()
		local second = networks.modules.poolingModule(pooling_size, pooling_size,
												  pooling_size, pooling_size)

		local parallel = nn.ConcatTable()
		parallel:add(first)
		parallel:add(second)
		conv:add(parallel)
		conv:add(nn.JoinTable(1,3))
	  else
		conv = first
	  end

	  return conv
	end

	-- Gives the number of output elements for a table of convolution layers
	-- Also returns the new height (=width) of the image
	function networks.convs_noutput(convs, input_size)
	  input_size = input_size or networks.base_input_size
	  -- Get the number of channels for conv that are multiscale or not
	  local nbr_input_channels = convs[1]:get(1).nInputPlane or
								 convs[1]:get(1):get(1).nInputPlane
	  local output = torch.Tensor(1,nbr_input_channels, input_size, input_size)
	  for _, conv in ipairs(convs) do
		output = conv:forward(output)
	  end
	  return output:nElement(), output:size(3)
	end

	-- Creates a fully connection layer with the specified size.
	function networks.new_fc(nbr_input, nbr_output)
	  local fc = nn.Sequential()
	  fc:add(nn.View(nbr_input))
	  fc:add(nn.Linear(nbr_input, nbr_output))
	  fc:add(networks.modules.nonLinearityModule())
	  return fc
	end

	-- Creates a classifier with the specified size.
	function networks.new_classifier(nbr_input, nbr_output)
	  local classifier = nn.Sequential()
	  classifier:add(nn.View(nbr_input))
	  classifier:add(nn.Linear(nbr_input, nbr_output))
	  return classifier
	end

	-- Creates a spatial transformer module
	-- locnet are the parameters to create the localization network
	-- rot, sca, tra can be used to force specific transformations
	-- input_size is the height (=width) of the input
	-- input_channels is the number of channels in the input
	-- no_cuda due to (1) below, we need to know if the network will run on cuda
	function networks.new_spatial_transformer(locnet, rot, sca, tra,
											 input_size, input_channels,
											 no_cuda)
	  input_size = input_size or networks.base_input_size
	  input_channels = input_channels or 3
	  require 'stn'
	  local nbr_elements = {}
	  for c in string.gmatch(locnet, "%d+") do
		nbr_elements[#nbr_elements + 1] = tonumber(c)
	  end


	  -- Get number of params and initial state
	  local init_bias = {}
	  local nbr_params = 0
	  if rot then
		nbr_params = nbr_params + 1
		init_bias[nbr_params] = 0
	  end
	  if sca then
		nbr_params = nbr_params + 1
		init_bias[nbr_params] = 1
	  end
	  if tra then
		nbr_params = nbr_params + 2
		init_bias[nbr_params-1] = 0
		init_bias[nbr_params] = 0
	  end
	  if nbr_params == 0 then
		-- fully parametrized case
		nbr_params = 6
		init_bias = {1,0,0,0,1,0}
	  end

	  local st = nn.Sequential()

	  -- Create a localization network same as cnn but with downsampled inputs
	  local localization_network = nn.Sequential()
	  local conv1 = networks.new_conv(input_channels, nbr_elements[1], false, true)
	  local conv2 = networks.new_conv(nbr_elements[1], nbr_elements[2], false, true)
	  local conv_output_size = networks.convs_noutput({conv1, conv2}, input_size/2)
	  local fc = networks.new_fc(conv_output_size, nbr_elements[3])
	  local classifier = networks.new_classifier(nbr_elements[3], nbr_params)
	  -- Initialize the localization network (see paper, A.3 section)
	  classifier:get(2).weight:zero()
	  classifier:get(2).bias = torch.Tensor(init_bias)

	  localization_network:add(networks.modules.poolingModule(2,2,2,2))
	  localization_network:add(conv1)
	  localization_network:add(conv2)
	  localization_network:add(fc)
	  localization_network:add(classifier)

	  -- Create the actual module structure
	  local ct = nn.ConcatTable()
	  local branch1 = nn.Sequential()
	  branch1:add(nn.Transpose({3,4},{2,4}))
	  if not no_cuda then -- see (1) below
		branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
	  end
	  local branch2 = nn.Sequential()
	  branch2:add(localization_network)
	  branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))
	  branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
	  if not no_cuda then -- see (1) below
		branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
	  end
	  ct:add(branch1)
	  ct:add(branch2)

	  st:add(ct)
	  local sampler = nn.BilinearSamplerBHWD()
	  -- (1)
	  -- The sampler lead to non-reproducible results on GPU
	  -- We want to always keep it on CPU
	  -- This does no lead to slowdown of the training
	  if not no_cuda then
		sampler:type('torch.FloatTensor')
		-- make sure it will not go back to the GPU when we call
		-- ":cuda()" on the network later
		sampler.type = function(type)
		  return self
		end
		st:add(sampler)
		st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
	  else
		st:add(sampler)
	  end
	  st:add(nn.Transpose({2,4},{3,4}))

	  return st
	end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))

  print(model)
     model:add(networks.new_spatial_transformer('64,300,64', nil,nil,nil, 56, 64, false))
	  
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 8))
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
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

   print(model)
   return model
end

return createModel
