--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require 'datasets/transforms'
local imagenetLabel = require '/pretrained/imagenet'

if #arg < 2 then
   io.stderr:write('Usage: th classify.lua [MODEL] [FILE]...\n')
   os.exit(1)
end
for _, f in ipairs(arg) do
   if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
   end
end


-- Load the model
local model = torch.load(arg[1])
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local N = 5

for i=2,#arg do
   -- load the image as a RGB float tensor with values 0..1
   local img = image.load(arg[i], 3, 'float')
   local name = arg[i]:match( "([^/]+)$" )

   -- Scale, normalize, and crop the image
   img = transform(img)

   -- View as mini-batch of size 1
   local batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()

   print(model)
   local output1 = model:get(1):forward(batch:cuda())
   print(output1:size())
   local output2 = model:get(2):forward(output1)
   print(output2:size())
   local output3 = model:get(3):forward(output2)
   print(output3:size())
   local output4 = model:get(4):forward(output3)
   print(output4:size())
   local output5 = model:get(5):forward(output4)
   print(output5:size())
   local output6 = model:get(6):forward(output5)
   print(output6:size())
   local output7 = model:get(7):forward(output6)
   print(output7:size())
   local output8 = model:get(8):forward(output7)
   print(output8:size())
   local output9 = model:get(9):forward(output8)
   print(output9:size())
   local output10 = model:get(10):forward(output9)
   print(output10:size())
   local output11 = model:get(11):forward(output10)
   print(output11:size())

   


   -- Get the top 5 class indexes and probabilities
   local probs, indexes = output:topk(N, true, true)
   print('Classes for', arg[i])
   for n=1,N do
     print(probs[n], imagenetLabel[indexes[n]])
   end
   print('')

end
