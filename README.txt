Instruction for Residual Networks with 2 Spatial Transformers

Written by Xingfang Yuan, University of Missouri
xyuan@mail.missouri.edu

The Code is written based on the Github respository: 
https://github.com/facebook/fb.resnet.torch.git
https://github.com/qassemoquab/stnbhwd


Requirements

1. Install Torch

	If you haven't got Torch installed on your device, please follow the three commands for installing Torch on Ubuntu 12+ or Mac OS X: 
	(Torch will be installed to your home folder in ~/torch)
		# in a terminal, run the commands
		git clone https://github.com/torch/distro.git ~/torch --recursive
		cd ~/torch; bash install-deps;
		./install.sh

	The first script installs the basic package dependencies that LuaJIT and Torch require. 
	The second script installs LuaJIT, LuaRocks, and then uses LuaRocks (the lua package manager) to install core packages like torch, nn and paths, as well as a few other packages.

	The above script should adds torch to your PATH variable once finish installation. You need to run the following script to refresh your env variables.

		# On Linux with bash
		source ~/.bashrc
		# On Linux with zsh
		source ~/.zshrc
		#On OSX or in Linux with none of the above.
		source ~/.profile

	For Torch uninstallation, simply run:

		rm -rf ~/torch

	For additional information about Torch, please refer to http://torch.ch for further help.

2. Install cuDNN v4 and the Torch cuDNN bindings

For installation instructions for step 1&2, please refer to ./DocsForResnet/INSTALL.md in the folder.


3. Data Preparing

	Preprocessed data are stored at the following link:

	https://www.dropbox.com/sh/swfm0q6aa52vu36/AABIWn6EqyO5XVWMlLV7hJcUa?dl=0

	All images are resized to 256*256 and cropped so that the bounding box region is placed at center of each image and sized 224*224.
	Each folder represent one category, which forms 120 categories for Stanford Dogs Dataset and 200 for CUB-200-2011 Bird Dataset.

4. Pretrained Model

	This model is for training. Model is stored at following link:

	https://www.dropbox.com/sh/swfm0q6aa52vu36/AABIWn6EqyO5XVWMlLV7hJcUa?dl=0

	There's one pre-trained model on each dataset. The model is trained with Imagenet pre-trained original residual network model.

For training

1. Make sure the path in line 393 in ./models/pre_st_xxx.lua direct to the corresponding pre-trained model.

2. run the code in terminal with following script:

	# for Stanford Dogs Dataset
	th main.lua -data DATAPATH -netType pre_st_dogs.lua -LR 0.001
	# for CUB-200-2011 Bird Dataset
	th main.lua -data DATAPATH -netType pre_st_birds.lua -LR 0.001

	Note: DATAPATH should be the path that direct to the dataset folder, which contains 2 sub-folders called 'train' & 'val'

3. Model will be saved at the end of each epoch in ./save
   There'll be a log file named 'train&val.log' that contains the top-1 & top-5 error rate for both training and validation set of each epoch.
   
4. (Optional) If the code get stuck after creating model, please delete the file in ./gen/ and run the script again.

5. (Optional) For training on other datasets, simply add '-nClasses CLASSNUMBER' at the end of the script where CLASSNUMBER denote the number of categories in the dataset. Data should be stored in 'train' & 'val' folders and each sub-folder represent one category.




