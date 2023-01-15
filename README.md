# Bone Age Maturity Estimation using Lateral Cephalogram X-ray Image
In this repository we collect the tools for the development and deployment of the X-ray image analysis for the purpose of Boen Age Maturity Estimation. Please follow the instructions below for contributing or using the already developed tools.

**Note!** whenever you see a `.keep` file inside a directory, ignore it. It is used only for adding empty directories to git.

**Note!** Do not commit any of the dataset files to git as it slows down the interaction with Github. Also, once we track data with git and push them to Github, we violate the data security principles! For data tracking, we may need to use other advanced tools and techniques. More on this to come if it becomes a bottleneck!

# Clone and branch steup
Clone the repo first. Then, create your own git branch and start developing on your own branch.

# Data download
You need to down load the data yourself and place them into the correct place as described below. Please note that you need to be part of the project team in order to gain access to some of the data we have gathered.

## Raw compressed files
Download the Raw Compressed files of the 3 datasets we have included in this project from the links below and place them into the `data/raw/` directory which should be an empty directory.

* first dataset: 
A collection of images which are collected in Isfahan University of Medical Sciences and annotated by our project team.
Link: https://drive.google.com/drive/folders/1ikZlD2ID3apc1rgsEvDTnzQmnu7Ail41?usp=share_link

* second dataset:
A collection of images which are collected from a radiology clinic in Tehran and will be annotated by our project team.
Link: https://drive.google.com/drive/folders/1oUmCvSRWKGdSsM_Obh-IHIDVhfshzbj8?usp=share_link

* third dataset:
A collection of images which are collected from open-source ISBI challenge 2014 and 2015 targeted at the detection of anatomical landmarks of the facial skeleton seen through lateral cephalogram X-ray images.
Link: https://drive.google.com/file/d/1X51BJFsHnTX4-DJopmDsScZ2MPo865dt/view?usp=share_link

**Note!** This third dataset was once uncompressed and transformed for annotation purposes. If you are a member of the project and see those transformed images inside the google drive of the project, be aware of this. Most porbably, we use the transformed version of the third dataset. However, if you would like to access the very original dataset, feel free to download it from the provieded link.

MORE DATASETS TO COME!! :)

# Experimentation and development
Please create your own experimentation and development in jupyter notebooks and place them inside the `notebooks/` directory. Once we are done with our experimentation and development, we can move the source code to the `src/` directory. The code which will be stored in the `src/` directory will be packaged.

## Code config
It is best to separate the configuration and hyperparameters from the source code. Please, use the `code_config/` directory for storing these configuration files.
