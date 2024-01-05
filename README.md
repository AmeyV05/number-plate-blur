# Number Plate Blur

This repository contains the model to blurr the number plates on the cars. Currently, the Pegaus office blurring is not of good quality and PM blurs all cars.
Plus, it is extremely slow (~ 1 day for a project of 30k images)
We build a custom model which is trained on Dutch number plates and does deblurring.

For this we will use Detectron2

# Installation

`conda create -n number-plate python=3.10` 
`conda activate number-plate`

`pip install -r requirements.txt`
`pip install git+https://github.com/facebookresearch/detectron2.git`

