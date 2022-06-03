# VICO_classifier

## Classification methods

Once that we have spotted every person within frame, we may classify them into these categories:
 - gender (binary, born to)
 - age range
 - customer or employee

`VICO` works with three different classification methods:
- Custom CNN models based on 'A Convnet for the 2020s' [https://arxiv.org/abs/2201.03545]
- Feature embedding classifiers, implementation of Online Triple Siamese Networks. Check our own [Triple Siamese Network](https://github.com/LyticaMx/siamese-networks-tf) models repo for further docu!
 - [Jetson ImageNet] based classifiers
 - [PyTorch native classifiers](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
 
