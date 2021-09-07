# Convolutional-Neural-Network
Download MNIST data and save 'train.csv', 'valid.csv' and 'test.csv'
# To train the model
Pre-train model is available in model Folder.
To train new model
```
python3 train.py --lr 0.00005 --batch_size 250 --init 0 --save_dir 'Folder_name' --load 1
```
## Train and Valid Loss Graphs

![batch_norm](https://user-images.githubusercontent.com/17472092/132354546-91df62aa-a4de-493a-9293-7b90bf3a047f.png)
![loss](https://user-images.githubusercontent.com/17472092/132354553-a5980376-b450-4e9e-8da1-b5d8b7ae0b62.png)

# For Guided backpropogation
Guided_bkp.py will generate 20 images on which guided backpropogation is done, use same run.sh to run guided backpropogation also.

![image](https://user-images.githubusercontent.com/17472092/132364731-635bf992-ffe6-4257-ac8f-3b05e26f6e18.png)
![image](https://user-images.githubusercontent.com/17472092/132364931-5d4ede3f-22c5-4fa1-8292-4ae14a91ac1b.png)
