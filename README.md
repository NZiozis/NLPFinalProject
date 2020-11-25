# NLP Final Project

Notes:
You can use pip install torch to install pytorch. The version that gets installed will depend on whether you have a GPU+CUDA enabled machine. The same code should work with multiple pytorch versions but I have been using version 1.6.0: https://pytorch.org/docs/1.6.0/ 

# Training video encoder

python train_ce.py --vocab_ing datasets/new_vocab_ing_3769.pkl --video_encoder True --batch_size 2

TODO: Change striding in dataloader to every 5 frames. Change video encoder maxpool layer to pool over sequence dimension.