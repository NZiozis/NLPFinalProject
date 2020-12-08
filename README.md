# NLP Final Project

Notes:
You can use pip install torch to install pytorch. The version that gets installed will depend on whether you have a GPU+CUDA enabled machine. The same code should work with multiple pytorch versions but I have been using version 1.6.0: https://pytorch.org/docs/1.6.0/ 

# Training original sentence encoder model on Recipe1M

python train_ce.py --vocab_ing data/new_vocab_ing_3769.pkl --batch_size 2

# Training with only sentence encoder

python train_baseline.py --batch_size 1

# Training video encoder and sentence encoder jointly

python train_joint_model.py --batch_size 1