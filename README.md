# PyTorch-autoencoders

*I'm doing this for education and fun.*

For Final Project I've decided to use some stuff I've heard about, but not yet used: [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
After finishing project I see that PT-Lightning is quite convenient tool, but I totally misunderstood how to use it:D  
Anyway, I've got some nice experience working with it. And maybe I will even refactor this sometime=) Or at least won't make these mistakes later.

So, here are my results on all tasks and my questions on them.

## Task 1
The most painful task for me. Here I realized that I completely don't know how to calculate convolutional layer outputs. 
It revealed that I've fitted model only for *num_channels x 45 x 45* input. And it crashed on any other. (This problem will appear later again in lighter form).  
That's the reason I've removed convolutional vanilla autoencoder=( 

But the main problem is: only model that works on image reconstruction is the simplest one-layer linear autoencoder without activations.
And I *(still)* don't know, where I've mistaken. I've checked suggested articles, I've checked various implementations of vanilla AEs, all of them worked, but not for me.
Most of mine models for Task 1 reconstruct same face for every input. This was first time I hated man that does not even exist.
My assumption is that I've somewhere screwed up preprocessing of lfw-dataset. It seems that for AEs things like normalization means a LOT.

What I tried:
* Multilayer linear AE with RELUs and Sigmoid/Tanh output – bad reconstructions; sampling better that in simplest, "smiling" does not work at all.
* Multilayer linear AE without any activations – same face generating/sampling/smiling;
* Single layer linear AE with Sigmoid activation – same face generating/sampling/smiling;
* Single layer linear AE without Sigmoid activation – worked; bad sampling and very bad "smiling" (but sometimes it works). Results are in Task1_VanillaAE.ipynb
* Single/Multilayer convolutional AE with RELUs and/without Sigmoid output – worked only for *num_channels x 45 x 45* input; sampled only noise; smiles does not worked.

### Questions:
1. What's my problem here?:D I know, that all approaches listed above should work in some ways.
2. What is optimal architecture for image reconstruction? Should we use here complicated Segnet-like stuff?

## Task 2
This task was pain again. I really haven't sleep for a week doing this. Models was still reconstructing same "Valera's" face. But it seems, that everything was ok:D
My observations here (thanks to @Ariadne from Telegram chat): VAEs shouldn't reconstruct images well. And this makes sense since we're "feeding" **random** vectors into them.
It revealed that this VAE can handle sampling pretty well! It generates quite diverse faces from random vectors. But again, it reconstructs same 'Valera' most of the time.  
But you will not see it, since I've done it in playground mode, when I finished this task on MNIST dataset with which this model managed extremely well.

### Important issue:
My misunderstanding of convolutional outputs gives us restriction: input shape **must** be divisible by four.
I will fix this sometime. It's the most shameful thing in my work in my opinion. Sorry for this:D

### Questions:
1. Is it correct that VAEs can't handle image reconstruction as well as vanilla AEs?
2. Why model can generate various faces, but reconstructs one and the same?
3. Why model can almost perfectly reconstruct MNIST digits, but can't reconstruct face? Is it preprocessing issue?
4. When can we use MSE as reconstruction loss? I've tried and it worked well. But after googling I understood that I don't understand anything.

## Task 3, Task 4
No pain, only relax and fun. Models are ready, and we can finally play with them=)
CVAE architecture taken from Task 2 (*with same restriction*).
Result quality on Task 4 can be not so good because of training just for 50 epochs.  
This model gives nice results for lfw-dataset only after ~500 epochs (checked in playground mode).


## What I wanted to do
* Try PyTorh Lightning – **done** in inconvenient way (however, according to docs);
* Try logging into TensorBoard – **done**; really cool stuff;
* Split my project into modules – **not done**; main idea was to have base AE class and do some inheritance, nice 'train.py' scripts, use parser and so on, like I did for CycleGAN.
But I even can't extract my models into "models/" directory, because they were losing access to "datasets.py". I need to check how to import modules from parent dir.
* Know about autoencoders – **particularly done**; I can use them now, I can indentify tasks for AEs, but it is ***very*** difficult for me to debug them.
* Make different AEs for same tasks, compare them, make analysis – **not done**; most of them didn't work:D
___
*P.S.:* in 3/4 tasks we were asked to do logging of train-val stuff. I've decided to try TensorBoard for this. I didn't regret: it is really nice tool.  
And it's convenient to use with PyTorch Lightning=)
So you won't see plots and reconstructed images in notebooks outputs as long as they were logged into TensorBoard. Maybe I'll attach logs separately.
*P.P.S.:* thanks for task=) It was nice to work on it.
