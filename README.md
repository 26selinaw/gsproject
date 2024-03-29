# gsproject

I have always wanted to experiment with AI and gesture recognition. I had some experience with python before, however, the development of this program took quite some time. Though the process was challenging, I was able to finish a completed program and I hope to further improve the speed and accuracy. This article describes the code in detail and explains the ups and downs as this program was created. The finished real-time program is designed to detect your hand gestures (American Sign Language) and display its English translation on-screen.

# Introduction
Gesture recognition has long been an extensively researched field in the Computer Vision community. In real time, hand segmentation can be a challenging problem. Humans can easily be taught to differentiate between different objects and textures, however for machines, images are essentially just 3-dimensional arrays.

Problem Motivation
Sign language translation is an important factor that will help bridge the communication gap between deaf and hearing people. Especially through the pandemic and working fully remotely, some have had more difficulty adapting than others. On video meeting apps, such as zoom and google meeting, those who are hearing impaired might have struggled communicating with others on video. This program allows easier communication while working remotely. The program can also be used to teach potential signers. The user can match each hand gesture to the letter and learn quickly using the program. 

# Overview
In order for machines to recognize images, we model the human brain using neural networks. A neural network is like a flowchart, outlining questions that eventually lead to a decision. By using neural networks and matrices, programmers are able to make a model that can detect, calculate, and solve. The accuracy and speed of AI is incomparable to traditional methods.

For this program, I referenced:

training and classifying- https://github.com/xuetsing/image-classification-tensorflow

hand detection and screen display - https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

dataset- https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

how to run:

clone the repo
```shell
git clone https://github.com/26selinaw/gsproject
```

enter project directory

folder contains training/classifying folder(image-classification-tensorflow) and main program
```shell
cd gsproject
```

run the main python program(run.py)
```shell
python3 run.py
```

