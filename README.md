# gsproject

This real-time program is designed to detect your hand gestures (American Sign Language) and display its English translation on-screen.

# Introduction
Gesture recognition has long been an extensively researched field in the Computer Vision community. In real time, hand segmentation can be a challenging problem. Humans can easily be taught to differentiate between different objects and textures, however for machines, images are essentially just 3-dimensional arrays.

Problem Motivation
Sign language translation is an important factor that will help bridge the communication gap between deaf and hearing people. Especially through the pandemic and working fully remotely, some have had more difficulty adapting than others. On video meeting apps, such as zoom and google meeting, those with hearing imparities might have struggled communicating with others on video. This program allows easier communication while working remotely. The program can also be used to teach potential signers. The user can match each hand gesture to the letter and learn quickly using the program. 

# Overview
In order for machines to recognize images, we model the human brain using neural networks. A neural network is like a flowchart, outlining questions that eventually lead to a decision. By using neural networks and matrices, programmers are able to make a model that can detect, calculate, and solve. The accuracy and speed of AI is incomparable to traditional methods.

For this program, I referenced:

training and classifying- https://github.com/xuetsing/image-classification-tensorflow

hand detection and screen display - https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

dataset- https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

how to run:

```shell
git clone https://github.com/26selinaw/gsproject
```

```shell
cd gsproject
```

```shell
python3 run.py
```

