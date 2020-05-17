# Gesture-Character-Recognition
A simple computer vision model to predict the English letters and words drawn as an air gesture <br>
This model uses Convolution Neural networks for predicting the drawn letter and open cv for live camera operations .
You will need a point object of a specific color to draw the letter(could be a pen cap) for eg. in this i have used a blue color pen-cap.
you can change it by changing the high and low RGB value in the color Boundry in the GCR.py<br>
For training the model run model_training.py<br>
Run GCR.py for running the model <br>

In the camera feed ,take the object near the camera as the model to detect it and draw the letter.<br>

Currently it sees both the capital and small letters as one and the same .As you will draw the letter it will predict the letter and it will give a speech output of it.Keep drawing the letters to create a word and press q to terminate .It will Give the speech output for the word also.You can also train it for more epochs to get the higher accuracy
