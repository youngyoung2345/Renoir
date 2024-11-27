# Renoir
## Renoir is project whose goal is artstyle recognition

### Problem Definition

 For creators, "copying" is not a trivial matter. In fact, there is a case where an illustrator suffered a disruption to their livelihood due to their style being imitated.
 
 Generative AI creates new data based on the datasets it was trained on. One could say that it generates new data by copying existing ones. Therefore, it is nearly impossible for creators to view generative art AI in a positive light.
 
 As a result, it became clear that a program capable of verifying whether a generative art AI has learned a specific artistic style is needed. However, Project Renoir is primarily focused on enabling AI to recognize art styles.


### Architecture

<img src="https://github.com/user-attachments/assets/bc1648f7-8c5d-423f-89bf-9a7f71140063" width="150"/>

Backbone : VGG19 without fully connected layer
Optimizer : Adam with learning rate 0.01

