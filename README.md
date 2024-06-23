# Renoir

Renoir is **a project aimed at recognizing the styles of digital artwork**. It originated as an assignment for the 'Introduction to Software Convergence' course at Kyunghee University's Department of Software Convergence.


## Why do we need a model that can recognize style of digital artwork?

Downplaying the issue of 'copying' is not something artists take lightly. There are actual cases where illustrators have suffered financially because their artistic style was mimicked.

Generative AI creates new data based on datasets used in its training. It can be seen as generating new data by copying existing ones. Therefore, it is nearly impossible for creators to view generative art AI positively.

As a result, we concluded that a program capable of verifying whether generative art AI has learned a specific artistic style is needed. However, Project Renoir initially focuses on enabling artificial intelligence to recognize artistic styles.


## Renoir 1.0
<img src="https://github.com/youngyoung2345/Renoir/assets/134286859/eba7735c-b09c-4cd5-9d18-052493252d5f.png" width="222" height="247.5"/>
- Backbone : VGG16
- Cost function : Binary Cross Entropy

## Renoir 2.0
- Backbone : Inception
- Cost function : Binary Cross Entropy
- Sadly, a code of Renoir 2.0 doesn't exist.

## Renoir 3.0
- Backbone : VGG19 without fully connected layer
- Cost function : ArcFace
