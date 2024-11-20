# 🎨 Renoir

Renoir is **a project aimed at recognizing the styles of digital artwork**. It originated as an assignment for the 'Introduction to Software Convergence' course at Kyunghee University's Department of Software Convergence.


## Why do we need a model that can recognize style of digital artwork?

Downplaying the issue of 'copying' is not something artists take lightly. There are actual cases where illustrators have suffered financially because their artistic style was mimicked.

Generative AI creates new data based on datasets used in its training. It can be seen as generating new data by copying existing ones. Therefore, it is nearly impossible for creators to view generative art AI positively.

As a result, we concluded that a program capable of verifying whether generative art AI has learned a specific artistic style is needed. However, Project Renoir initially focuses on enabling artificial intelligence to recognize artistic styles.


## Renoir 1.0
<div align="center">
  <img src="https://github.com/youngyoung2345/Renoir/assets/134286859/5f7aae94-ceb3-4314-b0de-8468695820c0.png" width="222" height="247.5"/>
</div>

- Backbone : VGG16
- Cost function : Binary Cross Entropy

## Renoir 2.0

<div align="center">
  <img src="https://github.com/youngyoung2345/Renoir/assets/134286859/eba7735c-b09c-4cd5-9d18-052493252d5f.png" width="222" height="247.5"/>
</div>


- Backbone : Inception
- Cost function : Binary Cross Entropy
- Sadly, a code of Renoir 2.0 doesn't exist.

## Renoir 3.0

<div align="center">
  <img src="https://github.com/youngyoung2345/Renoir/assets/134286859/2e347cda-ab0f-419f-9fca-5577aa9a1a26" width="108.8" height="119.1"/>
</div>


- Backbone : VGG19 without fully connected layer and with Batch Normalization
- Cost function : CosFace
- You can download the .pth file at this link : https://drive.google.com/file/d/11rjLBSiwp_CJ8oYXAP0U6Q9O3VMC8fZ3/view?usp=sharing

### Renoir 3.0 Accuracy Graph

**First Graph**
- It includes training accuracy and test accuracy during epochs 1 to 22.

**Second Graph**
- It includes training accuracy and test accuracy during epochs 21 to 28.

<figure class="half"> 
  <a href="link"><img src="https://github.com/youngyoung2345/Renoir/assets/134286859/a3785b05-e2a0-46ae-a0ab-32b141ba98b0" width="411.75" height="243.25"></a>  
  <a href="link"><img src="https://github.com/youngyoung2345/Renoir/assets/134286859/b1cc111d-7f48-4c30-afb9-01be9a7d7bfc" width="411.75" height="243.25"></a>  
</figure>

if you want to read more information, visit my webpage.
Please note that the posts on my webpage are written in Korean.

link : https://carpal-money-c50.notion.site/Renoir-362407490c6b403681048c471e52226f?pvs=4

Maybe I will create a GUI for Renoir soon...
