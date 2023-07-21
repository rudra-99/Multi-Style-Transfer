# Multi-Style-Transfer
NST is a technique by which we can render a picture of our choice with the styles from another image, maybe an artwork. Here is a summarised version of the whole process. You take a picture of your choice and this gets set to "content image" and then you choose another image, preferably an artwork and this gets set to "Style image". The neural network then learns the elements of both the content image and the style image and then updates the pixel values of a random noise with the graadients learned at each iteration. The loss function is two parter - one is the content loss measured with the mean squared error between the corresponding pixel values of the noise and the content image and the second one is the style loss which is measured with the mean squared error between the elements of the gram matrix obtained from the style image and the content image. The content loss is a measure of the amount of information lost while learning the features of the image and the style loss measures the amount of style lost while rendering the image with that particular style. The final loss is a weighted sum of the content and style loss. To understand the full working of the NST, it is encouraged to read the paper " A neural algorithm of artistic style " by L.A Gatys et al. 

MST is an extension of the work previously done by LA Gatys. The whole idea of having the image rendered with one style is a bit restricting. So MST is a way of rendering images with multiple artworks. The work is still in its intial phase and this one is just an overview of the idea. Feel free to share your feedbacks or ideas.


<table>
  <tr></td>
     <td>Content Image</td>
     <td>Stylised Image: with  3 Vangogh artworks</td>
  </tr>
  <tr>
    <td><img src="https://github.com/rudra-99/Multi-Style-Transfer/assets/107755049/c00eb34e-1603-4546-8514-1111671d826d" width = "500" height = "400"></td>
    <td><img src = "https://github.com/rudra-99/Multi-Style-Transfer/assets/107755049/1dd828c0-4040-4c74-9ba7-2dc55d5b3548" width = "500" height = "400"></td>
  </tr>
 </table>

To run the code and see how it works follow the steps:

Open terminal and navigate to the folder where you've downloaded the files

```
cd " << copy and paste the file location here >> "
```
Then activate the conda environment. N.B : Tensorflow is required 

```
conda activate <<type/ copy paste the name of the conda environment>>
```
Then just run the main file
```
python mst.py
```
Select the file you want to stylise and then you will be asked to choose an artist ( whose styles you want your image to get rendered with )
For the time being I have just included the styles "picasso" and "vangogh", so you have to choose from one of these two. You can also keep on decorating and also add "True"/ "False" for the Augment

Below is a screen shot which summarises the process pretty much

<img width="1440" alt="Screenshot 2023-07-21 at 10 45 47 AM" src="https://github.com/rudra-99/Multi-Style-Transfer/assets/107755049/6a119b24-d7d3-4652-9871-573f72c0bf79">

