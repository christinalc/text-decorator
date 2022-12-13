import os
import shutil
import torch
import gradio as gr
MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')

from PIL import Image,ImageFont,ImageDraw
from gradio.mix import Series
#from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

YOUR_TOKEN=MY_SECRET_TOKEN
device="cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=YOUR_TOKEN)
pipe.to(device)

#draw an image based off of user's text input

def drawImage(text, text_size, prompt, strength, guidance_scale): #(text, text_size, font)
    out = Image.new("RGB", (512, 512), (0, 0, 0))
    #add some code here to move font to font-directory   
    font = './font-directory/DimpleSans-Regular.otf'
    fnt = ImageFont.truetype(font, int(text_size))
    d = ImageDraw.Draw(out)
    d.multiline_text((16, 64), text, font=fnt, fill=(255, 255, 255))

    #init_image = out
    out.save('initImage.png')
    images = []
    images = pipe(prompt=prompt, image=out, strength=strength, guidance_scale=guidance_scale).images
    #images[0].save = ("image.png")
    #images = []
    #images.append(out)
    #out.show()
    return images[0]

#def newImage(image, prompt):
    
    #return images

#drawImage = gr.Interface(fn=drawImage, inputs=gr.Textbox(placeholder="shift + enter for new line",label="what do you want to say?"),outputs="image")
#newImage = gr.Interface(fn=newImage,inputs=[gr.Textbox(placeholder="prompt",label="how does your message look and feel?")],outputs="image")

#demo = gr.Series(drawImage,newImage)

demo = gr.Interface(
    title="this will be very slow since it's on CPU! msg me if you would like to try",
    description="turn on acceleration from7-9,google form",
    #theme='huggingface',
    css="""
    body {system-ui, Helvetica, Arial, sans-serif}
    """
    fn=drawImage, 
    inputs=[
        gr.Textbox(placeholder="shift + enter for new line",label="what do you want to say?"),
        #"file"
        gr.Number(label="text size",value=240),
        gr.Textbox(placeholder="eg. imagery, art style, materials, emotions",label="how does your message look and feel?"), #figure out models in series 
        gr.Slider(label="strength (how much noise will be added to the input image)",minimum=0, maximum=1, step=0.01, value=0.7),
        gr.Slider(label="guidance scale (how much the image generation follows the prompt)",value=15, maximum=20),
        ],
    outputs="image")
demo.launch()