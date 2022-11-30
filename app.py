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
device="cpu"
img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=YOUR_TOKEN)
img_pipe.to(device)

#draw an image based off of user's text input

def drawImage(text, prompt): #(text, font, prompt)
    out = Image.new("RGB", (512, 512), (0, 0, 0))
    #add some code here to move font to font-directory 
    
    font = './font-directory/DimpleSans-Regular.otf'
    fnt = ImageFont.truetype(font, 160)
    d = ImageDraw.Draw(out)
    d.multiline_text((10, 64), text, font=fnt, fill=(255, 255, 255))

    init_image = out
    out_image = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)
    #out.show()
    return out_image

demo = gr.Interface(
    title="AI text decorator",
    description="christina",
    fn=drawImage, 
    inputs=[
        gr.Textbox(placeholder="shift + enter for new line",label="what do you want to say?"),
        #"file"
        gr.Textbox(placeholder="prompt",label="how does your message look and feel?") #figure out models in series 
        ],
    outputs="image")
demo.launch()