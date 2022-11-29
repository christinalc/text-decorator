from PIL import Image,ImageFont,ImageDraw
import os
import shutil
import gradio as gr
from gradio.mix import Series

#draw an input image based off of user's text input

def drawImage(text, font): #add another argument for prompt later
    out = Image.new("RGB", (512, 512), (0, 0, 0))
    #move font to font-directory 
    print("test")
    #print (os.path.abspath(str(type(font))))
    srcPath = './app.py' + (str(type(font)))
    print(srcPath)
    shutil.move(srcPath,'font-directory')
    
    fnt = ImageFont.truetype(font, 40)
    d = ImageDraw.Draw(out)
    #d.multiline_text((10, 64), text, fill=(255, 255, 255))
    d.multiline_text((10, 64), text, font=fnt, fill=(0, 0, 0))
    out.show()
    return out

demo = gr.Interface(
    title="AI text decorator",
    description="christina",
    fn=drawImage, 
    inputs=[
        gr.Textbox(placeholder="shift + enter for new line",label="what do you want to say?"),
        "file"
        #gr.Textbox(placeholder="prompt",label="how does your message look and feel?") #figure out models in series 
        ],
    outputs="image")
demo.launch()