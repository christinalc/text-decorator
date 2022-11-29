from PIL import Image,ImageFont,ImageDraw
import gradio as gr
from gradio.mix import Series

#draw an input image based off of user's text input

def drawImage(text, font): #add another argument for prompt
  out = Image.new("RGB", (512, 512), (255, 255, 255))
  fnt = ImageFont.truetype(font, 40)
  #console.log(fnt)
  d = ImageDraw.Draw(out)
  d.multiline_text((10, 64), text, fill=(0, 0, 0))
  #d.multiline_text((10, 64), text, font=fnt, fill=(0, 0, 0))
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