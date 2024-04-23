from flask import Flask, request, render_template, send_file
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import os
import torch
import tempfile

app = Flask(__name__)
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v6", torch_dtype=torch.float16
).to("cuda")

@app.route('/')
def index():
    # Render a form to upload an image and set parameters
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('image')
    if not uploaded_file:
        return "No file uploaded", 400
    
    # Save the uploaded image to a temporary file and get the path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        uploaded_file.save(temp_file_path)

    # Get the parameters from the form
    strength = float(request.form.get("strength", 0.3))
    guidance_scale = float(request.form.get("guidance_scale", 5.0))
    num_steps = int(request.form.get("num_steps", 30))

    # Load the image from the temporary file path
    init_image = load_image(temp_file_path).convert("RGB")
    
    # Define the prompt and negative prompt
    prompt = "3D cartoon Pixar style"
    negative_prompt = "blurry watermark blur haze filter border frame ugly hyperrealist real glitch pattern tile wallpaper"

    # Generate the output image using the specified parameters
    output_image = pipe(prompt, image=init_image, negative_prompt=negative_prompt,
                        num_inference_steps=num_steps,
                        strength=strength,
                        guidance_scale=guidance_scale).images[0]

    # Save the generated image
    output_path = "output_image.png"
    output_image.save(output_path)

    # Return the generated image to the user
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
