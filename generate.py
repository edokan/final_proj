# -*- coding: utf-8 -*- 

#------------------------------------ Imports ----------------------------------#

# Import python imaging libs
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Import operating system lib
import os

# Import random generator
from random import randint

#------------------------------------ Cleanup ----------------------------------#
                    
def Cleanup():    
    # Delete ds_store file
    if os.path.isfile(font_dir + '.DS_Store'):
        os.unlink(font_dir + '.DS_Store')
    
    # Delete all files from output directory
    for file in os.listdir(out_dir):
        file_path = os.path.join(out_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    return

#------------------------------ Generate Characters ----------------------------#

def GenerateCharacters():
    # Counter
    k = 1
    # For each character do
    for i in range(len(characters)):
        os.mkdir(out_dir +str(i))
        # Process the font files
        for dirname, dirnames, filenames in os.walk(font_dir):
            # For each font do
            for filename in filenames:
                # Get font full file path
                font_resource_file = os.path.join(dirname, filename)
                # For each font size do
                for font_size in font_sizes:
                    if font_size > 0:
                        # For each background color do
                        for background_color in background_colors:
                            # Convert the character into unicode
                            character = str(characters[i])
            
                            # Create character image : 
                            # Grayscale, image size, background color
                            char_image = Image.new('L', (image_size, image_size),\
                            background_color)
            
                            # Draw character image
                            draw = ImageDraw.Draw(char_image)
            
                            # Specify font : Resource file, font size
                            font = \
                            ImageFont.truetype(font_resource_file, font_size)
            
                            # Get character width and height
                            (font_width, font_height) = font.getsize(character)
            
                            # Calculate x position
                            x = (image_size - font_width)/2
            
                            # Calculate y position
                            y = (image_size - font_height)/2
            
                            # Draw text : Position, String, 
                            # Options = Fill color, Font
                            draw.text((x, y), character, (245-background_color) + \
                            randint(0, 10) , font=font)
                    
                            # Final file name                    
                            file_name = out_dir + str(i) + '/' + str(k) + '.png'
            
                            # Save image
                            char_image.save(file_name)
                    
                            # Print character file name
                            print(file_name);
                    
                            # Increment counter
                            k = k + 1
    return

#---------------------------------- Input and Output ---------------------------#

# Directory containing fonts
font_dir = 'test/'

# Output
out_dir = 'data/'

#------------------------------------ Characters -------------------------------#

# Numbers
numbers = [c for c in '0123456789']

# Small letters
small_letters = [c for c in 'abcdefghijklmnopqrstuvwxyz']

# Capital letters
capital_letters = [c for c in'abcdefghijklmnopqrstuvwxyz'.upper()]
        
# Select characters
characters = numbers + small_letters + capital_letters

#------------------------------------- Colors ----------------------------------#

# Background color
# white_colors = (215, 225, 235, 245)
# black_colors = (0, 10, 20, 30)
# gray_colors = (135, 145, 155)

# background_colors = white_colors + black_colors + gray_colors
background_colors = (245,)
        
#-------------------------------------- Sizes ----------------------------------#

# Character sizes
small_sizes = (12, 14, 16) #, 12, 16)
medium_sizes = (18, 20, 22, 24) #, 24, 28)
large_sizes = (26, 28, 30, 32) #, 36, 40)

font_sizes = small_sizes + medium_sizes + large_sizes
        
# Image size
image_size = 32

#-------------------------------------- Main -----------------------------------#

# Do cleanup
Cleanup()

# Generate characters
GenerateCharacters()