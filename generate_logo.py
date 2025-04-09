import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Create a directory if it doesn't exist
if not os.path.exists('Resources'):
    os.makedirs('Resources')

# Set up image dimensions
width, height = 1024, 1024
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 2 - 40

# Create a transparent RGBA image
img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Define colors
purple = (128, 0, 255, 255)  # Vibrant purple
green = (0, 255, 128, 255)   # Vibrant green
dark_purple = (64, 0, 128, 255)
dark_green = (0, 128, 64, 255)

# Create a circular gradient background
for y in range(height):
    for x in range(width):
        # Calculate distance from center
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        if distance < radius:
            # Normalize distance to 0-1
            normalized_dist = distance / radius
            
            # Create circular gradient with transparency
            alpha = 255 - int(normalized_dist * 200)
            
            # Calculate angle for color variation
            angle = math.atan2(y - center_y, x - center_x)
            angle_normalized = (angle + math.pi) / (2 * math.pi)
            
            # Alternate between green and purple based on angle
            if (angle_normalized > 0.25 and angle_normalized < 0.5) or (angle_normalized > 0.75 and angle_normalized < 1.0):
                r = int(purple[0] * (1 - normalized_dist) + dark_purple[0] * normalized_dist)
                g = int(purple[1] * (1 - normalized_dist) + dark_purple[1] * normalized_dist)
                b = int(purple[2] * (1 - normalized_dist) + dark_purple[2] * normalized_dist)
            else:
                r = int(green[0] * (1 - normalized_dist) + dark_green[0] * normalized_dist)
                g = int(green[1] * (1 - normalized_dist) + dark_green[1] * normalized_dist)
                b = int(green[2] * (1 - normalized_dist) + dark_green[2] * normalized_dist)
            
            img.putpixel((x, y), (r, g, b, alpha))

# Draw audio wave patterns
wave_count = 7
for i in range(wave_count):
    # Calculate wave parameters
    amplitude = radius * 0.1 * (wave_count - i) / wave_count
    frequency = 20 + i * 5
    phase = i * math.pi / wave_count
    thickness = int(5 + 15 * (wave_count - i) / wave_count)
    
    # Alternate colors
    color = purple if i % 2 == 0 else green
    
    # Draw wave
    points = []
    for angle in range(0, 360, 2):
        rad = math.radians(angle)
        wave_offset = amplitude * math.sin(frequency * rad + phase)
        wave_radius = radius * 0.4 + wave_offset + i * radius * 0.07
        x = center_x + wave_radius * math.cos(rad)
        y = center_y + wave_radius * math.sin(rad)
        points.append((x, y))
    
    # Connect last point to first
    points.append(points[0])
    
    # Draw the wave as a line
    for j in range(len(points) - 1):
        draw.line([points[j], points[j+1]], fill=color, width=thickness)

# Add some fractal-like elements
for _ in range(50):
    start_angle = random.uniform(0, 2 * math.pi)
    length = random.uniform(radius * 0.2, radius * 0.7)
    x1 = center_x + math.cos(start_angle) * length * 0.5
    y1 = center_y + math.sin(start_angle) * length * 0.5
    x2 = center_x + math.cos(start_angle) * length
    y2 = center_y + math.sin(start_angle) * length
    
    # Choose color based on position
    if random.random() > 0.5:
        color = (purple[0], purple[1], purple[2], 150)
    else:
        color = (green[0], green[1], green[2], 150)
    
    draw.line([(x1, y1), (x2, y2)], fill=color, width=3)

# Add central bloom effect
center_radius = radius * 0.25
draw.ellipse((center_x - center_radius, center_y - center_radius, 
              center_x + center_radius, center_y + center_radius), 
             fill=(255, 255, 255, 100))

# Apply blur for a softer look
img = img.filter(ImageFilter.GaussianBlur(radius=3))

# Save the image
img.save('Resources/logo_placeholder.png')

print("Logo saved as Resources/logo_placeholder.png")

