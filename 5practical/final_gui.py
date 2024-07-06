#pip install numpy pandas matplotlib tensorflow opencv-python pillow
#Create the img Directory:


from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('mnist.h5')

# Set the folder to save images
image_folder = "img/"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Initialize the main Tkinter window
root = Tk()
root.resizable(0, 0)
root.title("Digit Recognizer")

# Initialize variables
lastx, lasty = None, None
image_number = 0

# Create a canvas to draw on
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Function to clear the canvas
def clear_widget():
    global cv
    cv.delete('all')

# Function to draw lines on the canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# Function to activate drawing
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

cv.bind('<Button-1>', activate_event)

# Function to recognize the drawn digit
def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'
    widget = cv

    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    
    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)
    
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        digit = padded_digit.reshape(1, 28, 28, 1).astype('float32') / 255
        pred = model.predict(digit)[0]
        final_pred = np.argmax(pred)
        data = f'{final_pred} {int(max(pred) * 100)}%'
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, data, (x, y - 5), font, 0.5, (255, 0, 0), 1)
    
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
    # Increment the image number for the next saved image
    image_number += 1

# Create buttons for recognizing the digit and clearing the canvas
btn_save = Button(text='Recognize Digit', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)

button_clear = Button(text='Clear Widget', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

# Run the Tkinter main loop
root.mainloop()
