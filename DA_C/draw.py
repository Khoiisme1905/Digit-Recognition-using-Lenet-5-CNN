import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import subprocess
import os

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")
        
        # Canvas để vẽ
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        # Nút điều khiển
        self.controls = ttk.Frame(root)
        self.controls.pack(pady=20)
        
        ttk.Button(self.controls, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # Nhãn kết quả
        self.result_label = ttk.Label(root, text="Draw a digit (0-9)")
        self.result_label.pack(pady=20)
        
        # Biến để vẽ
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind sự kiện chuột
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)
        self.old_x = None
        self.old_y = None
    
    def paint(self, event):
        if self.old_x and self.old_y:
            # Vẽ trên canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                  width=20, fill='black', capstyle=tk.ROUND, 
                                  smooth=tk.TRUE)
            # Vẽ trên image
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill='black', width=20)
        
        self.old_x = event.x
        self.old_y = event.y
    
    def reset_coordinates(self, event):
        self.old_x = None
        self.old_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit (0-9)")
    
    def predict(self):
        # Resize về 28x28 pixels
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        # Lưu ảnh
        img_resized.save("input.png")
        
        # Chuyển đổi ảnh thành format MNIST
        img_array = np.array(img_resized)
        # Normalize và invert màu
        img_array = 255 - img_array
        
        # Lưu thành file binary format giống MNIST
        with open("input.idx3-ubyte", "wb") as f:
            # Header của format MNIST
            f.write(bytes([0,0,8,3,0,0,0,1,0,0,0,28,0,0,0,28]))
            f.write(img_array.tobytes())
        
        # Gọi chương trình C
        try:
            result = subprocess.check_output(["./digit_predict"])
            prediction = result.decode().strip()
            self.result_label.config(text=f"Predicted: {prediction}")
        except subprocess.CalledProcessError as e:
            self.result_label.config(text="Error in prediction")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()