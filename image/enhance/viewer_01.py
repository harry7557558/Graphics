import os
import sys
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import argparse
from pathlib import Path

# Import from training script
from train_01 import UNet

class ImageEnhancerGUI:
    def __init__(self, model_path, default_image_path=None):
        self.setup_gui()
        self.load_model(model_path)
        
        # Default image if provided
        if default_image_path and os.path.exists(default_image_path):
            self.load_and_process_image(default_image_path)
        else:
            self.display_no_image_message()
        
        # Setup drag and drop
        self.setup_drag_drop()
        
    def setup_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Image Quality Enhancement Viewer")
        self.root.geometry("1200x800")
        
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image_dialog)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Create a frame for the matplotlib figure
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        
        # Create two subplots
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Turn off axis
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # Set titles
        self.ax1.set_title('Original Image', fontsize=12)
        self.ax2.set_title('Enhanced Image', fontsize=12)
        
        # Add the figure to the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        
        # Setup status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Store image data
        self.original_img = None
        self.enhanced_img = None
        
        # Configure event handlers for interactive zoom and pan
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.pressed = False
        self.x_prev = 0
        self.y_prev = 0
        
    def setup_drag_drop(self):
        # Enable drag and drop on Windows
        if os.name == 'nt':  # Windows
            self.root.drop_target_register('DND_Files')
            self.root.dnd_bind('<<Drop>>', self.handle_drop)
        else:  # For Linux/Mac, we can use TkDnD
            try:
                self.root.tk.eval('package require tkdnd')
                self.root.tk.call('tkdnd::drop_target', 'register', self.root, ('DND_Files',))
                self.root.tk.call('bind', self.root, '<<Drop>>', 
                                 self.root.register(self.handle_drop) + ' %D')
            except tk.TclError:
                # TkDnD not available, fallback to manual file open
                self.status_bar.config(text="Drag and drop not supported. Use File > Open instead.")
    
    def handle_drop(self, event):
        if os.name == 'nt':  # Windows
            file_path = event.data
        else:  # Linux/Mac
            file_path = event  # In TkDnD, the event is the file path
        
        if isinstance(file_path, str) and os.path.isfile(file_path):
            self.load_and_process_image(file_path)
        else:
            messagebox.showerror("Error", "Invalid file dropped")
    
    def load_model(self, model_path):
        try:
            # Check if CUDA is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.status_bar.config(text=f"Using device: {self.device}")
            
            # Load the saved model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration from checkpoint
            if 'num_hiddens' in checkpoint:
                num_hiddens = checkpoint['num_hiddens']
            else:
                # Default configuration if not found
                num_hiddens = [16, 32, 48, 64]
            print('num_hiddens:', num_hiddens)
            
            # Create model
            self.model = UNet(3, num_hiddens)
            
            # Handle both DataParallel and regular model checkpoints
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if it exists (from DataParallel)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.status_bar.config(text=f"Model loaded successfully from {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_bar.config(text=f"Error loading model: {str(e)}")
    
    def open_image_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:
            self.load_and_process_image(file_path)
    
    def display_no_image_message(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.text(0.5, 0.5, "No Image Loaded\nUse File > Open or Drag & Drop", 
                     ha='center', va='center')
        self.ax2.text(0.5, 0.5, "Enhanced Image will appear here", 
                     ha='center', va='center')
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.canvas.draw()
    
    def load_and_process_image(self, file_path):
        try:
            self.status_bar.config(text=f"Loading image: {file_path}")
            
            # Load image with PIL for wider format support
            pil_img = Image.open(file_path)
            
            # Convert to numpy array for OpenCV
            img = np.array(pil_img)
            
            # Convert to RGB if needed
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Make image size multiple of 16
            tile = 16
            h, w = img.shape[:2]
            h, w = tile*(h//tile), tile*(w//tile)
            img = img[:h, :w]
            
            # Create degraded version for testing
            degraded_img = img

            # Store original and degraded images
            self.original_img = img
            self.degraded_img = degraded_img
            
            # Process with model
            self.process_image()
            
            # Update status
            self.status_bar.config(text=f"Processed image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_bar.config(text=f"Error loading image: {str(e)}")
    
    def process_image(self):
        if self.degraded_img is None:
            return
        
        try:
            # Convert to tensor and normalize
            to_tensor = transforms.ToTensor()
            degraded_tensor = to_tensor(self.degraded_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Process image through model
                enhanced_tensor = self.model(degraded_tensor)
                
                # Convert back to numpy array
                enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()
                enhanced_np = np.transpose(enhanced_np, (1, 2, 0))
                
                # Clip values to valid range
                enhanced_np = np.clip(enhanced_np, 0, 1)
                
                # Convert to uint8 for display
                enhanced_np = (enhanced_np * 255).astype(np.uint8)
                
                # Store enhanced image
                self.enhanced_img = enhanced_np
                
                # Display images
                self.display_images()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_bar.config(text=f"Error processing image: {str(e)}")
    
    def display_images(self):
        if self.degraded_img is None or self.enhanced_img is None:
            return
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Display images
        self.ax1.imshow(self.degraded_img)
        self.ax2.imshow(self.enhanced_img)
        
        # Set titles
        self.ax1.set_title('Original Image', fontsize=12)
        self.ax2.set_title('Enhanced Image', fontsize=12)
        
        # Turn off axis
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_scroll(self, event):
        # Zoom in/out on both plots
        if event.inaxes:
            cur_xlim = event.inaxes.get_xlim()
            cur_ylim = event.inaxes.get_ylim()
            
            xdata = event.xdata
            ydata = event.ydata
            
            # Scale factor: 1.1 = zoom in, 0.9 = zoom out
            base_scale = 1.1
            scale = 1/base_scale if event.button == 'up' else base_scale
            
            # Calculate new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale
            
            # Center around mouse position
            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            # Apply same zoom to both axes
            for ax in [self.ax1, self.ax2]:
                ax.set_xlim([
                    xdata - new_width * (1-relx),
                    xdata + new_width * relx
                ])
                ax.set_ylim([
                    ydata - new_height * (1-rely),
                    ydata + new_height * rely
                ])
            
            self.canvas.draw()
    
    def on_press(self, event):
        if event.inaxes:
            self.pressed = True
            self.x_prev = event.xdata
            self.y_prev = event.ydata
    
    def on_release(self, event):
        self.pressed = False
    
    def on_motion(self, event):
        if self.pressed and event.inaxes:
            dx = event.xdata - self.x_prev
            dy = event.ydata - self.y_prev
            
            # Apply pan to both axes
            for ax in [self.ax1, self.ax2]:
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                ax.set_xlim([x_lim[0] - dx, x_lim[1] - dx])
                ax.set_ylim([y_lim[0] - dy, y_lim[1] - dy])
            
            self.canvas.draw()
            
            self.x_prev = event.xdata
            self.y_prev = event.ydata
    
    def run(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Image Quality Enhancement Viewer")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to default test image')
    
    args = parser.parse_args()
    
    app = ImageEnhancerGUI(args.model, args.image)
    app.run()

if __name__ == "__main__":
    main()
