import os
import tkinter as tk
import sys
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
import time
import threading
import string

from customtkinter import CTkImage
from sign_detector import SignDetectionFrame

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # where PyInstaller unpacks files
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class LandingPage(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure the window
        self.title("DeafCeption Demonstration App")
        self.geometry("900x500")
        self.minsize(800, 450)
        self.configure(bg="#242424")  # Set background color to match CTk frames

        # Try to set the app icon if available
        try:
            self.iconbitmap(resource_path('appdata/l0g0_dcg.ico'))
        except:
            print("Icon not found, using default")

        # Create a grid with 2 columns
        self.grid_columnconfigure(0, weight=1)  # Left side
        self.grid_columnconfigure(1, weight=1)  # Right side
        self.grid_rowconfigure(0, weight=1)

        # Create frames for left and right sides
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ctk.CTkFrame(self, corner_radius=0,fg_color='#8D6F3A',border_color='#FFCC70')
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure the right frame for widgets
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Setup slideshow in the left frame
        self.slideshow_label = ctk.CTkLabel(self.left_frame, text="")
        self.slideshow_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Add title and buttons to right frame - using custom font if available
        try:
            # Custom font rendering for landing label
            font_path = resource_path("appdata/SpaceGrotesk-Light.ttf")
            font_size = 36
            font = ImageFont.truetype(font_path, font_size)

            text = "Welcome to DeafCeption"
            image = Image.new("RGBA", (600, 60), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text(((600 - w) / 2, (60 - h) / 2), text, font=font, fill=(255, 255, 255, 255))
            self.text_img = ImageTk.PhotoImage(image)

            self.title_label = ctk.CTkLabel(
                self.right_frame,
                image=self.text_img,
                text=""
            )
        except Exception as e:
            print(f"Custom font rendering failed: {e}, using default")
            self.title_label = ctk.CTkLabel(
                self.right_frame,
                text="Welcome to DeafCeption",
                font=ctk.CTkFont(size=32, weight="bold")
            )

        self.title_label.grid(row=1, column=0, padx=20, pady=(40, 20))

        # Create a frame to hold the buttons
        self.button_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.button_frame.grid(row=2, column=0, padx=20, pady=20)
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        # Add buttons
        self.sign_to_text_btn = ctk.CTkButton(
            self.button_frame,
            text="Sign → Text",
            font=ctk.CTkFont(size=14),
            width=120,
            height=40,
            command=lambda: self.delayed_transition(self.open_sign_to_text)
        )
        self.sign_to_text_btn.grid(row=0, column=0, padx=20, pady=20)

        self.text_to_sign_btn = ctk.CTkButton(
            self.button_frame,
            text="Text → Sign",
            font=ctk.CTkFont(size=14),
            width=120,
            height=40,
            command=lambda: self.delayed_transition(self.open_text_to_sign)
        )
        self.text_to_sign_btn.grid(row=0, column=1, padx=20, pady=20)

        # Initialize slideshow
        self.image_paths = [resource_path("appdata/images/logo.png"), resource_path("appdata/images/me.png"),
                            resource_path("appdata/images/glasses.png"),resource_path("appdata/images/demo.png")]
        try:
            # Check if the image directory exists and has images
            image_dir = os.path.join("appdata", "images")
            if os.path.exists(image_dir):
                image_files = []
                for item in os.listdir(image_dir):
                    if item.endswith(('.png', '.jpg', '.jpeg')) and item[:-4].isdigit():
                        image_files.append(os.path.join(image_dir, item))

                if image_files:
                    self.image_paths = sorted(image_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        except Exception as e:
            print(f"Error loading image paths: {e}")

        self.current_image_index = 0
        self.images = []
        self.load_images()

        # Sign detection frame setup
        self.sign_to_text_frame = SignDetectionFrame(master=self)
        self.sign_to_text_frame.grid_forget()  # Hide initially

        # Text to sign frame setup
        self.setup_text_to_sign_frame()

        # Bind closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Database for Text->Sign
        self.database = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'hi', 'hello',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                         't', 'u', 'v', 'w', 'x', 'y', 'z', 'big', 'you', 'how']
        self.display_data = []
        self.current_index = -1  # Start at -1 since it gets incremented before first display

        self.wait_for_ready()

    def wait_for_ready(self):
        self.update_idletasks()  # ensure layout calculations are up to date
        frame_width = self.right_frame.winfo_width()
        frame_height = self.right_frame.winfo_height()

        if frame_width > 1 and frame_height > 1:
            print(True)
            self.run_slideshow()
        else:
            print(False)

    def setup_text_to_sign_frame(self):
        """Setup the Text to Sign translation frame"""
        self.text_to_sign_frame = ctk.CTkFrame(self)

        # Configure grid
        self.text_to_sign_frame.grid_columnconfigure(0, weight=1)
        self.text_to_sign_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        # Title
        self.text_to_sign_title = ctk.CTkLabel(
            self.text_to_sign_frame,
            text="Text to Sign Translation",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.text_to_sign_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Input Entry
        self.entry = ctk.CTkEntry(
            self.text_to_sign_frame,
            placeholder_text="Enter Text to Translate into Sign Language",
            width=500
        )
        self.entry.grid(row=1, column=0, padx=20, pady=10)

        # Translate Button
        self.translate_btn = ctk.CTkButton(
            self.text_to_sign_frame,
            text="Translate",
            command=self.handle_translate,
            width=120,
            height=40
        )
        self.translate_btn.grid(row=2, column=0, padx=20, pady=10)

        # Display label for current word
        self.display_label = ctk.CTkLabel(
            self.text_to_sign_frame,
            text="",
            font=ctk.CTkFont(size=20)
        )
        self.display_label.grid(row=3, column=0, padx=20, pady=10)

        # Image display label
        self.image_label = ctk.CTkLabel(self.text_to_sign_frame, text="")
        self.image_label.grid(row=4, column=0, padx=20, pady=10)

    def load_images(self):
        """Load images for slideshow"""
        try:
            for img_path in self.image_paths:
                img = Image.open(img_path)
                self.images.append(img)

            if not self.images:
                print("No images found")
                # Create a blank image with text as a placeholder
                img = Image.new('RGB', (400, 300), color=(240, 240, 240))
                self.images.append(img)
        except Exception as e:
            print(f"Error loading images: {e}")
            # Create a blank image as a placeholder
            img = Image.new('RGB', (400, 300), color=(240, 240, 240))
            self.images.append(img)

    def run_slideshow(self):
        if self.current_image_index < len(self.images):
            img = self.images[self.current_image_index]
            # Resize image as needed before display
            frame_width = self.left_frame.winfo_width()
            frame_height = self.left_frame.winfo_height()
            resized = self.resize_image(img, frame_width - 20, frame_height - 20)
            self.update_image(resized)

            self.current_image_index += 1
        else:
            self.current_image_index = 0

        self.after(3000, self.run_slideshow)  # call again after 3 seconds

    def update_image(self, img):
        """Convert image and update label safely in main thread"""
        photo = ImageTk.PhotoImage(img)
        self.slideshow_label.configure(image=photo)
        self.slideshow_label.image = photo

    def resize_image(self, img, width, height):
        """Resize image while maintaining aspect ratio, ensuring positive dimensions."""
        original_width, original_height = img.size

        # Calculate aspect ratios
        aspect_frame = width / height
        aspect_img = original_width / original_height

        if aspect_img > aspect_frame:
            # Image is wider than frame (relative to heights)
            new_width = max(100, int(width))
            new_height = max(100, int(width / aspect_img))
        else:
            # Image is taller than frame (relative to widths)
            new_width = max(100, int(height * aspect_img))
            new_height = max(100, int(height))

        return img.resize((new_width, new_height), Image.LANCZOS)

    def delayed_transition(self, func):
        """Add a small delay with loading indicator before transition"""
        loading = ctk.CTkLabel(self, text="Loading...", font=ctk.CTkFont(size=18))
        loading.place(relx=0.5, rely=0.5, anchor="center")
        self.after(300, lambda: (loading.destroy(), func()))

    def open_sign_to_text(self):
        """Open Sign to Text window"""
        print("Opening Sign to Text window")
        # Hide main layout
        self.left_frame.grid_forget()
        self.right_frame.grid_forget()

        # Configure for sign detection
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # Row for back button

        # Show sign detection frame
        self.sign_to_text_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sign_to_text_frame.start_detection()

        # Add back button below the sign detection frame
        self.back_button_sign = ctk.CTkButton(
            self,
            text="Back to Main",
            font=ctk.CTkFont(size=14),
            width=120,
            height=40,
            command=self.go_back
        )
        self.back_button_sign.grid(row=1, column=0, pady=(0, 10))

        # Resize window for sign detection
        self.geometry("800x700")

    def open_text_to_sign(self):
        """Open Text to Sign window"""
        print("Opening Text to Sign window")
        # Hide main layout
        self.left_frame.grid_forget()
        self.right_frame.grid_forget()

        # Configure for text to sign
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Show text to sign frame
        self.text_to_sign_frame.grid(row=0, column=0, sticky="nsew")

        # Reset window size
        self.geometry("900x600")

        # Add back button below the sign detection frame
        self.back_button_sign = ctk.CTkButton(
            self,
            text="Back to Main",
            font=ctk.CTkFont(size=14),
            width=120,
            height=40,
            command=self.go_back
        )
        self.back_button_sign.grid(row=1, column=0, pady=(0, 10))

    def go_back(self):
        """Return to the main landing page"""
        print("Returning to main page")

        # Stop sign detection if active
        try:
            self.sign_to_text_frame.stop_detection()
        except:
            pass

        # Hide all app frames
        self.sign_to_text_frame.grid_forget()
        self.text_to_sign_frame.grid_forget()

        # Hide back buttons if visible
        try:
            self.back_button_sign.grid_forget()
        except:
            pass

        try:
            self.back_button_text.grid_forget()
        except:
            pass

        # Reset grid configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Show main layout
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Reset window size
        self.geometry("900x500")

    def on_closing(self):
        """Handle window closing"""
        # Stop sign detection if active
        try:
            self.sign_to_text_frame.stop_detection()
        except:
            pass

        # Hide back buttons if visible
        try:
            self.back_button_sign.grid_forget()
        except:
            pass

        try:
            self.back_button_text.grid_forget()
        except:
            pass

        self.slideshow_active = False
        self.destroy()

    # === Text to Sign Translation Functionality ===

    def remove_custom_stopwords(self, sentence, stopwords_list):
        """Remove stopwords from sentence"""
        words = sentence.split()
        return ' '.join([word for word in words if word.lower() not in stopwords_list])

    def remove_special_characters(self, input_string):
        """Remove special characters from string"""
        allowed_chars = string.ascii_letters + string.digits + " "
        return ''.join([char for char in input_string if char in allowed_chars])

    def sentence_to_list(self, sentence):
        """Convert sentence to list of words"""
        return sentence.split()

    def load_image(self, path):
        """Load image for sign display"""
        try:
            img = Image.open(path)
            img = ctk.CTkImage(light_image=img, size=(576, 324))
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def update_display(self):
        """Update display for text to sign translation"""
        if self.current_index < len(self.display_data):
            word, img_path = self.display_data[self.current_index]
            self.display_label.configure(text=word)
            img_path = resource_path("appdata/signsDB/" + img_path)
            img = self.load_image(img_path)
            if img:
                self.image_label.configure(image=img, text="")
                self.image_label.image = img  # keep reference
            else:
                self.image_label.configure(image=None, text=f"[Missing: {img_path}]")
            self.current_index += 1
            self.after(1000, self.update_display)

    def handle_translate(self):
        """Handle translation of text to sign"""
        self.current_index = 0
        sentence = self.entry.get()

        # Empty check
        if not sentence.strip():
            self.display_label.configure(text="Please enter some text to translate")
            self.image_label.configure(image=None, text="")
            return

        custom_stopwords = ["is", "a", "for", "the", "to", "are"]
        cleaned = self.remove_special_characters(self.remove_custom_stopwords(sentence, custom_stopwords))
        words = self.sentence_to_list(cleaned)

        self.display_data = []

        for word in words:
            word = word.lower()
            if word in self.database:
                path = f"{word}.jpg"
                self.display_data.append((word, path))
            else:
                for char in word:
                    if char.lower() in string.ascii_lowercase:
                        path = f"{char}.jpg"
                        self.display_data.append((char, path))

        if self.display_data:
            self.update_display()
        else:
            self.display_label.configure(text="No translatable content found")
            self.image_label.configure(image=None, text="")


if __name__ == "__main__":
    app = LandingPage()
    app.mainloop()