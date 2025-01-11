import os
import pandas as pd
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Paths
index_csv = "described/index.csv"
described_dir = "described"

# Load CSV
data = pd.read_csv(index_csv)

# Text matching function
def find_top_matches(input_text, descriptions, top_n=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_text] + list(descriptions))
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    return ranked_indices[:top_n], cosine_similarities

# GUI Application
class ImageSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Search")
        self.geometry("800x680")
        self.configure(bg="white")
        
        # Input Frame
        input_frame = tk.Frame(self, bg="white")
        input_frame.pack(pady=10)
        tk.Label(input_frame, text="Enter a description to search:", font=("Arial", 14), bg="white").pack(side=tk.LEFT, padx=5)
        self.input_entry = tk.Entry(input_frame, font=("Arial", 14), width=40)
        self.input_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="Search", command=self.search_images, font=("Arial", 14), bg="#007ACC", fg="white").pack(side=tk.LEFT, padx=5)

        # Results Frame
        self.results_frame = tk.Frame(self, bg="white")
        self.results_frame.pack(pady=20)

    def search_images(self):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Get user input and find matches
        user_input = self.input_entry.get()
        if not user_input:
            tk.Label(self.results_frame, text="Please enter a description.", font=("Arial", 12), fg="red", bg="white").pack()
            return

        top_n = 3
        ranked_indices, similarities = find_top_matches(user_input, data["description"], top_n)

        # Display matches
        for i, idx in enumerate(ranked_indices):
            similarity_score = similarities[idx]
            matched_file = data.iloc[idx]["filename"]
            matched_description = data.iloc[idx]["description"]

            # Create a result card
            card = tk.Frame(self.results_frame, bg="#F0F0F0", pady=10, padx=10, relief="groove", bd=2)
            card.pack(pady=10, fill=tk.X, padx=10)

            # Load and display image
            image_path = os.path.join(described_dir, matched_file)
            image = Image.open(image_path).resize((150, 150))
            photo = ImageTk.PhotoImage(image)
            tk.Label(card, image=photo, bg="#F0F0F0").image = photo  # Keep a reference to avoid garbage collection
            tk.Label(card, image=photo, bg="#F0F0F0").pack(side=tk.LEFT, padx=10)

            # Display description and similarity score 
            details = tk.Frame(card, bg="#F0F0F0")
            details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(details, text=f"Description: {matched_description}", font=("Arial", 12), bg="#F0F0F0", wraplength=500, justify="left").pack(anchor="w", pady=5)
            tk.Label(details, text=f"Similarity Score: {similarity_score:.2f}", font=("Arial", 12, "italic"), fg="#555555", bg="#F0F0F0").pack(anchor="w", pady=5)

            # Add a select button
            tk.Button(card, text="Select", command=lambda file=matched_file: self.select_image(file), font=("Arial", 12), bg="#007ACC", fg="white").pack(side=tk.RIGHT, padx=10)

    def select_image(self, filename):
        # Action on image selection
        messagebox.showinfo("Image Selected", f"You have selected: {filename}")
        self.destroy()  # Close the application

# Run the application
if __name__ == "__main__":
    app = ImageSearchApp()
    app.mainloop()