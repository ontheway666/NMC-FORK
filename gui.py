import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time
import os

class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("现代UI示例")
        self.root.geometry("650x550")

        # 主容器（卡片效果）
        self.card = tb.Frame(root, padding=20)
        self.card.pack(fill=BOTH, expand=True, padx=30, pady=30)

        # =====================
        # 勾选框
        # =====================
        self.check_var = tb.BooleanVar()
        self.checkbox = tb.Checkbutton(
            self.card,
            text="启用功能",
            variable=self.check_var,
            bootstyle="success-round-toggle"
        )
        self.checkbox.pack(pady=10)

        # =====================
        # 文件选择
        # =====================
        file_frame = tb.Frame(self.card)
        file_frame.pack(pady=15, fill=X)

        self.file_label = tb.Label(
            file_frame,
            text="未选择文件",
            bootstyle="secondary",
            width=40
        )
        self.file_label.pack(side=LEFT, padx=5)

        self.file_button = tb.Button(
            file_frame,
            text="选择文件",
            bootstyle="info",
            command=self.load_file
        )
        self.file_button.pack(side=LEFT)

        # =====================
        # 进度条
        # =====================
        self.progress = tb.Progressbar(
            self.card,
            bootstyle="success-striped",
            length=450
        )
        self.progress.pack(pady=20)

        self.start_button = tb.Button(
            self.card,
            text="开始任务",
            bootstyle="primary",
            command=self.start_task
        )
        self.start_button.pack(pady=5)

        # =====================
        # 图片展示区域（带边框卡片）
        # =====================
        self.image_frame = tb.Frame(self.card, padding=10, bootstyle="light")
        self.image_frame.pack(pady=25, fill=BOTH, expand=True)

        self.image_label = tb.Label(
            self.image_frame,
            text="图片显示区域",
            anchor="center"
        )
        self.image_label.pack(expand=True)

        self.image_obj = None

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            self.file_label.config(text=os.path.basename(path))
            self.show_image(path)

    def show_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((450, 300))
            self.image_obj = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image_obj, text="")
        except:
            self.image_label.config(text="无法显示该文件")

    def start_task(self):
        threading.Thread(target=self.fake_task).start()

    def fake_task(self):
        for i in range(101):
            time.sleep(0.03)
            self.progress["value"] = i

if __name__ == "__main__":
    # 主题可改：cosmo, flatly, darkly, minty, morph 等
    root = tb.Window(themename="flatly")
    app = ModernApp(root)
    root.mainloop()