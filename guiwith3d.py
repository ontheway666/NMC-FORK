import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import open3d as o3d
from tkinter import filedialog
import os


import ttkbootstrap as tb
from ttkbootstrap.constants import *

gapx = 10
gapy = 8
LEFT_WIDTH = 500


app = ttk.Window(themename="sandstone")

app.geometry("900x600")
app.title("控制台")

# =============================
# 主容器
# =============================
main_frame = tb.Frame(app)
main_frame.pack(fill=BOTH, expand=True)

# =============================
# 左侧固定宽度容器
# =============================
left_container = tb.Frame(main_frame, width=LEFT_WIDTH)
left_container.pack(side=LEFT, fill=Y)

left_container.pack_propagate(False)  # 关键：禁止自动缩放

# =============================
# 分割线
# =============================
separator = tb.Separator(main_frame, orient=VERTICAL)
separator.pack(side=LEFT, fill=Y, padx=5)

# =============================
# 右侧自适应区域
# =============================
right_frame = tb.Frame(main_frame)
right_frame.pack(side=LEFT, fill=BOTH, expand=True)

# ==================================================
# 左侧可滚动结构（Canvas + 内部Frame）
# ==================================================

canvas = tb.Canvas(left_container, highlightthickness=0)
scrollbar = tb.Scrollbar(left_container, orient=VERTICAL, command=canvas.yview)

scrollable_frame = tb.Frame(canvas)

# 让内部frame大小变化时自动更新scroll区域
def update_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", update_scrollregion)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# ==================================================
# 左侧内部组件（用 grid 纵向排列）
# ==================================================
# for i in range(30):
#     btn = tb.Button(
#         scrollable_frame,
#         text=f"按钮 {i+1}",
#         bootstyle=PRIMARY
#     )
#     btn.grid(row=i, column=0, padx=gapx, pady=gapy, sticky="ew")



# scrollable_frame.columnconfigure(0, weight=0)
# scrollable_frame.columnconfigure(1, weight=0)

# ==================================================
# 右侧测试内容
# ==================================================
label = tb.Label(right_frame, text="右侧区域", font=("Arial", 30))
label.pack(expand=True)



# -----------------------------------

gapy=5
gapx=5

def show_model():
    pcd = o3d.io.read_point_cloud(r"D:\CODE\CCONV RES\mix\static_Stick___csm300_1111example_still\static_Stick___csm300_1111example_still\fluid_0994.ply")
   
    print(pcd)
    print("点数量:", len(pcd.points))
    o3d.visualization.draw_geometries([pcd])






# # 主容器（卡片效果）
card = ttk.Frame(app, padding=20)

file_frame = ttk.Frame(card)


# file_label.pack(side=LEFT, padx=5)

def load_file():
    path = filedialog.askopenfilename()
    if path:
        file_label.config(text=os.path.basename(path))
        # show_image(path)
# file_button = ttk.Button(
#         file_frame,
#         text="导入模型(.pt)",
#         bootstyle="info",
#         command=load_file
#     )
# file_button.pack(side=LEFT)


btn=ttk.Button(scrollable_frame, text=f"数据图", bootstyle=PRIMARY,command=show_model)\
    .grid(row=0, column=0, padx=gapx, pady=gapy)

filelabelsarr=[]
for i in range(0,4):
    file_label = ttk.Label(
                scrollable_frame,
                text="未选择文件",
                bootstyle="secondary",
                width=40
            )
    filelabelsarr.append(file_label)


# ttk.Button(app, text=f"zxc", bootstyle=PRIMARY)\
    # .grid(row=0, column=1, padx=gapx, pady=gapy)



separator = ttk.Separator(scrollable_frame, orient=HORIZONTAL)
separator.grid(row=1, column=0, columnspan=3, sticky="ew", pady=gapy)

# 第二排


for i in range(0,4):
    btn=ttk.Button(scrollable_frame, text=f"模型插槽"+str(i+1)+"（.h5）", bootstyle=SUCCESS,command=load_file)
    btn.grid(row=i+2, column=0,  padx=gapx, pady=gapy,sticky="w")

    btn=ttk.Button(scrollable_frame, text=f"TEMP"+str(i+1)+"（.h5）", bootstyle=SUCCESS,command=load_file)
    btn.grid(row=i+2, column=1,  padx=gapx, pady=gapy, sticky="w")

    # filelabelsarr[i].grid(row=i+2,column=1, padx=0, pady=gapy,sticky="w")





separator = ttk.Separator(scrollable_frame, orient=HORIZONTAL)
separator.grid(row=6, column=0, columnspan=3, sticky="ew", pady=gapy)

btn=ttk.Button(scrollable_frame, text=f"强度大小", bootstyle=PRIMARY,command=show_model)\
    .grid(row=7, column=0, padx=gapx, pady=gapy)

value = ttk.IntVar()

btn=ttk.Button(scrollable_frame, text=f"导入边界与固体信息（JSON）", bootstyle=PRIMARY,command=show_model)\
    .grid(row=8, column=0, padx=gapx, pady=gapy)

nowrow=9

# tex1 = ttk.Label(
#                 scrollable_frame,
#                 text="强度GAMMA",
#                 bootstyle="secondary",
#                 width=40
#             )
# tex1.grid(row=nowrow+1,column=0, padx=gapx, pady=gapy,sticky="w")


nowrow+=1


# 复选框变量
var = ttk.BooleanVar(value=False)
def toggle_entry():
    if var.get():
        linegam.config(state="normal")
    else:
        linegam.config(state="disabled")
# 勾选框
check = ttk.Checkbutton(
    scrollable_frame,
    text="启用线性强度",
    variable=var,
    command=toggle_entry,
    bootstyle="success-round-toggle"
)
check.grid(row=nowrow,column=0, padx=gapx, pady=gapy,sticky="w")


# 输入框（默认禁用）
linegam = ttk.Entry(scrollable_frame, state="disabled")
linegam.grid(row=nowrow,column=1, padx=gapx, pady=gapy,sticky="w")


nowrow+=1






tex1 = ttk.Label(
                scrollable_frame,
                text="强度GAMMA",
                bootstyle="secondary",
                width=40
            )
tex1.grid(row=nowrow,column=0, padx=gapx, pady=gapy,sticky="w")

# var = ttk.FloatVar(value=0.5)
entry = ttk.Entry(scrollable_frame, textvariable=var)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy)

nowrow+=1


tex1 = ttk.Label(
                scrollable_frame,
                text="迭代次数 (Iteration)",
                bootstyle="secondary",
                width=40
            )
tex1.grid(row=nowrow,column=0, padx=gapx, pady=gapy,sticky="w")

var = ttk.IntVar(value=50000)
entry = ttk.Entry(scrollable_frame, textvariable=var)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy)

nowrow+=1

tex1 = ttk.Label(
                scrollable_frame,
                text="重力大小(x/y/z)",
                bootstyle="secondary",
                width=15,
                
            )
tex1.grid(row=nowrow,column=0, padx=gapx, pady=gapy,sticky="w")

entry = ttk.Entry(scrollable_frame, textvariable=value)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy,sticky="w")

entry = ttk.Entry(scrollable_frame, textvariable=value)
entry.grid(row=nowrow,column=2,padx=gapx, pady=gapy)

entry = ttk.Entry(scrollable_frame, textvariable=value)
entry.grid(row=nowrow,column=3,padx=gapx, pady=gapy)


nowrow+=1

def on_scale_change(value):
    idx = int(float(value))
    # id_label.config(text=f"ID: {idx}")
    show_image(idx)
    pass

image_list=None
def load_folder():
    folder = filedialog.askdirectory()
    if not folder:
        return

    # 读取图片
    global image_list
    image_list = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_list:
        return



    show_image(0)

def show_image( idx):
        if not image_list:
            return
        from PIL import Image, ImageTk
        print('try open' + str(image_list[idx]))
        img = Image.open(image_list[idx])
        img = img.resize((600, 400))  # 可自行调整大小

        current_photo = ImageTk.PhotoImage(img)
        global image_label
        image_label.configure(image=current_photo)

scale = ttk.Scale(
        scrollable_frame,
        from_=1,
        to=400,
        orient="horizontal",
        command=on_scale_change
    )
scale.grid(row=nowrow,column=0,padx=gapx, pady=gapy)
nowrow+=1


image_label = ttk.Label(scrollable_frame)
image_label.grid(row=nowrow,column=0,padx=gapx, pady=gapy)
nowrow+=1

btn=ttk.Button(scrollable_frame, text=f"TEMP", bootstyle=SUCCESS,command=load_file)
btn.grid(row=nowrow, column=1,  padx=gapx, pady=gapy, sticky="w")

load_folder()

app.mainloop()