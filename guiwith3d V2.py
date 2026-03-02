import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
import os
import open3d as o3d

current_photo=None

def show_model():
    pcd = o3d.io.read_point_cloud(r"D:\CODE\CCONV RES\mix\static_Stick___csm300_1111example_still\static_Stick___csm300_1111example_still\fluid_0994.ply")
   
    print(pcd)
    # print(pcd.get_property_names())
    print("点数量:", len(pcd.points))
    
    # dot 
    # o3d.visualization.draw_geometries([pcd])


    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 20.0
    opt.background_color = [0, 0, 0]

    vis.run()
    vis.destroy_window()




def load_file():
    path = filedialog.askopenfilename()
    if path:
        file_label.config(text=os.path.basename(path))

def show_image(idx):

    global image_list
    if not image_list:
        print('no image list')
        return
    from PIL import Image, ImageTk
    print('try open' + str(image_list[idx]))
    img = Image.open(image_list[idx])
    img = img.resize((600, 400))  # 可自行调整大小

    global imgwin
    global current_photo

    current_photo = ImageTk.PhotoImage(img)
    imgwin.configure(image=current_photo)

globalvar=5
def on_scale_change(value):
    assert(globalvar==5)
    idx = int(float(value))
    id_label.config(text=f"当前帧 {idx}")
    print(type(image_list))
    print(len(image_list))

    show_image(idx)




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
        assert(False)
        return



    # show_image(0)
load_folder()
print(type(image_list))
print(len(image_list))



gapx = 7
gapy = 8
LEFT_WIDTH = 300

app = tb.Window(themename="sandstone")
app.geometry("900x600")

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
scrollbarX = tb.Scrollbar(left_container, orient=HORIZONTAL, command=canvas.xview)

scrollable_frame = tb.Frame(canvas)

# 让内部frame大小变化时自动更新scroll区域
def update_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", update_scrollregion)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.configure(xscrollcommand=scrollbarX.set)


canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)
scrollbarX.pack(side=BOTTOM, fill=X)


framerow=0


left0 = tb.Frame(scrollable_frame)
left0.grid(row=framerow, column=0,sticky="w")
left0.columnconfigure(0, weight=0)
left0.columnconfigure(1, weight=0)


framerow+=1


left1 = tb.Frame(scrollable_frame)
left1.grid(row=framerow, column=0,sticky="w")
left1.columnconfigure(0, weight=0)
left1.columnconfigure(1, weight=0)
framerow+=1

linearframe = tb.Frame(scrollable_frame)
linearframe.grid(row=framerow, column=0,sticky="w")
linearframe.columnconfigure(0, weight=0)
linearframe.columnconfigure(1, weight=0)
framerow+=1


gravframe = tb.Frame(scrollable_frame)
gravframe.grid(row=framerow, column=0,sticky="w")
gravframe.columnconfigure(0, weight=0)
gravframe.columnconfigure(1, weight=0)

framerow+=1

separator = tb.Separator(scrollable_frame, orient=HORIZONTAL)
separator.grid(row=framerow, column=0, columnspan=30, sticky="ew", pady=gapy)


framerow+=1

left2 = tb.Frame(scrollable_frame)
left2.grid(row=framerow, column=0,sticky="w")

framerow+=1

scalebarframe = tb.Frame(scrollable_frame)
scalebarframe.grid(row=framerow, column=0,sticky="w")

framerow+=1

separator = tb.Separator(scrollable_frame, orient=HORIZONTAL)
separator.grid(row=framerow, column=0, columnspan=30, sticky="ew", pady=gapy)



framerow+=1


cconvframe = tb.Frame(scrollable_frame)
cconvframe.grid(row=framerow, column=0,sticky="w")

framerow+=1

# ==================================================
# 左侧内部组件（用 grid 纵向排列）
# ==================================================
nowrow=0


btn=tb.Button(left0, text=f"模型插槽 1（.h5）", bootstyle=SUCCESS,command=load_file)
btn.grid(row=0, column=0,  padx=gapx, pady=gapy,sticky="w")
file_label = tb.Label(
                left0,
                text="未选择文件",
                bootstyle="secondary",
                width=40
            )
file_label.grid(row=0,column=1, padx=0, pady=gapy,sticky="w")

btn=tb.Button(left0, text=f"模型插槽 2（.h5）", bootstyle=SUCCESS,command=load_file)
btn.grid(row=1, column=0,  padx=gapx, pady=gapy,sticky="w")
file_label = tb.Label(
                left0,
                text="未选择文件",
                bootstyle="secondary",
                width=40
            )
file_label.grid(row=1,column=1, padx=0, pady=gapy,sticky="w")

btn=tb.Button(left0, text=f"模型插槽 3（.h5）", bootstyle=SUCCESS,command=load_file)
btn.grid(row=2, column=0,  padx=gapx, pady=gapy,sticky="w")
file_label = tb.Label(
                left0,
                text="未选择文件",
                bootstyle="secondary",
                width=40
            )
file_label.grid(row=2,column=1, padx=0, pady=gapy,sticky="w")

btn=tb.Button(left0, text=f"模型插槽 4（.h5）", bootstyle=SUCCESS,command=load_file)
btn.grid(row=3, column=0,  padx=gapx, pady=gapy,sticky="w")
file_label = tb.Label(
                left0,
                text="未选择文件",
                bootstyle="secondary",
                width=40
            )
file_label.grid(row=3,column=1, padx=0, pady=gapy,sticky="w")



# btn=tb.Button(left0, text=f"TEMP（.h5）", bootstyle=SUCCESS,command=load_file)
# btn.grid(row=0, column=1,  padx=gapx, pady=gapy, sticky="w")

# filelabelsarr[i].grid(row=i+2,column=1, padx=0, pady=gapy,sticky="w")


nowrow+=4

tex1 = tb.Label(
                left1,
                text="强度GAMMA",
                bootstyle="secondary",
                width=12
            )
tex1.grid(row=nowrow, column=0, padx=gapx, pady=gapy, sticky="w")
# btn = tb.Button(
#     left1,
#     text=f"重力",
#     bootstyle=PRIMARY,
#     width=5,
# )
# btn.grid(row=i, column=1, padx=gapx, pady=gapy, sticky="ew")
var = tb.IntVar(value=1)
entry = tb.Entry(left1, textvariable=var,width=10)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy,sticky="w")

nowrow+=1




check = tb.Checkbutton(
    linearframe,
    text="启用线性强度",
    variable=var,
    # command=toggle_entry,
    bootstyle="success-round-toggle"
)
check.grid(row=nowrow,column=0, padx=gapx, pady=gapy,sticky="w")





var = tb.IntVar(value=1)
entry = tb.Entry(linearframe, textvariable=var,width=5)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy,sticky="w")

var = tb.IntVar(value=1)
entry = tb.Entry(linearframe, textvariable=var,width=5)
entry.grid(row=nowrow,column=2,padx=gapx, pady=gapy,sticky="w")



tex1 = tb.Label(
               gravframe,
                text="重力",
                bootstyle="secondary",
                width=5
            )
tex1.grid(row=nowrow, column=0, padx=gapx, pady=gapy, sticky="w")

gravy=[0,-9.8,0]
gx = tb.IntVar(value=gravy[0])
entry = tb.Entry(gravframe, textvariable=gx,width=5)
entry.grid(row=nowrow,column=1,padx=gapx, pady=gapy,sticky="w")
gy = tb.IntVar(value=gravy[1])
entry = tb.Entry(gravframe, textvariable=gy,width=5)
entry.grid(row=nowrow,column=2,padx=gapx, pady=gapy,sticky="w")
gz = tb.IntVar(value=gravy[2])
entry = tb.Entry(gravframe, textvariable=gz,width=5)
entry.grid(row=nowrow,column=3,padx=gapx, pady=gapy,sticky="w")

btn = tb.Button(
    left2,
    text=f"涡度场",
    bootstyle=PRIMARY
)
btn.grid(row=nowrow, column=0, padx=gapx, pady=gapy, sticky="w")


btn = tb.Button(
   left2,
    text=f"速度场",
    bootstyle=PRIMARY
)
btn.grid(row=nowrow, column=1, padx=gapx, pady=gapy, sticky="w")

btn = tb.Button(
    left2,
    text=f"漩涡核心",
    bootstyle=PRIMARY
)
btn.grid(row=nowrow, column=2, padx=gapx, pady=gapy, sticky="w")



id_label = tb.Label(scalebarframe, text="ID: 0")
id_label.grid(row=nowrow,column=1,padx=gapx, pady=gapy)
print(type(image_list))
print(len(image_list))
print(272)

scale = tb.Scale(
        scalebarframe,
        from_=1,
        to=400,
        orient="horizontal",
        command=on_scale_change,

)
scale.grid(row=nowrow,column=0,padx=gapx, pady=gapy)





btn = tb.Button(
    cconvframe,
    text=f"启动模拟",
    bootstyle=PRIMARY,
    command=show_model
)
btn.grid(row=nowrow, column=0, padx=gapx, pady=gapy, sticky="ew")

btn = tb.Button(
    cconvframe,
    text=f"暂停模拟",
    bootstyle=PRIMARY,
    command=show_model
)
btn.grid(row=nowrow, column=1, padx=0, pady=gapy, sticky="w")

nowrow+=1

btn = tb.Button(
    cconvframe,
    text=f"点云信息(PLY)",
    bootstyle=PRIMARY,
    command=show_model
)
btn.grid(row=nowrow, column=0, padx=gapx, pady=gapy, sticky="w")

nowrow+=1

# ==================================================
# 右侧测试内容
# ==================================================
label = tb.Label(right_frame, text="右侧区域", font=("Arial", 30))
label.pack(expand=True)

imgwin = tb.Label(right_frame, text="结果图片", width=15)
imgwin.pack(expand=True)


app.mainloop()