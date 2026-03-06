import ttkbootstrap as tb
import pygubu

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
import os
import open3d as o3d

from PIL import Image, ImageTk





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


globalvar=5

class App:

    
    def show_model(self,val):
        plydir=r"D:\CODE\CCONV RES\mix\static_Stick___csm300_1111example_still\static_Stick___csm300_1111example_still\fluid_0994.ply"

        # pcd = o3d.io.read_point_cloud(plydir)
        # print("点数量:", len(pcd.points))


        pcd=o3d.t.io.read_point_cloud(plydir)
        print(pcd)
        print(pcd.point)
        curlabs=pcd.point["curlabs"]
        values = curlabs.numpy().reshape(-1)
        vmin = values.min()
        vmax = values.max()
        norm = (values - vmin) / (vmax - vmin + 1e-8)
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis")
        colors = cmap(norm)[:, :3]   # RGB
        pcd.point["colors"] = o3d.core.Tensor(
            colors,
            dtype=o3d.core.Dtype.Float32,
            # device=o3d.core.Device("CUDA:0")
            device=o3d.core.Device("CPU:0")

        )
        # pcd = pcd.to(o3d.core.Device("CUDA:0"))
        pcd = pcd.to(o3d.core.Device("CPU:0"))

        o3d.visualization.draw([
            pcd
        ])

        

        # radius = 0.01
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius)

        # meshes = []
        # cnt=0
        # for p in pcd.points:
        #     cnt+=1
        #     m = o3d.geometry.TriangleMesh(sphere)
        #     m.translate(p)
        #     meshes.append(m)
        #     if(cnt==100):
        #         break

        # mesh = meshes[0]
        # for m in meshes[1:]:
        #     mesh += m

        # o3d.visualization.draw_geometries([mesh])


        
        # dot 
        # o3d.visualization.draw_geometries([pcd])


        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(pcd)

        opt = vis.get_render_option()

        # abandon
        opt.point_size = 20.0
        opt.background_color = [0, 0, 0]

        vis.run()
        vis.destroy_window()



    def __init__(self, master):

        self.builder = pygubu.Builder()
        self.builder.add_from_file("guiV3.ui")

        self.mainwindow = self.builder.get_object("frml", master)

        self.builder.connect_callbacks(self)


        self.tree = self.builder.get_object("treeview1")
        # self.tree.heading("name", text="场景名")
        # self.tree.heading("age", text="bounding box")
        # self.tree.heading("age", text="流体size")
        # self.tree.heading("age", text="流体center")
        # self.tree.heading("age", text="固体1(obj)引用")
        # self.tree.heading("age", text="固体2(obj)引用")
        data = [
            ["apple", 10],
            ["banana", 20],
            ["orange", 30]
        ]
        for row in data:
            self.tree.insert("", "end", values=row)


    def on_scale_change(self,value):
        assert(globalvar==5)
        idx = int(float(value))
        id_label=self.builder.get_object("id_label")

        id_label.config(text=f"当前帧 {idx}")
        print(type(image_list))
        print(len(image_list))

        self.show_image(idx)


    def show_image(self,idx):

        global image_list
        if not image_list:
            print('no image list')
            return
        from PIL import Image, ImageTk
        print('try open' + str(image_list[idx]))
        img = Image.open(image_list[idx])
        img = img.resize((600, 400))  # 可自行调整大小


        global current_photo
        imgwin=self.builder.get_object("imgwin")
        current_photo = ImageTk.PhotoImage(img)
        imgwin.configure(image=current_photo)


app = tb.Window(themename="pulse")

App(app)





app.mainloop()