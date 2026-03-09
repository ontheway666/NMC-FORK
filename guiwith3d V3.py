import pygubu

import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
import os
import open3d as o3d

from PIL import Image, ImageTk





def load_folder():
    # folder = filedialog.askdirectory()
    folder = r"D:\ZS\BS\temp karman"
    if not folder:
        return
    
    foldervor=os.path.join(folder, "vorticity")
    # print(foldervor)
    # assert(False)


    foldervel=os.path.join(folder, "velocity")
    # folderstr=os.path.join(folder, "streamline")
    folderstr=os.path.join(folder, "pressure")
    folderqc=os.path.join(folder, "qc")



    # 读取图片
    global image_list
    global image_listvel
    global image_listqc

    image_list = [
        os.path.join(foldervor, f)
        for f in sorted(os.listdir(foldervor))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_listvel=[
        os.path.join(foldervel, f)
        for f in sorted(os.listdir(foldervel))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_listqc=[
        os.path.join(foldervel, f)
        for f in sorted(os.listdir(folderqc))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_liststr=[
        os.path.join(foldervel, f)
        for f in sorted(os.listdir(folderstr))
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
    def write_log(self,msg):

        iterval=str(1)
        info="————————————\n"+\
            "[iter]\t"+iterval+"\t"+\
        "[loss]\t"+iterval+"\n"+\
        "[ene]\t"+iterval+"\t"+\
        "[alpha]\t"+iterval+"\n"+\
        "[part]\t"+iterval+"\n"
			  

        self.logtext.config(state="normal")
        self.logtext.insert(tk.END, info)
        self.logtext.see(tk.END)  # 自动滚动到底部
        self.logtext.config(state="disabled")

    
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


        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        # vis.add_geometry(pcd)

        # opt = vis.get_render_option()

        # # abandon
        # opt.point_size = 20.0
        # opt.background_color = [0, 0, 0]

        # vis.run()
        # vis.destroy_window()

    def _traverse_children(self, widget):

        children = widget.winfo_children()  
        
        for child in children:
            print(child)
            if isinstance(child, tb.Button):
            
                 child.config(bootstyle="outline button info") 
            self._traverse_children(child)

    def var_updated(self, *args):
        print("var update")
    def __init__(self, master):

        self.grav=tk.DoubleVar(value=9.8)
        self.grav.trace_add("write", self.var_updated)

       


        self.builder = pygubu.Builder()
        self.builder.add_from_file("guiV31.ui")
        

        self.mainwindow = self.builder.get_object("frame1", master)  
        self.logtext=self.builder.get_object("logtext", master)

        self.imgwin=self.builder.get_object("imgwin")
        self.imgvor=self.builder.get_object("imgvor")
        self.imgstr=self.builder.get_object("imgqc")
        self.imgqc=self.builder.get_object("imgstr")


        self.imgstat1=self.builder.get_object("imgstat1")
        self.imgstat2=self.builder.get_object("imgstat2")


        # 创建主窗口
        # root = self.builder.get_object('root')  # 获取根窗口

        # 获取所有控件对象
        # self.all_widgets = self.mainwindow.winfo_children()


        # for widget in self.all_widgets:
        #     pass
        #     # 检查控件是否是 ttk.Button 类型
        #     # if isinstance(widget, tb.Button):
        #         # pass
        #         # widget.config(bootstyle="success")  # 设置默认样式


        self._traverse_children(self.mainwindow)

        self.startbtn=self.builder.get_object("start")
        self.startbtn.config(bootstyle='outline button success') 
        self.stopbtn=self.builder.get_object("stop")
        self.stopbtn.config(bootstyle='outline button secondary')




        self.builder.connect_callbacks(self)


        (self.builder.tkvariables['grav'].set(9.8))
        (self.builder.tkvariables['gamma'].set(0.5))
        (self.builder.tkvariables['kernelsize'].set(4))
        (self.builder.tkvariables['layerstr'].set("[3,64,64,3]"))
        (self.builder.tkvariables['modelname'].set("cconv Default"))
        (self.builder.tkvariables['channel'].set(64))



        self.tree = self.builder.get_object("treeview1")
        # self.tree.heading("name", text="场景名")
        # self.tree.heading("age", text="bounding box")
        # self.tree.heading("age", text="流体size")
        # self.tree.heading("age", text="流体center")
        # self.tree.heading("age", text="固体1(obj)引用")
        # self.tree.heading("age", text="固体2(obj)引用")
        data = [
            ["wave", "有",  "box.obj",   "fluid.obj",  "有"],
            ["rotating panel", "有",  "box.obj",   "fluid.obj",  "有"],
            ["vessel", "有",  "box.obj",   "fluid.obj",  "有"],
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

    def cutImg(self,img,ratio=0.4):
        w, h = img.size

      

        new_w = int(w * 1)
        new_h = int(h * ratio)

        # 中心裁剪坐标
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h

        img = img.crop((left, top, right, bottom))
        return img

    def show_image(self,idx):

        global image_list
        if not image_list:
            print('no image list')
            return
        from PIL import Image, ImageTk
        print('try open' + str(image_list[idx]))
        print(self.builder.tkvariables['grav'].get())
        img =         Image.open(image_list[idx])
        imgvel=     Image.open(image_listvel[idx])
        # imgqc=      Image.open(image_listqc[idx])
        img_ene=Image.open(r"d:\ZS\BS\ene.png")
        

        # cut 
        img=self.cutImg(img)
        imgvel=self.cutImg(imgvel)


        imgvel=imgvel       .resize((int(60*5),   int(30*5)    ))
        img = img       .resize((int(60*5),   int(30*5)    ))


        img_ene = img_ene.resize((int(60*5),   int(40*5)    ))

        global current_photo
        global current_photoene
        global current_photovel
        current_photo = ImageTk.PhotoImage(img)
        current_photoene=ImageTk.PhotoImage(img_ene)
        current_photovel=ImageTk.PhotoImage(imgvel)
        # current_photoqc=ImageTk.PhotoImage(imgqc)

        self.imgwin.configure(image=current_photovel)
        self.imgvor.configure(image=current_photo)
        # self.imgqc.configure(image=current_photoqc)
        self.imgstr.configure(image=current_photo)


        # self.imgstat1.configure(image=current_photoene)
        # self.imgstat2.configure(image=current_photo)


# app=tk.PanedWindow(themename="clam")
app = tb.Window(themename="morph")
# app = tb.Window(themename="united")
# app = tb.Window(themename="litera")
# app = tb.Window(themename="cerculean")
# app = tb.Window(themename="yeti")


# print(tb.Style().theme_names())
# assert(False)




# app = tb.Window(themename="clam")

# app = tb.Window(themename="lumen")





App(app)

def var_updated():
    print('updated global')
grav=tk.DoubleVar(value=98)
grav.trace_add("write", var_updated)




app.mainloop()