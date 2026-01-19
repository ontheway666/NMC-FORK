

# zxc temp test
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
import os

VMIN = 0.0
VMAX = 0.7


def drawU(
    x, y, u, 
    index: int,
    txtpath,
    dpi=300,
    cmap="viridis",
    fname="u_component"
):
    """

    Parameters
    ----------
    x, y : 2D ndarray
        网格坐标
    u, v : 2D ndarray
        矢量场分量
    index : int
        文件编号
    txtpath : str
        输出目录
    dpi : int
        保存图片的 DPI
    cmap : str
        颜色映射
    """
    # os.makedirs(txtpath, exist_ok=True)

    # extent 用于保证坐标轴和 quiver 一致
    extent = [x.min(), x.max(), y.min(), y.max()]

    # ===== u 分量 =====
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        u,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
        vmin=VMIN,
        vmax=VMAX
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("u value")

    ax.set_title(fname + f"(index={index})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    print('[save u comp to dir]' + str(txtpath))

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(txtpath), fname+f"_{index:04d}.png"),
        dpi=dpi
    )
    print('[save u comp]')
    plt.close(fig)


def draw_stream(x,y,myu,myv,tempcnt,sampleRate=1):
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(y.shape==myu.shape)
    assert(myu.shape==myv.shape)

    # ----------------作者产生的数据理论上就是规则网格，这里再次处理一下
    # ------------------------optional  interpolate -----------------------------
    from scipy.interpolate import griddata
    points = np.column_stack([x.flatten(), y.flatten()])
    values_u = myu.flatten()
    values_v = myv.flatten()

    # 规则网格
    x1d = np.linspace(x.min(), x.max(), x.shape[0]*sampleRate)
    y1d = np.linspace(y.min(), y.max(), x.shape[1]*sampleRate)
    Xg, Yg = np.meshgrid(x1d, y1d)

    u_grid = griddata(points, values_u, (Xg, Yg),method='cubic')
    v_grid = griddata(points, values_v, (Xg, Yg),method='cubic')
    print('zxc after interpolate')
    print(Xg.shape)
    print(Yg.shape)
    print(u_grid.shape)
    print(v_grid.shape)

    norm = Normalize(vmin=VMIN, vmax=VMAX)
    
    fig = plt.streamplot(
        Xg,Yg,u_grid,v_grid,
        density=DENSITY,
        color=np.sqrt(u_grid**2 + v_grid**2),
        cmap='viridis',
        norm=norm
    )

    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.axis('equal')

    # ---------------------interpolate -------------------------------------------





    # x1d = x[0, :]      # (N,)
    # y1d = y[:, 0]      # (M,)

    # # 2. 保证 x 递增，并同步重排 u, v
    # if not np.all(np.diff(x1d) > 0):
    #     idx_x = np.argsort(x1d)
    #     x1d = x1d[idx_x]
    #     myu = myu[:, idx_x]
    #     myv = myv[:, idx_x]

    # # 3. 保证 y 递增，并同步重排 u, v
    # if not np.all(np.diff(y1d) > 0):
    #     idx_y = np.argsort(y1d)
    #     y1d = y1d[idx_y]
    #     myu = myu[idx_y, :]
    #     myv = myv[idx_y, :]

    # print('zxc x1d')
    # print(x1d[:10,...])
    # fig= plt.streamplot(
    #     # x.flatten(), 
    #     # y.flatten(),
    #     x1d,y1d, 
    #     myu,myv,
    #     # grid_values[..., 0].T,
    #     # grid_values[..., 1].T, 
    #     density=1.2,
    #     cmap='viridis'
    # )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity Streamplot')
    plt.colorbar(label='Speed')
    plt.tight_layout()
    plt.savefig(veldir+r"\zxcstream\zxc--"+str(tempcnt)+".png", bbox_inches='tight',dpi=300)


bDrawUv=1
bDrawStream=0
bTranslateXY=0

veldir = r"C:\Users\123\Desktop\TEMP\\"

NX = 81
NX = 62    #taylor, liddriven


DENSITY=2


# karman_4edge3obs
NX = 201
SAMPLE_RATE=2
DENSITY=3







for i in tqdm(range(120,125)):
    print(i)
      
    try:
        samples_v = np.loadtxt(veldir + r"velocity_samples_t{0:03}.txt".format(i))
        values_v =  np.loadtxt(veldir + r"velocity_values_t{0:03}.txt".format(i))

    except:
        print('error at '+str(i))
        continue
    
    print(values_v.shape)
    print(samples_v.shape)
    samplenum = values_v.shape[0]
    values_v = values_v.reshape ((NX, int(samplenum/NX), 2))
    samples_v= samples_v.reshape((NX, int(samplenum/NX), 2))

    x = samples_v[:,:,0]
    y = samples_v[:,:,1]
    myu = values_v[:,:,0]
    myv = values_v[:,:,1]


    

    if(bDrawStream):
        draw_stream(x,y,myu,myv,i, sampleRate=SAMPLE_RATE)
    if(bDrawUv):
        if(bTranslateXY):
            x = x + 1.5
            y = y + 0.7
        drawU(
            x=x,
            y=y,
            u=myu,
            index=i,
            txtpath=veldir + r"velocity_values_t{0:03}.txt".format(i),
        )
        drawU(
            x=x,
            y=y,
            u=myv,
            index=i,
            txtpath=veldir + r"velocity_values_t{0:03}.txt".format(i),
            fname="v_component"
        )
    plt.close()

print("Done")