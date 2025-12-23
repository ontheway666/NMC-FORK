

# zxc temp test
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

def draw_stream(x,y,myu,myv,tempcnt,sampleRate=1):
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(y.shape==myu.shape)
    assert(myu.shape==myv.shape)
    # ----------------optional -----interpolate -----------------------------
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

    norm = Normalize(vmin=0.0, vmax=0.7)
    
    fig = plt.streamplot(
        Xg,Yg,u_grid,v_grid,
        density=2,
        color=np.sqrt(u_grid**2 + v_grid**2),
        cmap='viridis',
        norm=norm
    )

    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.axis('equal')

    # ---------------------interpolate -----------------------------





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
    plt.savefig("zxcstream-"+str(tempcnt)+".png", bbox_inches='tight',dpi=300)




res = int( 1000 )+2
for i in range(120,125):
    print(i)
  

    
    try:
        samples_v = np.loadtxt(r"C:\Users\123\Desktop\TEMP\velocity_samples_t{0:03}.txt".format(i))
        values_v = np.loadtxt(r"C:\Users\123\Desktop\TEMP\velocity_values_t{0:03}.txt".format(i))

    except:
        break
    
    print(values_v.shape)
    print(samples_v.shape)
    values_v= values_v.reshape((18,909,2))
    samples_v= samples_v.reshape((18,909,2))

    x = samples_v[:,:,0]
    y= samples_v[:,:,1]
    myu= values_v[:,:,0]
    myv=values_v[:,:,1]

    draw_stream(x,y,myu,myv,i)
    plt.close()

print("Done")