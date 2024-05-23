import numpy as np
from ex_post_funs import *


from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.plotting import figure, show

N = 500

m_init = 40
kap_init = 0.5
rho_init = 1.5
the_init = 2.0
theb_init = 2.0
av_init = 1.25
bv_init = 1.25
a0_init = 0.5
b0_init = 0.1

a_vec_init = np.linspace(a0_init, a0_init + av_init, N)
b_vec_init = np.linspace(b0_init, b0_init + bv_init, N)

eps = 1e-10

x,y = find_group_br_a_vecb(m_init,the_init,kap_init,a0_init,av_init,b_vec_init,eps)
x[x==0] = np.NaN
y[y==0] = np.NaN
z,w = find_group_br_b_veca(m_init,theb_init,kap_init,rho_init,b0_init,bv_init,a_vec_init,eps)
z[z==0] = np.NaN
w[w==0] = np.NaN

# Set up plot
plot = figure(height=600, width=750, title="Best responses",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[a0_init-0.05,a0_init+av_init+0.05], y_range=[b0_init-0.05,b0_init+bv_init+0.05])

source_a = ColumnDataSource(data=dict(x=x,y=y))
source_b = ColumnDataSource(data=dict(x=z,y=w))

plot.scatter("x","y",source=source_a,size=1,color="#440145FF",legend_label="A-consistent strategies")
plot.scatter("x","y",source=source_b,size=1,color="#7AD151FF",legend_label="B-consistent strategies")

ab_data = np.linspace(max(a0_init,b0_init),min(a0_init+av_init,b0_init+bv_init),N)
source_ab = ColumnDataSource(data=dict(x=ab_data,y=ab_data))
plot.line("x","y",source=source_ab,line_width=2,line_color="black",legend_label="a=b",line_dash="dashed")

# Set up widgets

rho_slider = Slider(title=r"$$\rho$$", value=1.5, start=1.0, end=5.0, step=0.05)
kap_slider = Slider(title=r"$$\kappa$$", value=0.5, start=0.0, end=1.0, step=0.05)
m_slider = Slider(title=r"$$m$$", value=40.0, start=0.1, end=60.0,step=0.1)
the_a_slider = Slider(title=r"$$\theta_A$$", value=2.0, start=0.5, end=5.0, step=0.05)
the_b_slider = Slider(title=r"$$\theta_B$$", value=2.0, start=0.5, end=5.0, step=0.05)
a0_slider = Slider(title=r"$$a_0$$", value=0.5, start=0.05, end=1.0, step=0.05)
b0_slider = Slider(title=r"$$b_0$$", value=0.1, start=0.05, end=1.0, step=0.05)
av_slider = Slider(title=r"$$a_v$$", value=1.25, start=0.0, end=2.0, step=0.05)
bv_slider = Slider(title=r"$$b_v$$", value=1.25, start=0.0, end=2.0, step=0.05)
def update_data(attrname,old,new):
    rho = rho_slider.value
    kap = kap_slider.value
    m = m_slider.value
    the = the_a_slider.value
    theb = the_b_slider.value
    a0 = a0_slider.value
    b0 = b0_slider.value
    av = av_slider.value
    bv = bv_slider.value

    b_vec_init = np.linspace(b0, b0 + bv, N)
    a_vec_init = np.linspace(a0, a0 + av, N)

    x,y = find_group_br_a_vecb(m,the,kap,a0,av,b_vec_init,eps)
    x[x==0] = np.NaN
    y[y==0] = np.NaN
    z,w = find_group_br_b_veca(m,theb,kap,rho,b0,bv,a_vec_init,eps)
    z[z==0] = np.NaN
    w[w==0] = np.NaN

    ab_data = np.linspace(max(a0,b0),min(a0+av,b0+bv),N)
    source_ab.data=dict(x=ab_data,y=ab_data)

    source_a.data = dict(x=x,y=y)
    source_b.data = dict(x=z,y=w)
    
    plot.x_range.update(start=a0-0.05,end=a0+av+0.05)
    plot.y_range.update(start=b0-0.05,end=b0+bv+0.05)


for w in [rho_slider, kap_slider, m_slider, the_a_slider, the_b_slider, a0_slider, b0_slider, av_slider, bv_slider]:
    w.on_change('value', update_data)

description=Div(text=r"Uniform cost on $$[0,\theta_A]$$ and $$[0,\theta_B]$$, respectively. Updating the plot may take some time after moving the sliders. Frequently updating may require a high data volume.",width=300)


inputs= column(rho_slider,kap_slider,m_slider,the_a_slider,the_b_slider,a0_slider,b0_slider,av_slider,bv_slider,description)
plot.add_layout(plot.legend[0],"right")
curdoc().add_root(row(inputs,plot))
curdoc().title = "Ex Post Visualization"