import numpy as np
from ex_post_funs_other_cost import *


from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.plotting import figure

def cost_function(cbar,ai,k,av,a0):
    return cbar*(ai-a0)**(k+1)/((k+1)*av**(k+1))

def benefit_function(x,m):
    return np.arctan(m*x)/np.arctan(m)

def cdf(x,cbar,k):
    return (x/cbar)**(1/k)

N = 300
N2 = 100

m_init = 20.0
kap_init = 0.5
cbar_init = 2.0
k_init = 1
av_init = 1.25
bv_init = 1.25
a0_init = 0.0
b0_init = 0.0

a_vec_init = np.linspace(av_init/N,av_init, N)
b_vec_init = np.linspace(bv_init/N,bv_init, N)

eps = 1e-10

x,y = find_group_br_a_vecb(m_init,cbar_init,kap_init,k_init,av_init,b_vec_init,a0_init,eps)
x[x==0] = np.NaN
y[y==0] = np.NaN
z,w = find_group_br_a_vecb(m_init,cbar_init,kap_init,k_init,bv_init,a_vec_init,b0_init,eps)
z[z==0] = np.NaN
w[w==0] = np.NaN

# Set up plot
plot = figure(height=600, width=750, title="Consistent strategies",
              tools="crosshair,pan,reset,save,wheel_zoom",x_axis_label='a',y_axis_label='b',
              x_range=[-0.05,av_init+0.05], y_range=[-0.05,bv_init+0.05])

cost_plot = figure(height=200, width=250, title="Cost function (of $$a^i$$)")
benefit_plot = figure(height=200, width=250, title=r"Benefit function (of $$\alpha$$)")
cdf_plot = figure(height=200, width=250, title=r"CDF (of $$x$$ in the support)")

cdf_x_init = np.linspace(0,cbar_init,N2)
benefit_x = np.linspace(-1,1,N2)

source_cost = ColumnDataSource(data=dict(x=a_vec_init,y=cost_function(cbar_init,a_vec_init,k_init,av_init,a0_init)))
source_benefit = ColumnDataSource(data=dict(x=benefit_x,y=benefit_function(benefit_x,m_init)))
source_cdf = ColumnDataSource(data=dict(x=cdf_x_init,y=cdf(cdf_x_init,cbar_init,k_init)))

source_a = ColumnDataSource(data=dict(x=x,y=y))
source_b = ColumnDataSource(data=dict(x=w,y=z))

plot.scatter("x","y",source=source_a,size=1,color="#440145FF",legend_label="A-consistent strategies")
plot.scatter("x","y",source=source_b,size=1,color="#7AD151FF",legend_label="B-constistent strategies")

cdf_plot.line("x","y",source=source_cdf,line_width=2,line_color="black")
benefit_plot.line("x","y",source=source_benefit,line_width=2,line_color="black")
cost_plot.line("x","y",source=source_cost,line_width=2,line_color="black")

ab_data = np.linspace(0,min(av_init,bv_init),N)
source_ab = ColumnDataSource(data=dict(x=ab_data,y=ab_data))
plot.line("x","y",source=source_ab,line_width=2,line_color="black",legend_label="a=b",line_dash="dashed")

# Set up widgets

kap_slider = Slider(title=r"$$\kappa$$", value=0.5, start=0.0, end=1.0, step=0.1)
m_slider = Slider(title=r"$$m$$", value=20.0, start=0.1, end=40.0,step=1.0)
cbar_slider = Slider(title=r"$$\bar{c}$$", value=2.0, start=0.5, end=5.0, step=0.25)
k_slider = Slider(title=r"$$k$$", value=1, start=1, end=12, step=1)
av_slider = Slider(title=r"$$a_v$$", value=1.25, start=0.05, end=2.05, step=0.1)
bv_slider = Slider(title=r"$$b_v$$", value=1.25, start=0.05, end=2.05, step=0.1)
a0_slider = Slider(title=r"$$a_0$$", value=0.0, start=0.0, end=1.0, step=0.1)
b0_slider = Slider(title=r"$$b_0$$", value=0.0, start=0.0, end=1.0, step=0.1)
def update_data(attrname,old,new):
    kap = kap_slider.value
    m = m_slider.value
    cbar = cbar_slider.value
    k = k_slider.value
    av = av_slider.value
    bv = bv_slider.value
    a0 = a0_slider.value
    b0 = b0_slider.value

    a_vec = np.linspace(a0+av/N,av+a0, N)
    b_vec = np.linspace(b0+bv/N,bv+b0, N)

    x,y = find_group_br_a_vecb(m,cbar,kap,k,av,b_vec,a0,eps)
    x[x==0] = np.NaN
    y[y==0] = np.NaN
    z,w = find_group_br_a_vecb(m,cbar,kap,k,bv,a_vec,b0,eps)
    z[z==0] = np.NaN
    w[w==0] = np.NaN

    ab_data = np.linspace(max(a0,b0),min(av+a0,bv+b0),N)
    source_ab.data=dict(x=ab_data,y=ab_data)

    source_a.data = dict(x=x,y=y)
    source_b.data = dict(x=w,y=z)

    cdf_x = np.linspace(0,cbar,N2)
    source_cost.data=dict(x=a_vec,y=cost_function(cbar,a_vec,k,av,a0))
    source_benefit.data = dict(x=benefit_x,y=benefit_function(benefit_x,m))
    source_cdf.data=dict(x=cdf_x,y=cdf(cdf_x,cbar,k))
    
    plot.x_range.update(start=a0-0.05,end=a0+av+0.05)
    plot.y_range.update(start=b0-0.05,end=b0+bv+0.05)


for w in [cbar_slider,k_slider, kap_slider, m_slider, av_slider, bv_slider, a0_slider, b0_slider]:
    w.on_change('value', update_data)


description=Div(text=r"Parametric family of cost functions with cdf: <br> $$F(x) = \left(\frac{x}{\bar{c}}\right)^{\frac{1}{k}}$$ <br> $$k=1$$ corresponds to uniformly distributed cost. Both groups face the same cost. Updating the plot may take some time after moving the sliders. Frequently updating may require a high data volume.",width=300)

inputs= column(cbar_slider,k_slider,kap_slider,m_slider,av_slider,bv_slider,a0_slider,b0_slider,description)
plot.add_layout(plot.legend[0],"right")
plots = row(column(plot),column(cost_plot,benefit_plot,cdf_plot))
curdoc().add_root(row(inputs,plots))
curdoc().title = "Ex Post Visualization, with more cost functions"