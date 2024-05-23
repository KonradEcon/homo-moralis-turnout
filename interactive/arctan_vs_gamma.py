import numpy as np


from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.plotting import figure

N = 100

m_init = 1.0
gamma_init = 1.5

x = np.linspace(-1,1,N)

def arctan_benefit(x,m):
    return np.arctan(m*x)/np.arctan(m)

def gamma_benefit(x,gamma):
    return ((1+x)**gamma - (1-x)**gamma) / ((1+x)**gamma + (1-x)**gamma)

# Set up plot
plot = figure(height=600, width=750, title="Benefit functions",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[-1,1], y_range=[-1,1])

source_arctan = ColumnDataSource(data=dict(x=x,y=arctan_benefit(x,m_init)))
source_gamma = ColumnDataSource(data=dict(x=x,y=gamma_benefit(x,gamma_init)))

plot.line("x","y",source=source_arctan,color="#440145FF",legend_label="h₁")
plot.line("x","y",source=source_gamma,color="#7AD151FF",legend_label="h₂")

# Set up widgets

gamma_slider = Slider(title=r"$$\gamma$$", value=1.5, start=1.0, end=40.0, step=0.05)
m_slider = Slider(title=r"$$m$$", value=1.0, start=0.1, end=100.0,step=0.05)

def update_data(attrname,old,new):
    m = m_slider.value
    gamma = gamma_slider.value

    source_arctan.data = dict(x=x,y=arctan_benefit(x,m))
    source_gamma.data = dict(x=x,y=gamma_benefit(x,gamma))


for w in [gamma_slider, m_slider]:
    w.on_change('value', update_data)


description=Div(text=r"The benefit functions are defined as follows: <br> $$h_1(x,m) = \frac{1}{\arctan(m)} \arctan(mx)$$ <br> $$h_2(x,\gamma) = \frac{(1+x)^\gamma - (1-x)^\gamma}{(1+x)^\gamma + (1-x)^\gamma}$$")

inputs= column(gamma_slider,m_slider,description)
plot.add_layout(plot.legend[0],"right")
curdoc().add_root(row(inputs,plot, width=800))
curdoc().title = "Benefit functions"