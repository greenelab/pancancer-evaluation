#!/usr/bin/env python
# coding: utf-8

# ## Generate paper figures
# 
# Uses the `svgutils.compose` API, [described here](https://svgutils.readthedocs.io/en/latest/tutorials/composing_multipanel_figures.html).

# In[1]:


import os
import shutil

from IPython.display import Image, display, SVG
from lxml import etree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from svgutils.compose import *

import pancancer_evaluation.config as cfg


# In[2]:


paper_figures_dir = os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'figures')


# ### Figure 1

# In[3]:


fig_1 = Figure(
    "925", "485",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_decile_vs_perf.svg')
    ).scale(0.65).move(30, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_diff_dist.svg')
    ).move(150, 260),
    Text("A", 15, 25, size=22, weight="bold", font="Arial"),
    Text("B", 450, 25, size=22, weight="bold", font="Arial"),
    Text("C", 140, 270, size=22, weight="bold", font="Arial"),
)
display(fig_1)


# In[4]:


os.makedirs(paper_figures_dir, exist_ok=True)

f1_svg = str(os.path.join(paper_figures_dir, 'figure_1.svg'))
f1_png = str(os.path.join(paper_figures_dir, 'figure_1.png'))

fig_1.save(f1_svg)


# In[5]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={f1_png} {f1_svg} -d 200')


# ### Figure 2

# In[6]:


fig_2 = Figure(
    "450", "450",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_coef_count_dist.svg')
    ).scale(0.9).move(20, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_coefficient_magnitudes.svg')
    ).scale(0.9).move(0, 230),
    Text("A", 10, 25, size=22, weight="bold", font="Arial"),
    Text("B", 10, 240, size=22, weight="bold", font="Arial"),
)
display(fig_2)


# In[7]:


f2_svg = str(os.path.join(paper_figures_dir, 'figure_2.svg'))
f2_png = str(os.path.join(paper_figures_dir, 'figure_2.png'))

fig_2.save(f2_svg)


# In[8]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={f2_png} {f2_svg} -d 200')


# ### Supplementary Figure 1

# In[9]:


supp_f1 = Figure(
    "1000", "335",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_parameter_to_decile.svg')
    ).scale(0.9).move(20, 10),
)

display(supp_f1)


# In[10]:


supp_f1_svg = str(os.path.join(paper_figures_dir, 'supp_figure_1.svg'))
supp_f1_png = str(os.path.join(paper_figures_dir, 'supp_figure_1.png'))

supp_f1.save(supp_f1_svg)


# In[11]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f1_png} {supp_f1_svg} -d 200')


# In[12]:


supp_f2 = Figure(
    "650", "320",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_coefs_dist.svg')
    ).scale(0.9).move(20, 10),
)

display(supp_f2)


# In[13]:


supp_f2_svg = str(os.path.join(paper_figures_dir, 'supp_figure_2.svg'))
supp_f2_png = str(os.path.join(paper_figures_dir, 'supp_figure_2.png'))

supp_f2.save(supp_f2_svg)


# In[14]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f2_png} {supp_f2_svg} -d 200')


# In[15]:


supp_f3 = Figure(
    "980", "280",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_parameter_vs_perf.svg')
    ).scale(0.7).move(20, 10),
)

display(supp_f3)


# In[16]:


supp_f3_svg = str(os.path.join(paper_figures_dir, 'supp_figure_3.svg'))
supp_f3_png = str(os.path.join(paper_figures_dir, 'supp_figure_3.png'))

supp_f3.save(supp_f3_svg)


# In[17]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f3_png} {supp_f3_svg} -d 200')

