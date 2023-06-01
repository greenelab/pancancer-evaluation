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
    "925", "500",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_constant_search_parameter_vs_perf.svg')
    ).scale(0.65).move(20, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_diff_dist_constant_search.svg')
    ).scale(0.8).move(25, 255),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_best_vs_largest_constant_search.svg')
    ).scale(0.85).move(470, 255),
    Text("A", 15, 25, size=22, weight="bold", font="Arial"),
    Text("B", 425, 25, size=22, weight="bold", font="Arial"),
    Text("C", 15, 270, size=22, weight="bold", font="Arial"),
    Text("D", 435, 270, size=22, weight="bold", font="Arial"),
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
    "900", "465",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_learning_rate_schedule_compare.svg')
    ).scale(0.6).move(20, 10),
    Text("A", 10, 60, size=22, weight="bold", font="Arial"),
    Text("B", 395, 60, size=22, weight="bold", font="Arial"),
    Text("C", 10, 260, size=22, weight="bold", font="Arial"),
    Text("D", 395, 260, size=22, weight="bold", font="Arial"),
)
display(fig_2)


# In[7]:


f2_svg = str(os.path.join(paper_figures_dir, 'figure_2.svg'))
f2_png = str(os.path.join(paper_figures_dir, 'figure_2.png'))

fig_2.save(f2_svg)


# In[8]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={f2_png} {f2_svg} -d 200')


# In[9]:


fig_3 = Figure(
    "1000", "515",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_coef_count_dist.svg')
    ).scale(1.0).move(20, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_constant_search_coef_weights.svg')
    ).scale(0.65).move(515, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_constant_search_loss_curves.svg')
    ).scale(0.65).move(40, 250),
    Text("A", 10, 25, size=22, weight="bold", font="Arial"),
    Text("B", 490, 25, size=22, weight="bold", font="Arial"),
    Text("C", 10, 300, size=22, weight="bold", font="Arial"),
    Text("D", 465, 300, size=22, weight="bold", font="Arial"),
)
display(fig_3)


# In[10]:


f3_svg = str(os.path.join(paper_figures_dir, 'figure_3.svg'))
f3_png = str(os.path.join(paper_figures_dir, 'figure_3.png'))

fig_3.save(f3_svg)


# In[11]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={f3_png} {f3_svg} -d 200')


# ### Supplementary Figure 1

# In[12]:


supp_f1 = Figure(
    "1000", "570",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_diff_dist_adaptive.svg')
    ).scale(0.9).move(20, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_diff_dist_constant.svg')
    ).scale(0.9).move(495, 10),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'all_optimizers_diff_dist_optimal.svg')
    ).scale(0.9).move(20, 285),
)

display(supp_f1)


# In[13]:


supp_f1_svg = str(os.path.join(paper_figures_dir, 'supp_figure_1.svg'))
supp_f1_png = str(os.path.join(paper_figures_dir, 'supp_figure_1.png'))

supp_f1.save(supp_f1_svg)


# In[14]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f1_png} {supp_f1_svg} -d 200')


# In[15]:


supp_f2 = Figure(
    "1000", "510",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_learning_rate_schedule_losses.svg')
    ).scale(0.65).move(20, 10),
)

display(supp_f2)


# In[16]:


supp_f2_svg = str(os.path.join(paper_figures_dir, 'supp_figure_2.svg'))
supp_f2_png = str(os.path.join(paper_figures_dir, 'supp_figure_2.png'))

supp_f2.save(supp_f2_svg)


# In[17]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f2_png} {supp_f2_svg} -d 200')


# In[18]:


supp_f3 = Figure(
    "700", "520",
    etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
    SVG(
        os.path.join(cfg.repo_root, '01_stratified_classification', 'optimizers_plots', 'KRAS_learning_rate_schedule_coefs.svg')
    ).scale(1.5).move(20, 10),
)

display(supp_f3)


# In[19]:


supp_f3_svg = str(os.path.join(paper_figures_dir, 'supp_figure_3.svg'))
supp_f3_png = str(os.path.join(paper_figures_dir, 'supp_figure_3.png'))

supp_f3.save(supp_f3_svg)


# In[20]:


# use inkscape command line to save as PNG
get_ipython().system('inkscape --export-png={supp_f3_png} {supp_f3_svg} -d 200')

