 {% extends 'python.tpl'%}
## Comments magic statement
## See https://github.com/ipython/ipython/issues/3707/#issuecomment-414813848
## use with jupyter nbconvert --to python --template=simplepython.tpl NAME_OF_NOTEBOOK.ipynb
 {% block codecell %}
 {{  super().replace('get_ipython','#get_ipython') if "get_ipython" in super() else super() }}
 {% endblock codecell %}
