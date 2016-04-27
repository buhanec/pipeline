import sys
import logging
import IPython as ip


python_version = '.'.join(map(str, sys.version_info[0:3]))
ipython_version = '.'.join(map(str, ip.version_info[0:3]))
vstr = 'IPython {} / Python {}'.format(ipython_version, python_version)

#------------------------------------------------------------------------------
# InteractiveShellApp configuration
#------------------------------------------------------------------------------

c.InteractiveShellApp.matplotlib = 'inline'

#------------------------------------------------------------------------------
# InteractiveShellApp configuration
#------------------------------------------------------------------------------

c.InteractiveShellApp.exec_lines = [
    '%autoreload 2',
    '%config InlineBackend.figure_formats = [\'svg\', \'pdf\']',
    'from pprint import pprint as pp',
    'import scipy',
    'import numpy as np',
    'import pandas as pd',
    'import matplotlib as mpl',
    'import matplotlib.pyplot as plt',
    'import seaborn as sns',
    'plt.style.use(\'seaborn-paper\')',
    'mpl.rcParams[\'savefig.dpi\'] = 144',
    'mpl.rcParams[\'figure.figsize\'] = (12.0, 5.0)',
    'mpl.rcParams[\'lines.linewidth\'] = 0.8'
]
c.InteractiveShellApp.exec_files = [
    'isetup.py'
]
c.InteractiveShellApp.extensions = [
    'autoreload'
]

#------------------------------------------------------------------------------
# Application configuration
#------------------------------------------------------------------------------

c.Application.log_level = logging.ERROR

#------------------------------------------------------------------------------
# InteractiveShell configuration
#------------------------------------------------------------------------------

c.InteractiveShell.banner1 = vstr
c.InteractiveShell.colors = 'Linux'
c.InteractiveShell.confirm_exit = False
c.InteractiveShell.editor = 'nano'
c.InteractiveShell.xmode = 'Context'
c.InteractiveShell.autoindent = True

#------------------------------------------------------------------------------
# PromptManager configuration
#------------------------------------------------------------------------------

c.PromptManager.in_template  = '[\#]>> '
c.PromptManager.in2_template = '   >> '
c.PromptManager.out_template = '[\#]<< '
c.PromptManager.justify = True

#------------------------------------------------------------------------------
# Other configuration
#------------------------------------------------------------------------------

c.PrefilterManager.multi_line_specials = True

c.AliasManager.user_aliases = [
    ('cat', 'cat'),
    ('git', 'git'),
    ('mkdir', 'mkdir'),
    ('mv', 'mv'),
    ('l', 'ls -F -Al --color'),
    ('lx', 'ls -F -o --color %l | grep ^-..x'),
    ('ls', 'ls -F --color'),
    ('lf', 'ls -F -o --color %l | grep ^-'),
    ('cp', 'cp'),
    ('rmdir', 'rmdir'),
    ('ldir', 'ls -F -o --color %l | grep /$'),
    ('ll', 'ls -F -o --color'),
    ('rm', 'rm'),
    ('lk', 'ls -F -o --color %l | grep ^l')
]

c.InlineBackend.figure_formats = {'svg', 'pdf'}
c.InlineBackend.rc.update({
    'savefig.dpi': 144,
    'figure.figsize': (12.0, 5.0)
})
