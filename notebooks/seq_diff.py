import re
from difflib import SequenceMatcher
from IPython.display import HTML, display

def default_mark(text):
    return text

def mark(text):
    return '<span style="background: #FF929B;">' + text + '</span>'

def wrap_text(text):
    return '<div style="width:100%; word-wrap: break-word; font-family: monospace">' + text + '</div>'

def diff_html(a, b, display_a = False, display_b = False):
    
    seqmatcher = SequenceMatcher(isjunk=None, a=a, b=b, autojunk=False)
    
    out_a, out_b = '', ''
    for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        markup = default_mark if tag == 'equal' else mark
        out_a += markup(a[a0:a1])
        out_b += markup(b[b0:b1])
        
    out_a = HTML(wrap_text(out_a))
    out_b = HTML(wrap_text(out_b))
    
    if display_a:
        display(out_a)

    if display_b:
        display(out_b)
    
    return out_a, out_b
