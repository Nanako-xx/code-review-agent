import html


def render_name(value):
    return html.escape(value, quote=False)
