# -*- coding: utf-8 -*-
"""
Decode HTML numeric character references (&#913; etc.) to actual UTF-8 Greek/text
so the HTML file is readable and editable in the editor.
"""
import re
import sys

def decode_entities(content):
    def replace_decimal(m):
        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return m.group(0)

    def replace_hex(m):
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    # Match &amp;#123; (in attributes) or &#123; (in text)
    content = re.sub(r'(?:&amp;#|&#)(\d+);', replace_decimal, content)
    content = re.sub(r'(?:&amp;#x|&#x)([0-9a-fA-F]+);', replace_hex, content)
    return content

def main():
    import os
    # Path relative to this script so it works in Docker and locally
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'project_presentation.html')
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    decoded = decode_entities(data)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(decoded)
    print("Done. Greek entities decoded to UTF-8 characters. File is now readable.")

if __name__ == '__main__':
    main()
