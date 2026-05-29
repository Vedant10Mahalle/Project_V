import os, re
def remove_emojis(text):
    pattern = re.compile('[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff\u2702-\u27b0\u2600-\u26FF]+', flags=re.UNICODE)
    return pattern.sub('', text)

targets = ['app.py']
for root, dirs, files in os.walk('templates'):
    for f in files:
        if f.endswith('.html'):
            targets.append(os.path.join(root, f))

specifics = ['??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '?', '?', '??', '??', '?', '?', '??', '??', '??', '??', '??', '??', '?', '?', '??', '??']
for fp in targets:
    with open(fp, 'r', encoding='utf-8') as f:
        c = f.read()
    for s in specifics:
        c = c.replace(s, '')
    c = c.replace('\uFE0F', '')
    c = remove_emojis(c)
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(c)
print('Emojis stripped.')
