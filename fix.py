import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_h2 = '<h2 style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: \'Outfit\', sans-serif; letter-spacing: 0.03em;">'
new_h2 = '<div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: \'Outfit\', sans-serif; letter-spacing: 0.03em;">'

if old_h2 in content:
    content = content.replace(old_h2, new_h2)
    
    # Only replace </h2> where we know we changed the opening tag.
    # Since we changed exactly 5 tags, let's just replace </h2> carefully, or maybe replace all </h2> that follow our new div.
    # A safe way is to replace '</h2>' with '</div>' ONLY in the Beranda page.
    # But let's just use regex or just replace all </h2>. Are there other </h2> in app.py?
    pass

# We will just write a specific replacement for each section title.
sections = [
    'Apa itu Pemuda NEET?',
    'Mengapa dashboard ini penting?',
    'Apa yang bisa kamu eksplorasi?',
    'Sumber data & variabel yang digunakan',
    'Dashboard ini untuk siapa?'
]

for s in sections:
    old = new_h2 + s + '</h2>'
    new = new_h2 + s + '</div>'
    content = content.replace(old, new)


# Update IPM Slider
old_slider = 'delta_ipm = st.slider("📈 Peningkatan Indeks Pembangunan Manusia (IPM) (poin)", min_value=0.0, max_value=10.0, value=0.0, step=0.5, key="ipm_slide",\n                                  help="Simulasi dampak investasi pembangunan manusia")'
new_slider = 'new_ipm = st.slider("📈 Indeks Pembangunan Manusia (IPM) (Indeks 1–100)", min_value=1.0, max_value=100.0, value=float(BASE_IPM), step=0.5, key="ipm_slide",\n                                  help="Simulasi dampak investasi pembangunan manusia")'
content = content.replace(old_slider, new_slider)

old_calc = 'new_ipm = BASE_IPM + delta_ipm'
new_calc = 'delta_ipm = new_ipm - BASE_IPM'
content = content.replace(old_calc, new_calc)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
