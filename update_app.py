import re
import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Opacity KOTAK PUTIH di navigation bar
content = content.replace('background: rgba(255, 255, 255, 0.5) !important;', 'background: rgba(255, 255, 255, 0.15) !important;')

# 2. Page title font
content = content.replace("font-family: 'Outfit', sans-serif;", "font-family: 'Inter', sans-serif;")

# 3. Card titles (lebih mencolok)
content = content.replace(
    ".card-title {\n    font-size: 13px;\n    font-weight: 700;\n    color: #0F172A;\n    margin-bottom: 4px;\n}",
    ".card-title {\n    font-size: 16px;\n    font-weight: 800;\n    color: #1E3A8A;\n    text-transform: uppercase;\n    letter-spacing: 0.05em;\n    margin-bottom: 6px;\n}"
)

# 4. Cluster 2 color -> Kuning
content = content.replace('#D97706', '#EAB308')

# 5. Bold important words in Insight text
# We will just replace specific words with <b>word</b>
words_to_bold = [
    'Indonesia Timur', 'Papua Tengah', 'Nusa Tenggara', 'disparitas geografis', 'rata-rata nasional',
    'Pulau Jawa', 'wilayah timur', 'Papua', 'Jawa', 'Bali', 'rentang (variabilitas)', 'median tertinggi',
    'Cluster 1', 'Risiko Rendah', 'Cluster 3', 'Kritis', 'polarisasi', 'korelasi negatif', 'korelasi positif',
    'IPM', 'TPT', 'SMK_Komp', 'Kemiskinan', 'NTT', 'ketimpangan struktural', 'R² sebesar 0.682', 'signifikansi statistik',
    'RMSE sebesar 2.63%', 'gradien barat-timur', 'ruang PCA', 'K-Means', 'Papua Pegunungan',
    'investasi pembangunan manusia'
]
# Ensure we don't bold something already bolded, or inside tags
for word in words_to_bold:
    # Use regex to match the exact word/phrase not preceded or followed by tag brackets
    pattern = re.compile(r'(?<!<)(?<!<b>)' + re.escape(word) + r'(?!</b>)(?!>)')
    content = pattern.sub(f'<b>{word}</b>', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Update completed successfully.")
