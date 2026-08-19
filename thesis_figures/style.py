ACCENT = "#2E75B6"
ACCENT_DARK = "#1B4F7A"
ACCENT_LIGHT = "#EBF4FB"
GREY = "#595959"
LIGHT_GREY = "#DDDDDD"
EN_COLOR = "#C44E52"
JA_COLOR = "#55A868"
GOLD = "#CCB974"

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
try:
    fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK JP']
except Exception:
    pass
