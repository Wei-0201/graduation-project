from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


OUTPUT = r"D:\畢業專題\官晸曄_推薦信_A4.docx"
FONT = "Microsoft JhengHei"

doc = Document()
section = doc.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(2.25)
section.bottom_margin = Cm(2.15)
section.left_margin = Cm(2.55)
section.right_margin = Cm(2.55)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.font.color.rgb = None
normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
normal.paragraph_format.line_spacing = 1.45
normal.paragraph_format.space_after = Pt(0)

title = doc.styles["Title"]
title.font.name = FONT
title.font.size = Pt(16)
title.font.bold = False
title.font.color.rgb = RGBColor(0, 0, 0)
title.font.hidden = False
title._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
title.paragraph_format.space_after = Pt(23)
title.paragraph_format.line_spacing = 1.0
for border in title._element.xpath("./w:pPr/w:pBdr"):
    border.getparent().remove(border)
style_ppr = title._element.get_or_add_pPr()
style_pbdr = OxmlElement("w:pBdr")
style_bottom = OxmlElement("w:bottom")
style_bottom.set(qn("w:val"), "nil")
style_pbdr.append(style_bottom)
style_ppr.append(style_pbdr)

p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = p.add_run("推薦信")
title_run.font.color.rgb = RGBColor(0, 0, 0)

p = doc.add_paragraph("中原大學電機工程學系碩士班招生委員會　鈞鑒：")
p.paragraph_format.space_after = Pt(15)

body = [
    "本人任教於中原大學電機工程學系，曾教授官晸曄同學「電子學（一）」及「類比電路實驗」，並指導其 CACC 專題研究，同時擔任其校外實習輔導老師，對官同學在課程、專題及實習期間的表現有一定的了解。",
    "官同學在修課期間態度認真，能按時完成課程及實驗要求，並透過電子學與類比電路實驗的學習，累積電子電路與實作方面的基礎，從課程中受益良多。",
    "在 CACC 專題研究中，官同學與組員進行車輛控制、程式設計及系統整合等實作，過程中能配合團隊進度，遇到問題也會主動討論並嘗試解決，對控制及系統實作有相當的學習興趣。",
    "此外，本人亦擔任官同學的校外實習輔導老師。實習期間，官同學工作態度認真，能配合實習單位的工作安排，並將在校所學運用於實際工作中，整體實習表現良好。",
    "綜合官同學在課程、專題及校外實習的表現，可以看出其具備良好的學習態度及實作能力。相信進入本系碩士班後，能持續累積專業知識與研究經驗。因此，本人樂意推薦官晸曄同學申請本系碩士班。",
]

for content in body:
    p = doc.add_paragraph(content)
    p.paragraph_format.first_line_indent = Cm(0.85)
    p.paragraph_format.space_after = Pt(11)
    p.paragraph_format.keep_together = True

p = doc.add_paragraph("敬請　招生委員會予以考量。")
p.paragraph_format.first_line_indent = Cm(0.85)
p.paragraph_format.space_before = Pt(1)
p.paragraph_format.space_after = Pt(23)

p = doc.add_paragraph("推薦人：＿＿＿＿＿＿")
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
p.paragraph_format.space_after = Pt(9)

p = doc.add_paragraph("日期：＿＿年＿＿月＿＿日")
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

doc.core_properties.title = "推薦信"
doc.core_properties.subject = "官晸曄同學研究所碩士班申請推薦信"
doc.core_properties.author = ""
doc.save(OUTPUT)
print(OUTPUT)
