import re
import html

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_srcdoc(slide_html):
    match = re.search(r'srcdoc="([^"]*)"', slide_html)
    if match:
        return html.unescape(match.group(1))
    return None

def set_srcdoc(slide_html, new_srcdoc):
    escaped = html.escape(new_srcdoc, quote=True)
    return re.sub(r'srcdoc="[^"]*"', f'srcdoc="{escaped}"', slide_html)

def merge_slides(slide1_content, slide2_content, title_suffix=""):
    # Extract CSS to modify
    style_pattern = r'<style>(.*?)</style>'
    
    # slide1 is the base
    s1_doc = get_srcdoc(slide1_content)
    s2_doc = get_srcdoc(slide2_content)
    
    if not s1_doc or not s2_doc:
        print("Error: Could not extract srcdoc")
        return slide1_content

    # 1. Update Title
    if title_suffix:
        # Use raw string for backreference \1
        s1_doc = re.sub(r'<title>(.*?)</title>', r'<title>\1 ' + title_suffix + '</title>', s1_doc)
        # Also update h1 if possible
        s1_doc = re.sub(r'<h1 class="title">.*?</h1>', f'<h1 class="title">Scenario Workflow (Complete)</h1>', s1_doc)

    # 2. Extract content from Slide 2
    
    s2_steps_block = re.search(r'<div class="workflow-steps">(.*?)</div>\s*<!-- (Info Panel|Ranking)', s2_doc, re.DOTALL)
    if not s2_steps_block:
         s2_steps_block = re.search(r'<div class="workflow-steps">(.*?)</div>\s*<div class="(info-panel|ranking-section)"', s2_doc, re.DOTALL)
    
    s2_steps_inner = ""
    start_marker = '<div class="workflow-steps">'
    start_idx = s2_doc.find(start_marker)
    if start_idx != -1:
        end_idx = -1
        next_block = re.search(r'<div class="(info-panel|ranking-section)"', s2_doc[start_idx:])
        if next_block:
            end_idx = start_idx + next_block.start()
            s2_steps_fragment = s2_doc[start_idx:end_idx]
            last_div_idx = s2_steps_fragment.rfind('</div>')
            if last_div_idx != -1:
                 s2_steps_inner = s2_steps_fragment[len(start_marker):last_div_idx]
    
    # 3. Extract Ranking Section from S2
    ranking_section = ""
    ranking_match = re.search(r'(<div class="ranking-section">.*?</div>)\s*<!-- Footer', s2_doc, re.DOTALL)
    if not ranking_match:
         ranking_match = re.search(r'(<div class="ranking-section">.*?</div>)\s*<div class="footer"', s2_doc, re.DOTALL)
    
    if ranking_match:
        ranking_section = ranking_match.group(1)

    # 4. Modify S1 CSS
    new_css = """
        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            grid-template-rows: auto auto;
            gap: 20px;
            overflow-y: auto;
            align-items: start;
            padding: 20px 60px;
        }
        .workflow-steps {
            grid-column: 1;
            grid-row: 1;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .info-panel {
            grid-column: 2;
            grid-row: 1;
            flex: none;
            width: 100%;
            height: auto;
        }
        .ranking-section {
            grid-column: 1 / -1;
            grid-row: 2;
            margin-top: 10px;
        }
        /* Adjust step padding */
        .workflow-step { padding: 15px; }
    """
    
    # Use raw string for backreference \1
    s1_doc = re.sub(r'<style>(.*?)</style>', r'<style>\1 ' + new_css + '</style>', s1_doc, flags=re.DOTALL)

    # 5. Insert content into S1
    insert_point_steps = -1
    info_panel_match = re.search(r'<div class="info-panel">', s1_doc)
    if info_panel_match:
        info_comment = s1_doc.find('<!-- Info Panel -->')
        if info_comment != -1:
             insert_point_steps = s1_doc.rfind('</div>', 0, info_comment)
        else:
             insert_point_steps = s1_doc.rfind('</div>', 0, info_panel_match.start())

    if insert_point_steps != -1 and s2_steps_inner:
        s1_doc = s1_doc[:insert_point_steps] + s2_steps_inner + s1_doc[insert_point_steps:]

    footer_comment = s1_doc.find('<!-- Footer -->')
    if footer_comment != -1:
        insert_point_ranking = s1_doc.rfind('</div>', 0, footer_comment)
        if insert_point_ranking != -1:
            s1_doc = s1_doc[:insert_point_ranking] + ranking_section + s1_doc[insert_point_ranking:]
            
    return set_srcdoc(slide1_content, s1_doc)


def main():
    content = read_file('html slides/project_presentation.html')
    
    slide_pattern = re.compile(r'(<section class="slide".*?</section>)', re.DOTALL)
    parts = slide_pattern.split(content)
    
    header = parts[0]
    slides = []
    footer = parts[-1]
    
    for p in parts[1:-1]:
        if p.strip().startswith('<section'):
            slides.append(p)
            
    print("Merging slides 7 and 8...")
    # Slides index 6 and 7 (0-based) match user's 7 and 8 (1-based)?
    # User said "7-8". Slides in file are data-index="1" to "21".
    # slides[0] is data-index="1".
    # So slides[6] is data-index="7". Correct.
    
    slides[6] = merge_slides(slides[6], slides[7], "Complete")
    
    s9 = slides[8]
    s10 = slides[9]
    
    s12 = slides[11]
    s13 = slides[12]
    
    print("Merging slides 9 and 10...")
    s9_merged = merge_slides(s9, s10, "Complete")
    
    print("Merging slides 12 and 13...")
    s12_merged = merge_slides(s12, s13, "Complete")
    
    new_slides = []
    
    # 1-6 (Indices 0-5)
    new_slides.extend(slides[0:6])
    
    # 7 Merged (Index 6)
    new_slides.append(slides[6])
    
    # 8 Skipped (Index 7)
    
    # 9 Merged (Index 8) -> using s9_merged
    new_slides.append(s9_merged)
    
    # 10 Skipped (Index 9)
    
    # 11 Untouched (Index 10)
    new_slides.append(slides[10])
    
    # 12 Merged (Index 11) -> using s12_merged
    new_slides.append(s12_merged)
    
    # 13 Skipped (Index 12)
    
    # Reordering remaining: 
    # Original: 14, 15, 16, 17, 18, 19, 20, 21 (Indices 13..20)
    # Requested: "15,16,17 after 18,19"
    # Order: 14, 18, 19, 15, 16, 17, 20, 21
    
    old_14 = slides[13]
    old_15 = slides[14]
    old_16 = slides[15]
    old_17 = slides[16]
    old_18 = slides[17]
    old_19 = slides[18]
    old_20 = slides[19]
    old_21 = slides[20]
    
    new_slides.append(old_14)
    new_slides.append(old_18)
    new_slides.append(old_19)
    new_slides.append(old_15)
    new_slides.append(old_16)
    new_slides.append(old_17)
    new_slides.append(old_20)
    new_slides.append(old_21)
    
    final_slides_html = []
    for i, slide in enumerate(new_slides):
        updated_slide = re.sub(r'data-index="\d+"', f'data-index="{i+1}"', slide)
        final_slides_html.append(updated_slide)
        
    final_content = header + "\n".join(final_slides_html) + footer
    
    write_file('html slides/project_presentation.html', final_content)
    print("Done.")

if __name__ == "__main__":
    main()