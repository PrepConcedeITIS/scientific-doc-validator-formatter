import json
import os
from TexSoup import TexSoup, TexNode

def extract_standalone(texsoup_item, standalone_section_names):
    result = {}
    error_blocks = []
    for i, block in enumerate(standalone_section_names):
        found = texsoup_item.find(block)
        if found is None:
            error_blocks.append(block)
        else:
            #if getattr(found, 'contents', None) is not TexNode:
            if isinstance(found, TexNode) == False or isinstance(getattr(found, 'contents', None), list) == False:
                error_blocks.append(block)
                continue
            #todo: нужно ли удалять новую строку
            #block_content_buffer = ''.join(map(lambda x: str(x).replace('\n', ' '), found.contents))
            block_content_buffer = ''.join(map(lambda x: str(x), found.contents))
            block_content_buffer = block_content_buffer.strip()
            result[block] = block_content_buffer
            #merge contents if obj is texnode
    return result, error_blocks

def extract_sections(texsoup_item, tex_file_content):
    bibliography_block_name = 'thebibliography'
    all_sections = texsoup_item.find_all("section")
    sorted_sections = list(sorted(all_sections, key=lambda s: s.position))
    bibliography = texsoup_item.find(bibliography_block_name)

    sections_parsed = {}
    for i, section in enumerate(sorted_sections):
        if i == (len(sorted_sections) - 1):
            section_text_content = tex_file_content[section.position: bibliography.position]
        else:
            section_text_content = tex_file_content[section.position: sorted_sections[i + 1].position]
        sections_parsed[str(section.string).lower()] = section_text_content

    sections_parsed_trimmed = {}
    for i, (key, value) in enumerate(sections_parsed.items()):
        parsed_section = TexSoup(value)
        section_text_items = parsed_section.all[1:]
        items_to_be_merged_to_string = list(
            filter(lambda item: item.strip() != '',
                   map(lambda item: str(item), section_text_items)))
        final_string = ''.join(items_to_be_merged_to_string)
        sections_parsed_trimmed[key] = final_string

    return sections_parsed_trimmed

def extract_tex_to_json_object(tex_file_content, standalone_section_names, section_names):
    texsoup_item = TexSoup(tex_file_content)
    standalone_extracts, errors = extract_standalone(texsoup_item, standalone_section_names)
    sections_extracts = extract_sections(texsoup_item, tex_file_content)

    named_sections = {}
    main_parts = []
    for section_name in sections_extracts.keys():
        if section_name in section_names:
            named_sections[section_name] = sections_extracts[section_name]
        else:
            main_parts.append(sections_extracts[section_name])

    sections_result = {}

    for section_name in standalone_section_names:
        sections_result[section_name] = standalone_extracts.get(section_name, '')
    for section_name in section_names:
        sections_result[section_name] = named_sections.get(section_name, '')
    sections_result['main_part'] = main_parts

    result = {'sections': sections_result, 'errors': errors}

    return result

def export_parsed_sections_to_json(parsed_sections_dict, file_name, folder):
    with open(f'{folder}/{file_name}_sections.json', 'w', encoding='utf-8') as json_file:
        json.dump(parsed_sections_dict, json_file, indent=4)


def main():
    standalone_block_names = ['title', 'abstract', 'author', 'email', 'shorttit', 'keywords', ]
    bibliografy_block_name = 'thebibliography'
    section_block_names = ['introduction', 'conclusion', 'acknowledgements']

    # Путь к папке с файлами
    tex_folder = "G:/UserData/Desktop/disser/TeX-collection-sections-fix/TeX-collection"
    file_paths = [os.path.join(tex_folder, f) for f in os.listdir(tex_folder) if f.endswith(".tex")]
    error_files = []

    for path in file_paths:
        try:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path, "r", encoding="latin-1") as f:
                        file_content = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="cp1252") as f:
                        file_content = f.read()
            parsed_sections_dict = extract_tex_to_json_object(file_content, standalone_block_names, section_block_names)
            file_name = os.path.basename(path)
            export_parsed_sections_to_json(parsed_sections_dict, file_name, 'processed_tex')
        except Exception as e:
            print(f"Ошибка при обработке файла {path}: {e}")
            error_files.append(path)

    with open(f'processed_tex/errors.json', 'w', encoding='utf-8') as json_file:
        json.dump(error_files, json_file, indent=4)

main()