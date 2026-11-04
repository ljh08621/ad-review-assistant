"""Bounded XLSX import: anchored embedded images only, no network or disk extraction."""
import base64
import io
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from materials import images_from_file, MAX_BYTES

MAX_EXCEL = 20 * 1024 * 1024
S='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
D='http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing'
A='http://schemas.openxmlformats.org/drawingml/2006/main'

def parse_excel(raw, max_rows=10):
    if len(raw)>MAX_EXCEL: raise ValueError('Excel不能超过20 MB。')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)>1000 or sum(x.file_size for x in infos)>64*1024*1024 or len({x.filename for x in infos})!=len(infos):
                raise ValueError('工作簿解压体积过大或结构异常。')
            if any('richData/' in n or 'cellimages' in n.lower() for n in z.namelist()):
                raise ValueError('不支持单元格内图片，请改用普通插入图片并放在对应行B列。')
            def xml(path):
                data=z.read(path)
                if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper(): raise ValueError('不支持含外部实体的工作簿。')
                return ET.fromstring(data)
            def rels(path):
                relpath=posixpath.join(posixpath.dirname(path),'_rels',posixpath.basename(path)+'.rels')
                if relpath not in z.namelist(): return {}
                result={}
                for node in xml(relpath):
                    if node.get('TargetMode')=='External': continue
                    target=node.get('Target','')
                    resolved=posixpath.normpath(posixpath.join(posixpath.dirname(path),target)) if not target.startswith('/') else target.lstrip('/')
                    if resolved.startswith('../'): raise ValueError('工作簿路径异常。')
                    result[node.get('Id')]=resolved
                return result
            sheets=xml('xl/workbook.xml').findall(f'{{{S}}}sheets/{{{S}}}sheet')
            if len(sheets)!=1: raise ValueError('请仅保留一个工作表，避免遗漏其他页的数据。')
            sheetpath=rels('xl/workbook.xml')[sheets[0].get(f'{{{R}}}id')]
            sheet=xml(sheetpath)
            if sheet.find(f'{{{S}}}mergeCells') is not None: raise ValueError('请取消合并单元格后上传。')
            strings=[]
            if 'xl/sharedStrings.xml' in z.namelist():
                strings=[''.join(n.itertext()) for n in xml('xl/sharedStrings.xml')]
            cells={};formula_rows=set()
            for c in sheet.findall(f'.//{{{S}}}sheetData/{{{S}}}row/{{{S}}}c'):
                ref=c.get('r','');v=c.find(f'{{{S}}}v');value=v.text if v is not None and v.text else ''
                if c.get('t')=='s':value=strings[int(value)]
                elif c.get('t')=='inlineStr': value=''.join(n.text or '' for n in c.findall(f'.//{{{S}}}t'))
                if c.find(f'{{{S}}}f') is not None: formula_rows.add(int(''.join(filter(str.isdigit,ref))))
                if value.strip(): cells[ref]=value
            if 1 in formula_rows: raise ValueError('表头不能使用公式。')
            if cells.get('A1','').strip()!='文本' or cells.get('B1','').strip()!='图片': raise ValueError('表头必须为A1“文本”、B1“图片”。')
            if any(''.join(filter(str.isalpha,k)) not in ('A','B') for k in cells): raise ValueError('仅支持“文本”和“图片”两列，请移除额外数据列。')
            pictures={};sheetrels=rels(sheetpath)
            for drawing in sheet.findall(f'{{{S}}}drawing'):
                path=sheetrels[drawing.get(f'{{{R}}}id')];links=rels(path)
                for anchor in xml(path):
                    start=anchor.find(f'{{{D}}}from');pics=anchor.findall(f'{{{D}}}pic')
                    if start is None or len(pics)!=1: raise ValueError('图片位置不明确，请用普通插入图片，左上角放在对应行B列。')
                    row=int(start.find(f'{{{D}}}row').text)+1;col=int(start.find(f'{{{D}}}col').text)
                    if col!=1 or row<2: raise ValueError('每张图片左上角必须位于对应数据行的B列。')
                    blip=pics[0].find(f'.//{{{A}}}blip');rid=blip.get(f'{{{R}}}embed') if blip is not None else None
                    if rid not in links: raise ValueError('请嵌入图片，不支持外链图片。')
                    pictures.setdefault(row,[]).append(links[rid])
            rows=sorted({int(''.join(filter(str.isdigit,k))) for k in cells if k not in ('A1','B1')}|set(pictures)| (formula_rows-{1}))
            if not 1<=len(rows)<=max_rows: raise ValueError(f'每份Excel需包含1～{max_rows}条素材。')
            output=[]
            for row in rows:
                text=cells.get(f'A{row}','').strip();errors=[];images=[]
                if len(text)>20000:errors.append('文本超过20000字')
                if row in formula_rows:errors.append('不支持公式，请粘贴为纯文本；图片请用普通插入图片')
                paths=pictures.get(row,[])
                if cells.get(f'B{row}','').strip():errors.append('图片列仅支持普通插入图片，请移除链接或单元格文字')
                if len(paths)>4:errors.append('每行最多4张图片')
                elif sum(z.getinfo(path).file_size for path in paths)>MAX_BYTES:errors.append('本行图片合计超过8 MB')
                else:
                    for index,path in enumerate(paths,1):
                        suffix=posixpath.splitext(path)[1].lower()
                        name=f'第{row}行-图{index}{suffix}';image=z.read(path)
                        try:
                            mime={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp'}.get(suffix)
                            if not mime:raise ValueError('图片仅支持PNG/JPG/WEBP')
                            images_from_file(name,image)
                            images.append(dict(filename=name,image=f'data:{mime};base64,'+base64.b64encode(image).decode()))
                        except ValueError as e:errors.append(f'图{index}：{e}')
                output.append(dict(row=row,text=text,images=images,errors=errors))
            return output
    except (zipfile.BadZipFile, KeyError, ET.ParseError, IndexError, TypeError) as exc:
        raise ValueError('Excel损坏或结构不支持，请使用模板重新填写。') from exc
