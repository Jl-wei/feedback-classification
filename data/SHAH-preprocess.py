import xml.etree.ElementTree as ET
import xlsxwriter

def convert_xml_to_xlsx(xml_name, worksheet, i):
    tree = ET.parse(f"./SHAH_UNION/{xml_name}.xml")
    root = tree.getroot()

    for child in root:
        review = child.find('text').text
        label = child.find('aspectTerms').find('aspectTerm').get('class') if child.find('aspectTerms') else 'rate'
        print(review, label)
        
        worksheet.write(i, 0, review)
        worksheet.write(i, 1, label)
        i += 1


if __name__ == '__main__':
    workbook = xlsxwriter.Workbook("SHAH.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'reviews')
    worksheet.write(0, 1, 'labels')
    i = 1
    
    convert_xml_to_xlsx('ANGRY_BIRD', worksheet, i)
    convert_xml_to_xlsx('DROP_BOX', worksheet, i)
    convert_xml_to_xlsx('EVERNOTE', worksheet, i)
    convert_xml_to_xlsx('PIC_ART', worksheet, i)
    convert_xml_to_xlsx('PINTEREST', worksheet, i)
    convert_xml_to_xlsx('TRIP_ADVISOR', worksheet, i)
    convert_xml_to_xlsx('WHATSAPP', worksheet, i)
    
    workbook.close()