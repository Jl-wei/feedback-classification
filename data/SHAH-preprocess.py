import xml.etree.ElementTree as ET
import xlsxwriter
import pandas as pd

def convert_xml_to_xlsx(xml_name, worksheet, i):
    tree = ET.parse(f"./SHAH_UNION/{xml_name}.xml")
    root = tree.getroot()

    for child in root:
        review = child.find('text').text
        label = child.find('aspectTerms').find('aspectTerm').get('class') if child.find('aspectTerms') else 'rate'
        print(review, label)
        
        # The '=)' review will cause error in BertTokenizer
        if (review != '=)'):
            worksheet.write_string(i, 0, review)
            worksheet.write_string(i, 1, label)
            i += 1
    
    return i


if __name__ == '__main__':
    workbook = xlsxwriter.Workbook("SHAH.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'reviews')
    worksheet.write(0, 1, 'labels')
    i = 1
    
    i = convert_xml_to_xlsx('ANGRY_BIRD', worksheet, i)
    i = convert_xml_to_xlsx('DROP_BOX', worksheet, i)
    i = convert_xml_to_xlsx('EVERNOTE', worksheet, i)
    i = convert_xml_to_xlsx('PIC_ART', worksheet, i)
    i = convert_xml_to_xlsx('PINTEREST', worksheet, i)
    i = convert_xml_to_xlsx('TRIP_ADVISOR', worksheet, i)
    i = convert_xml_to_xlsx('WHATSAPP', worksheet, i)
    
    workbook.close()
    
    df = pd.read_excel('SHAH.xlsx')

    df['Judgement'] = df['labels'].replace(['rate', 'E', 'R', 'B', 'M'], [0, 1, 2, 3, 3], inplace=False)

    df.to_excel('SHAH.xlsx', index=False)
    
    print(df['Judgement'].value_counts())
