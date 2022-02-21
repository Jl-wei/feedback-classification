import pandas as pd

df = pd.read_excel('P3-Golden.xlsx')

df['Judgement'] = df['labels'].replace(['none', 'Stability', 'quality', 'performance', 'feature'],
                        [0, 1, 2, 3, 4], inplace=False)

df.to_excel('P3-Golden.xlsx', index=False)

print(df['Judgement'].value_counts())
