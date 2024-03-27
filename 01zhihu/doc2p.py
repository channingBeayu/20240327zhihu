import re

import pandas as pd

from utils.conn import Conn


conn = Conn()
doc_infos = conn.get_contents()
df = pd.DataFrame(columns=('doc_id', 'p'))

for doc_info in doc_infos:
    doc_id, content = doc_info[0], doc_info[1]
    ps = re.split('\n', content)
    ps = [p.replace(" ", "") for p in ps if len(p) > 50]
    row = {'doc_id': [doc_id]*len(ps), 'p': ps, }
    df = df.append(pd.DataFrame(row), ignore_index=True)

conn.save_df('01docs_p', df)

