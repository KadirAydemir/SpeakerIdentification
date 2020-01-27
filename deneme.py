import pandas as pd
import os
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.sql import select
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


engine = create_engine("mysql+pymysql://root:003785Kadir.@127.0.0.1/speakeridentificationdemo")
con = engine.connect()
mycursor = engine.connect()

df1 = pd.read_csv(os.path.join(app.instance_path, "features.csv"), error_bad_lines=False, encoding='utf-8')
df2 = pd.read_csv(os.path.join(app.instance_path, "users.csv"), error_bad_lines=False, encoding='utf-8')
df2['Name'] = df2['0']
df2 = df2.drop(columns=['0'])
result = pd.concat([df2, df1], axis=1, sort=False)
result.to_sql(con=con,name='features', if_exists='replace',index=False)

result = con.execute('SELECT * FROM features')
rows = result.fetchall()
rows = np.asarray(rows)
print(rows[:,0])