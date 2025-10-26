import pandas as pd

src = "data/Synthetic-Persona-Chat_train.csv"
dst = "data/Synthetic-Persona-Chat_train-mini.csv"

df = pd.read_csv(src, nrows=100, encoding="utf-8")
df.to_csv(dst, index=False, encoding="utf-8")
