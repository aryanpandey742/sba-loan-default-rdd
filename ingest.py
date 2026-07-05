import pandas as pd
import csv

def load_raw(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        sample = f.read(5000)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    sep = dialect.delimiter
    print(f"Detected separator: {repr(sep)}")
    return pd.read_csv(path, sep=sep, on_bad_lines='warn', engine='python')