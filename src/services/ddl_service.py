import re

def load_ddl_text(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


def parse_ddl(ddl_text: str) -> dict:
    tables = {}
    table_defs = re.findall(r'CREATE TABLE\s+(\w+)\s*\((.*?)\);', ddl_text, re.S | re.I)
    
    for table_name, columns_str in table_defs:
        columns = [col.strip() for col in columns_str.split(',') if col.strip()]
        tables[table_name] = columns
        
    return tables
