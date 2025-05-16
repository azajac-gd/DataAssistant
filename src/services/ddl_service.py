import re
from typing import List, Dict, Any


def parse_ddl(ddl_text: str) -> List[Dict[str, Any]]:
    tables = []
    table_defs = re.findall(r'CREATE TABLE\s+(\w+)\s*\((.*?)\);', ddl_text, re.S | re.I)

    for table_name, columns_str in table_defs:
        columns = []
        primary_keys = set()
        foreign_keys = []

        lines = [line.strip().rstrip(',') for line in columns_str.splitlines() if line.strip()]

        for line in lines:
            if "PRIMARY KEY" in line.upper() and "FOREIGN KEY" not in line.upper():
                match = re.match(r'(\w+)\s+(\w+).*PRIMARY KEY', line, re.I)
                if match:
                    col_name, col_type = match.groups()
                    columns.append({"name": col_name, "type": col_type.upper(), "primary_key": True})
                    primary_keys.add(col_name)
                    continue

            fk_match = re.match(r'FOREIGN KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)', line, re.I)
            if fk_match:
                col_name, ref_table, ref_col = fk_match.groups()
                foreign_keys.append({
                    "column": col_name,
                    "references": {
                        "table": ref_table,
                        "column": ref_col
                    }
                })
                continue

            pk_match = re.match(r'PRIMARY KEY\s*\(([\w,\s]+)\)', line, re.I)
            if pk_match:
                pk_cols = [col.strip() for col in pk_match.group(1).split(",")]
                primary_keys.update(pk_cols)
                continue

            col_match = re.match(r'(\w+)\s+(\w+)', line)
            if col_match:
                col_name, col_type = col_match.groups()
                columns.append({"name": col_name, "type": col_type.upper()})

        for col in columns:
            if col["name"] in primary_keys:
                col["primary_key"] = True

            for fk in foreign_keys:
                if fk["column"] == col["name"]:
                    col["foreign_key"] = fk["references"]

        tables.append({
            "table_name": table_name,
            "columns": columns
        })

    return tables
