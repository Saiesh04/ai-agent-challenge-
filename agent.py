import os
import argparse
import importlib.util
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import difflib
import google.generativeai as genai
import fitz  # PyMuPDF

# === Configuration ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-flash"

# === Utility: Clean LLM Code Output ===
def clean_llm_output(raw_code: str) -> str:
    lines = raw_code.strip().splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines)

# === CSV Utilities ===
def get_csv_info(csv_path):
    df = pd.read_csv(csv_path, nrows=20)
    header = ",".join(list(df.columns))
    sample = df.head(5).to_csv(index=False)
    return header, sample

# === Always-available fallback parser template ===
def fallback_parser_code(header: str) -> str:
    columns = [c.strip() for c in header.split(",")]
    col_list = "[" + ", ".join(f"'{c}'" for c in columns) + "]"
    return f"""
import pandas as pd
import fitz  # PyMuPDF

def parse(pdf_path: str) -> pd.DataFrame:
    columns = {col_list}
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        lines = text.splitlines()
        rows = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) >= 5 and "-" in parts[0]:
                date = parts[0]
                balance = parts[-1]
                credit = parts[-2]
                debit = parts[-3]
                description = " ".join(parts[1:-3])
                rows.append([date, description, debit if debit != '-' else '', credit if credit != '-' else '', balance])
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns)
"""

# === LLM Code Generation with auto fallback ===
def llm_generate_parser(bank, header, sample, parser_filename):
    prompt = f"""You are an expert Python data engineer.
Given this CSV schema:
{header}
And these sample rows:
{sample}
Write a Python module named '{parser_filename}' in the 'custom_parsers' folder which exports a function:
    def parse(pdf_path: str) -> pd.DataFrame
The function should:
- Parse the {bank.upper()} statement PDF at pdf_path, and return all transactions matching exactly the columns and structure above.
- Prefer using camelot for tables; fallback to PyMuPDF text parsing if no tables found.
- Return an empty DataFrame with correct columns if unable to parse.
All code should be inside parse(). No comments or logging."""
    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt)
        raw = resp.text if hasattr(resp, "text") else resp.candidates[0]['content']['parts'][0]['text']
        code = clean_llm_output(raw)
        if "def parse" not in code:
            raise ValueError("No parse() in generated code")
        return code
    except Exception as e:
        print(f"⚠️ LLM generation failed: {e}")
        return fallback_parser_code(header)

# === Parser Execution & Test Runner ===
def run_test(bank):
    parser_path = Path("custom_parsers") / f"{bank}_parser.py"
    spec = importlib.util.spec_from_file_location(bank, parser_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    expected = pd.read_csv(f"data/{bank}/{bank}_sample.csv")
    actual = mod.parse(f"data/{bank}/{bank}_sample.pdf")

    expected.columns = expected.columns.str.strip()
    actual.columns = actual.columns.str.strip()

    for col in expected.columns:
        if col in actual.columns:
            if expected[col].dtype in ['float64', 'int64']:
                actual[col] = pd.to_numeric(actual[col], errors='coerce')
            else:
                actual[col] = actual[col].astype(str).str.strip()

    exp = expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
    act = actual.sort_values(by=list(actual.columns)).reset_index(drop=True)

    return act.equals(exp), act, exp

def diff_dataframes(expected, actual):
    try:
        exp_lines = expected.fillna("").to_csv(index=False).splitlines()
        act_lines = actual.fillna("").to_csv(index=False).splitlines()
        diff = list(difflib.unified_diff(exp_lines, act_lines, fromfile="expected.csv", tofile="actual.csv"))
        snippet = "\n".join(diff[:30])
        return snippet + ("\n...diff truncated..." if len(diff) > 30 else "")
    except Exception as e:
        return f"Diff generation error: {e}"

# === Folder Setup ===
def ensure_custom_parsers_dir():
    p = Path("custom_parsers")
    p.mkdir(exist_ok=True)
    init = p / "__init__.py"
    if not init.exists():
        init.write_text("")

def ensure_test_file(bank):
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    test_path = test_dir / f"test_{bank}_parser.py"
    if not test_path.exists():
        test_path.write_text(f"""import pandas as pd
from custom_parsers.{bank}_parser import parse

def test_parse():
    df_e = pd.read_csv('data/{bank}/{bank}_sample.csv')
    df_a = parse('data/{bank}/{bank}_sample.pdf')
    assert df_e.equals(df_a), "Parsed DataFrame mismatch."
""")

# === CLI Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank name, e.g. icici")
    args = parser.parse_args()
    bank = args.target.lower()

    csv_path = Path("data") / bank / f"{bank}_sample.csv"
    pdf_path = Path("data") / bank / f"{bank}_sample.pdf"
    if not csv_path.exists():
        print(f"❌ Missing CSV at {csv_path}")
        return
    if not pdf_path.exists():
        print(f"❌ Missing PDF at {pdf_path}")
        return

    ensure_custom_parsers_dir()
    ensure_test_file(bank)

    header, sample = get_csv_info(csv_path)
    parser_filename = f"{bank}_parser.py"

    passed = False
    for attempt in range(1, 4):
        print(f"\n🔁 Attempt {attempt} generating parser for '{bank}'...")

        code = llm_generate_parser(bank, header, sample, parser_filename)
        if not code.strip():
            print("⚠️ Empty LLM code. Using fallback.")
            code = fallback_parser_code(header)

        parser_path = Path("custom_parsers") / parser_filename
        parser_path.write_text(code)
        Path("custom_parsers" / f"{bank}_attempt{attempt}.py").write_text(code)

        print("\n📄 Generated Code:\n" + "-" * 50)
        print(code)
        print("-" * 50)

        try:
            passed, actual, expected = run_test(bank)
        except Exception as e:
            print(f"[Attempt {attempt}] ⚠️ Runtime error: {e}")
            passed, actual, expected = False, pd.DataFrame(), pd.DataFrame()

        if passed:
            print(f"✅ SUCCESS in attempt {attempt}")
            break
        else:
            print(f"❌ FAIL attempt {attempt}")
            print("Expected head:\n", expected.head())
            print("Actual head:\n", actual.head())
            print("Diff:\n", diff_dataframes(expected, actual))

            actual.to_csv("actual_output.csv", index=False)
            expected.to_csv("expected_output.csv", index=False)

    if not passed:
        print("❌ FINAL FAIL: Could not parse after 3 attempts.")
        print("Saved actual_output.csv and expected_output.csv for inspection.")

if __name__ == "__main__":
    main()
