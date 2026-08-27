"""
Conta quantas vezes cada termo/giria ja aparece no dataset (todos os arquivos
em data/) e aponta quais estao super-representados, para os scripts de geracao
(generate_targeted_normalizer_eval_with_openai.py, build_*_term_specs.py) nao
gerarem mais conteudo duplicado para esses termos.

Fontes lidas (recursivo em data/):
- Arquivos de "term specs" (dict com "term", usados para pedir geracao): cada
  spec conta 1 ocorrencia por termo.
- Arquivos de linhas geradas/aprovadas (dict com "term" ou "target_term",
  ex: master_normalizer_train.json, targeted_normalizer_eval_approved.json,
  slang_normalizer_v4_train.json): cada linha conta 1 ocorrencia.
- Arquivos sem campo de termo explicito, mas com "slang"/"slang_text" curto
  (ate 4 palavras): tratado como termo.
- slang_normalization.csv: coluna slang_text, mesma heuristica de tamanho.

Arquivos de relatorio/report (nome contendo "_report") sao ignorados, assim
como o proprio output deste script.

Uso:
  python scripts/analyze_term_frequency.py
  python scripts/analyze_term_frequency.py --min-count 25
  python scripts/analyze_term_frequency.py --data-dir data --output data/term_frequency_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data")
DEFAULT_REPORT_PATH = Path("data/term_frequency_report.json")
DEFAULT_OVERREPRESENTED_PATH = Path("data/overrepresented_terms.json")

MAX_FALLBACK_TERM_WORDS = 4
SKIP_NAME_FRAGMENTS = ("_report",)
# Gerado por este proprio script; nunca ler como fonte.
SKIP_FILENAMES = {DEFAULT_REPORT_PATH.name, DEFAULT_OVERREPRESENTED_PATH.name}


def norm_term(term: str) -> str:
    term = " ".join(str(term or "").strip().lower().split())
    term = re.sub(r"^[^\w#@]+|[^\w!?]+$", "", term)
    return term


def extract_term(row: dict[str, Any]) -> str | None:
    for key in ("term", "target_term"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return norm_term(value)
    for key in ("slang", "slang_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if len(candidate.split()) <= MAX_FALLBACK_TERM_WORDS:
                return norm_term(candidate)
    return None


def should_skip(path: Path) -> bool:
    if path.name in SKIP_FILENAMES:
        return True
    return any(fragment in path.name for fragment in SKIP_NAME_FRAGMENTS)


def iter_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []

    if isinstance(data, dict):
        data = data.get("terms") or data.get("data") or []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def iter_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def analyze(data_dir: Path) -> tuple[Counter, dict[str, Counter]]:
    """Retorna (contagem total por termo, contagem por termo por arquivo)."""
    total_counts: Counter = Counter()
    per_file_counts: dict[str, Counter] = defaultdict(Counter)

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue

        if path.suffix == ".json":
            rows = iter_json_rows(path)
        elif path.suffix == ".csv":
            rows = iter_csv_rows(path)
        else:
            continue

        rel_name = str(path.relative_to(data_dir))
        for row in rows:
            term = extract_term(row)
            if not term:
                continue
            total_counts[term] += 1
            per_file_counts[rel_name][term] += 1

    return total_counts, per_file_counts


def build_report(
    total_counts: Counter,
    per_file_counts: dict[str, Counter],
    min_count: int,
) -> dict[str, Any]:
    terms_report = []
    for term, count in total_counts.most_common():
        sources = {
            file_name: file_counts[term]
            for file_name, file_counts in per_file_counts.items()
            if file_counts.get(term)
        }
        terms_report.append(
            {
                "term": term,
                "count": count,
                "overrepresented": count >= min_count,
                "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
            }
        )

    overrepresented = [t["term"] for t in terms_report if t["overrepresented"]]

    return {
        "min_count_threshold": min_count,
        "unique_terms": len(terms_report),
        "total_occurrences": sum(total_counts.values()),
        "overrepresented_count": len(overrepresented),
        "terms": terms_report,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conta a frequencia de termos/girias no dataset e lista os super-representados."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-count", type=int, default=20, help="Contagem minima para marcar um termo como super-representado.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH, help="Relatorio completo (contagem por termo e por arquivo).")
    parser.add_argument("--overrepresented-output", type=Path, default=DEFAULT_OVERREPRESENTED_PATH, help="Lista simples de termos a evitar na proxima geracao.")
    parser.add_argument("--top", type=int, default=30, help="Quantos termos mostrar no resumo impresso.")
    args = parser.parse_args()

    total_counts, per_file_counts = analyze(args.data_dir)
    report = build_report(total_counts, per_file_counts, args.min_count)

    write_json(args.output, report)
    overrepresented_terms = [t["term"] for t in report["terms"] if t["overrepresented"]]
    write_json(args.overrepresented_output, overrepresented_terms)

    print(f"Termos unicos encontrados: {report['unique_terms']}")
    print(f"Ocorrencias totais: {report['total_occurrences']}")
    print(f"Termos super-representados (>= {args.min_count} ocorrencias): {report['overrepresented_count']}")
    print(f"\nTop {args.top} termos mais repetidos:")
    for entry in report["terms"][: args.top]:
        flag = " [SUPER-REPRESENTADO]" if entry["overrepresented"] else ""
        print(f"  {entry['count']:>5}  {entry['term']}{flag}")

    print(f"\nRelatorio completo: {args.output}")
    print(f"Lista de termos a evitar: {args.overrepresented_output}")


if __name__ == "__main__":
    main()
