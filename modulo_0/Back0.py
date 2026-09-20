"""Backend unificado del Modulo 0.

Contiene dificultad, validacion, generadores, examen, PDF y servidor HTTP en un solo archivo.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import io
import json
import math
import os
import random
import re
import secrets
import textwrap
import time
import urllib.parse
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable

from typing import Any
import matplotlib.pyplot as plt

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr as sympy_parse_expr,
    standard_transformations,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# DIFICULTAD
# ---------------------------------------------------------------------------

DIFFICULTIES = ("facil", "normal", "avanzado")

DIFFICULTY_ALIASES = {
    "easy": "facil",
    "basico": "facil",
    "basica": "facil",
    "sencillo": "facil",
    "simple": "facil",
    "medio": "normal",
    "media": "normal",
    "intermedio": "normal",
    "medium": "normal",
    "dificil": "avanzado",
    "difícil": "avanzado",
    "hard": "avanzado",
    "advanced": "avanzado",
    "avanzada": "avanzado",
}


@dataclass(frozen=True)
class DifficultyProfile:
    key: str
    label: str
    level: int
    mental_seconds: int
    base_points: int
    max_terms: int
    max_variables: int
    allows_negatives: bool
    allows_fractions: bool


PROFILES = {
    "facil": DifficultyProfile(
        key="facil",
        label="Facil",
        level=1,
        mental_seconds=28,
        base_points=10,
        max_terms=2,
        max_variables=1,
        allows_negatives=False,
        allows_fractions=False,
    ),
    "normal": DifficultyProfile(
        key="normal",
        label="Normal",
        level=2,
        mental_seconds=22,
        base_points=16,
        max_terms=4,
        max_variables=2,
        allows_negatives=True,
        allows_fractions=False,
    ),
    "avanzado": DifficultyProfile(
        key="avanzado",
        label="Avanzado",
        level=3,
        mental_seconds=16,
        base_points=24,
        max_terms=6,
        max_variables=3,
        allows_negatives=True,
        allows_fractions=True,
    ),
}


class DifficultyError(ValueError):
    pass


def difficulty_key(value: str | None = None) -> str:
    raw = (value or "normal").strip().lower()
    key = DIFFICULTY_ALIASES.get(raw, raw)
    if key not in DIFFICULTIES:
        raise DifficultyError(f"Dificultad no valida: {value}")
    return key


def level(diff: str | None = None) -> int:
    return {"facil": 1, "normal": 2, "avanzado": 3}[difficulty_key(diff)]


def profile(diff: str | None = None) -> DifficultyProfile:
    return PROFILES[difficulty_key(diff)]


def difficulty_label(diff: str | None = None) -> str:
    return profile(diff).label


def mental_time(diff: str | None = None, complexity: int = 0) -> int:
    p = profile(diff)
    return max(8, p.mental_seconds - max(0, complexity) * (2 if p.level >= 2 else 1))


def timed_points(diff: str | None, seconds_left: float, seconds_total: float) -> int:
    p = profile(diff)
    if seconds_total <= 0:
        return p.base_points
    ratio = max(0.0, min(1.0, seconds_left / seconds_total))
    return max(1, round(p.base_points * (0.45 + 0.55 * ratio)))


# ---------------------------------------------------------------------------
# VALIDACION Y MATEMATICA
# ---------------------------------------------------------------------------

x, y, z = sp.symbols("x y z", positive=True)
VARIABLES = (x, y, z)

LOCALS = {
    "x": x,
    "y": y,
    "z": z,
    "sqrt": sp.sqrt,
    "root": sp.root,
    "Rational": sp.Rational,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "oo": sp.oo,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def nice(value: Any, digits: int = 6) -> str:
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if isinstance(value, sp.Rational) and not isinstance(value, sp.Integer):
        return f"{int(value.p)}/{int(value.q)}"
    if isinstance(value, sp.Integer):
        return str(int(value))
    if isinstance(value, float):
        if math.isclose(value, round(value), abs_tol=10**-digits):
            return str(int(round(value)))
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _exact_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, sp.Integer):
        return Decimal(int(value))
    if isinstance(value, sp.Rational):
        return Decimal(int(value.p)) / Decimal(int(value.q))
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(str(value))


def round_nice(value: Any, digits: int = 4) -> int | float:
    d = _exact_decimal(value)
    quant = Decimal(1).scaleb(-digits)
    r = float(d.quantize(quant, rounding=ROUND_HALF_UP))
    return int(r) if r == int(r) else r


def clean_num(raw: Any) -> str:
    s = str(raw).strip().lower().replace("$", "").replace("%", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    return s


def parse_fraction(raw: Any) -> Fraction:
    mixed = str(raw).strip().replace(" ", "_")
    s = clean_num(mixed)
    m = re.fullmatch(r"(-?\d+)_(\d+)/(\d+)", s)
    if m:
        whole, num, den = map(int, m.groups())
        part = Fraction(num, den)
        return Fraction(whole, 1) + (part if whole >= 0 else -part)
    return Fraction(s) if "/" in s else Fraction(Decimal(s))


def prep_expr(raw: Any) -> str:
    s = str(raw).strip()
    replacements = {
        "^": "**",
        "×": "*",
        "·": "*",
        "÷": "/",
        "−": "-",
        "√": "sqrt",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\)\s*\(", ")*(", s)
    s = re.sub(r"\)\s*([A-Za-z0-9])", r")*\1", s)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1*\2", s)
    s = re.sub(r"([xyz])\s*\(", r"\1*(", s)
    return s


def parse_math_expr(raw: Any) -> sp.Expr:
    if isinstance(raw, sp.Expr):
        return raw
    if isinstance(raw, Fraction):
        return sp.Rational(raw.numerator, raw.denominator)
    return sympy_parse_expr(
        prep_expr(raw),
        local_dict=LOCALS,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def to_expr(value: Any) -> sp.Expr:
    if isinstance(value, sp.Expr):
        return value
    return parse_math_expr(value)


def canonical(value: Any) -> sp.Expr:
    expr = to_expr(value)
    return sp.simplify(sp.cancel(sp.together(expr)))


def algebraically_equivalent(user_expr: Any, expected_expr: Any) -> bool:
    try:
        u = to_expr(user_expr)
        e = to_expr(expected_expr)
    except Exception:
        return False
    checks = (
        lambda a, b: sp.simplify(a - b),
        lambda a, b: sp.cancel(a - b),
        lambda a, b: sp.factor(sp.together(a - b)),
        lambda a, b: sp.simplify(sp.expand(a) - sp.expand(b)),
        lambda a, b: sp.simplify(sp.cancel(sp.together(a) - sp.together(b))),
        lambda a, b: sp.apart(sp.together(a - b)),
    )
    for fn in checks:
        try:
            if fn(u, e) == 0:
                return True
        except Exception:
            continue
    return False


def answer_obj(value: Any, kind: str, plain: str | None = None, latex: str | None = None) -> dict[str, Any]:
    kind = kind.lower()
    if kind in ("fraction", "mixed_fraction"):
        if isinstance(value, Fraction):
            f = value
        elif isinstance(value, sp.Rational):
            f = Fraction(int(value.p), int(value.q))
        else:
            f = Fraction(value)
        expr = sp.Rational(f.numerator, f.denominator)
        return {
            "type": kind,
            "plain": plain or nice(f),
            "latex": latex or sp.latex(expr),
            "value": f"{f.numerator}/{f.denominator}",
        }
    if kind == "number":
        v = round_nice(value) if not isinstance(value, int) else value
        return {
            "type": "number",
            "plain": plain or nice(v),
            "latex": latex or sp.latex(v),
            "value": v,
        }
    if kind == "percent":
        v = round_nice(value, 2)
        return {
            "type": "percent",
            "plain": plain or f"{nice(v)}%",
            "latex": latex or f"{nice(v)}\\%",
            "value": v,
        }
    if kind == "expression":
        e = canonical(value)
        return {
            "type": "expression",
            "plain": plain or sp.sstr(e),
            "latex": latex or sp.latex(e),
            "sympy": sp.sstr(e),
        }
    if kind == "symbol":
        return {"type": "symbol", "plain": plain or str(value), "latex": latex or str(value), "value": plain or str(value)}
    return {"type": "text", "plain": plain or str(value), "latex": latex or str(value), "value": plain or str(value)}


def _numeric_close(user_value: float, expected_value: float) -> bool:
    tolerance = 0.01 if abs(expected_value) < 100 else max(0.01, abs(expected_value) * 0.0005)
    return abs(user_value - expected_value) <= tolerance


def _labeled_qr(raw: str) -> tuple[str, str] | None:
    q = re.search(
        r"\b(?:q|cociente)\s*[:=]\s*(.+?)(?=(?:[,;]\s*\b(?:r|residuo|resto)\b\s*[:=])|$)",
        raw,
        re.IGNORECASE,
    )
    rem = re.search(r"\b(?:r|residuo|resto)\s*[:=]\s*(.+)$", raw, re.IGNORECASE)
    if q and rem:
        return q.group(1).strip(), rem.group(1).strip()
    return None


def _validate_polynomial_division(raw: str, metadata: dict[str, Any], expected_sympy: str) -> tuple[bool, str | None]:
    if algebraically_equivalent(raw, expected_sympy):
        return True, sp.sstr(canonical(raw))
    labeled = _labeled_qr(raw)
    if not labeled:
        return False, None
    quotient_raw, remainder_raw = labeled
    try:
        divisor = parse_math_expr(metadata["divisor"])
        candidate = parse_math_expr(quotient_raw) + parse_math_expr(remainder_raw) / divisor
    except Exception:
        return False, None
    ok = algebraically_equivalent(candidate, expected_sympy)
    return ok, sp.sstr(canonical(candidate)) if ok else None


def validate_payload(payload: dict[str, Any], user_answer: Any) -> dict[str, Any]:
    answer = payload.get("answer") or payload.get("expected")
    if not answer:
        raise ValueError("No hay respuesta para validar")
    metadata = payload.get("metadata") or {}
    qid = payload.get("id")
    kind = answer.get("type", "text")
    ok = False
    norm = None
    error_type = None
    try:
        if kind in ("number", "percent"):
            u = float(Decimal(clean_num(user_answer)))
            e = float(answer["value"])
            ok = _numeric_close(u, e)
            norm = nice(u)
        elif kind in ("fraction", "mixed_fraction"):
            u = parse_fraction(user_answer)
            e = Fraction(answer["value"])
            ok = u == e or abs(float(u - e)) <= 1e-4
            norm = nice(u)
        elif kind == "expression":
            expected = answer.get("sympy") or answer.get("plain")
            if metadata.get("validation") == "polynomial_division":
                ok, norm = _validate_polynomial_division(str(user_answer), metadata, expected)
            else:
                ok = algebraically_equivalent(user_answer, expected)
                norm = sp.sstr(canonical(user_answer)) if ok else sp.sstr(parse_math_expr(user_answer))
        elif kind == "symbol":
            aliases = {"mayor": ">", "mayorque": ">", "menor": "<", "menorque": "<", "igual": "=", "==": "="}
            u = str(user_answer).strip().lower().replace(" ", "")
            ok = aliases.get(u, u) == answer["plain"]
            norm = u
        else:
            u = re.sub(r"\s+", "", str(user_answer).lower())
            e = re.sub(r"\s+", "", str(answer["plain"]).lower())
            ok = u == e
            norm = u
    except Exception as exc:
        ok = False
        error_type = "parse_error"
        norm = None
        _ = exc
    return {
        "correcto": bool(ok),
        "correct": bool(ok),
        "questionId": qid,
        "respuestaCorrecta": answer["plain"],
        "correctAnswer": answer["plain"],
        "answer": answer,
        "normalizedUserAnswer": norm,
        "errorType": error_type,
    }


def _explain_from_payload(payload: dict[str, Any], user_answer: Any) -> dict[str, Any]:
    result = validate_payload(payload, user_answer)
    answer = payload.get("answer") or payload.get("expected") or {}
    steps: list[str] = []
    error_type = result.get("errorType")
    if result["correct"]:
        steps.append("Tu respuesta es equivalente a la forma simplificada esperada.")
        return {"correct": True, "errorType": None, "steps": steps}
    if answer.get("type") == "expression":
        try:
            user_expr = parse_math_expr(user_answer)
            expected_expr = parse_math_expr(answer.get("sympy") or answer.get("plain"))
            if algebraically_equivalent(-user_expr, expected_expr):
                error_type = "sign_error"
                steps.append("El procedimiento parece tener un cambio de signo global.")
            elif algebraically_equivalent(sp.expand(user_expr), sp.expand(expected_expr)):
                error_type = "not_simplified"
                steps.append("La idea algebraica es correcta, pero falta simplificar la forma final.")
            else:
                error_type = "algebra_mismatch"
                steps.append("Compara distribucion, signos y terminos semejantes paso por paso.")
            steps.append(f"Una forma esperada es: {answer.get('plain')}")
        except Exception:
            error_type = "parse_error"
            steps.append("No pude interpretar la respuesta como expresion matematica.")
    else:
        steps.append(f"Respuesta esperada: {answer.get('plain')}")
    return {"correct": False, "errorType": error_type, "steps": steps}


# ---------------------------------------------------------------------------
# UTILIDADES DE GENERADORES
# ---------------------------------------------------------------------------

SYMBOLS = {"x": x, "y": y, "z": z}


def calc(a: Fraction, op: str, b: Fraction) -> Fraction:
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]


def nonzero_int(r, low: int, high: int) -> int:
    value = 0
    while value == 0:
        value = r.randint(low, high)
    return value
    

def signed_coeff(r, hi: int, allow_negative: bool = True) -> int:
    low = -hi if allow_negative else 1
    return nonzero_int(r, low, hi)


def choose_variables(r, diff: str) -> tuple[sp.Symbol, ...]:
    lv = level(diff)
    if lv == 1:
        return (x,)
    if lv == 2:
        return tuple(r.sample((x, y), 2))
    return tuple(r.sample(VARIABLES, r.randint(2, 3)))


def expression_signature(expr: Any) -> str:
    text = sp.sstr(sp.factor(sp.expand(expr))) if isinstance(expr, sp.Expr) else str(expr)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def pattern_signature(
    topic: str,
    subtopic: str,
    diff: str,
    template: str,
    pattern: str,
    knobs: dict[str, Any],
    instance: dict[str, Any] | None = None,
) -> str:
    raw = json.dumps(
        {
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty_key(diff),
            "template": template,
            "pattern": pattern,
            "knobs": knobs,
            "instance": instance or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def draft_question(
    topic: str,
    subtopic: str,
    diff: str,
    prompt: str,
    value: Any,
    kind: str,
    *,
    plain: str | None = None,
    latex: str | None = None,
    answer_plain: str | None = None,
    answer_latex: str | None = None,
    instructions: str = "Responde con el resultado simplificado.",
    template: str = "general",
    pattern: str = "default",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(meta or {})
    meta.setdefault("template", template)
    meta.setdefault("pattern", pattern)
    meta.setdefault("difficultyLevel", level(diff))
    meta.setdefault("structureSignature", pattern_signature(topic, subtopic, diff, template, pattern, meta.get("knobs", {})))
    meta.setdefault(
        "antiRepeatSignature",
        pattern_signature(
            topic,
            subtopic,
            diff,
            template,
            pattern,
            meta.get("knobs", {}),
            {"plain": plain or prompt, "value": str(value)},
        ),
    )
    return {
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty_key(diff),
        "prompt": prompt,
        "instructions": instructions,
        "display": {"plain": plain or prompt, "latex": latex, "format": "latex" if latex else "text"},
        "value": value,
        "kind": kind,
        "answerPlain": answer_plain,
        "answerLatex": answer_latex,
        "metadata": meta,
    }


def monomial_from_exponents(coeff: int, variables: Iterable[sp.Symbol], exponents: Iterable[int]) -> sp.Expr:
    term = sp.Integer(coeff)
    for var, exp in zip(variables, exponents):
        if exp:
            term *= var ** exp
    return term


def random_exponent_tuple(r, var_count: int, max_degree: int, allow_constant: bool = False) -> tuple[int, ...]:
    while True:
        exps = tuple(r.randint(0, max_degree) for _ in range(var_count))
        total = sum(exps)
        if (allow_constant or total > 0) and total <= max_degree:
            return exps


def random_multivar_poly(
    r,
    variables: tuple[sp.Symbol, ...],
    *,
    terms: int,
    max_degree: int,
    coeff_hi: int,
    allow_negative: bool = True,
    require_constant_gap: bool = False,
) -> sp.Expr:
    seen: set[tuple[int, ...]] = set()
    expr = sp.Integer(0)
    tries = 0
    while len(seen) < terms and tries < terms * 40:
        tries += 1
        exps = random_exponent_tuple(r, len(variables), max_degree, allow_constant=True)
        if not any(exps) and not require_constant_gap and r.random() < 0.7:
            continue
        if exps in seen:
            continue
        seen.add(exps)
        expr += monomial_from_exponents(signed_coeff(r, coeff_hi, allow_negative), variables, exps)
    return sp.expand(expr)


def random_univar_poly(
    r,
    *,
    degree: int,
    terms: int,
    coeff_hi: int,
    allow_negative: bool = True,
    var: sp.Symbol = x,
) -> sp.Expr:
    degrees = sorted(r.sample(range(degree + 1), min(terms, degree + 1)), reverse=True)
    if degree not in degrees:
        degrees[0] = degree
    expr = sp.Integer(0)
    for deg in degrees:
        expr += signed_coeff(r, coeff_hi, allow_negative) * var ** deg
    return sp.expand(expr)


def shuffled_sum_latex(parts: list[sp.Expr], op: str) -> tuple[str, str]:
    plain = f" {op} ".join(f"({sp.sstr(p)})" for p in parts)
    latex = f" {op} ".join(f"\\left({sp.latex(p)}\\right)" for p in parts)
    return plain, latex


# ---------------------------------------------------------------------------
# GENERADOR: Agilidad mental
# ---------------------------------------------------------------------------

def _kind(value: Fraction) -> str:
    return "number" if value.denominator == 1 else "fraction"


def _final(topic: str, subtopic: str, diff: str, plain: str, latex: str, result: Fraction, template: str, pattern: str, complexity: int):
    p = profile(diff)
    seconds = mental_time(diff, complexity)
    return draft_question(
        topic,
        subtopic,
        diff,
        f"Calcula: {plain}",
        result,
        _kind(result),
        plain=plain,
        latex=latex,
        instructions="Responde el resultado exacto y simplificado.",
        template=template,
        pattern=pattern,
        meta={
            "interactive": True,
            "seconds": seconds,
            "basePoints": p.base_points,
            "scorePolicy": "time_remaining",
            "knobs": {"complexity": complexity, "seconds": seconds},
        },
    )


def _division_exacta(r, hi: int) -> tuple[int, int, Fraction]:
    divisor = r.randint(2, hi)
    quotient = r.randint(2, hi)
    return divisor * quotient, divisor, Fraction(quotient)


def generate_mental(topic: str, subtopic: str, diff: str, r):
    lv = profile(diff).level

    if subtopic == "calculo_rapido":
        if lv == 1:
            variant = r.choice(["suma_resta", "multiplicacion", "division_exacta"])
            if variant == "suma_resta":
                a, b = r.randint(10, 40), r.randint(5, 20)
                op = r.choice(["+", "-"])
                # Evitar resultados negativos en nivel fácil
                if op == "-" and b > a:
                    a, b = b, a
                result = Fraction(a + b if op == "+" else a - b)
                return _final(topic, subtopic, diff, f"{a} {op} {b}", f"{a} {op} {b}", result, "mental_directa", variant, 0)
            elif variant == "multiplicacion":
                a, b = r.randint(2, 10), r.randint(2, 10)
                result = Fraction(a * b)
                return _final(topic, subtopic, diff, f"{a} * {b}", f"{a}\\times {b}", result, "mental_directa", variant, 0)
            else:
                # Diseño Inverso: División siempre exacta
                dividend, divisor, quotient = _division_exacta(r, 12)
                return _final(topic, subtopic, diff, f"{dividend} / {divisor}", f"{dividend}\\div {divisor}", quotient, "mental_directa", variant, 0)
        
        elif lv == 2:
            variant = r.choice(["porcentaje_mental", "doble_mitad"])
            if variant == "porcentaje_mental":
                percent = r.choice([10, 20, 25, 50])
                amount = r.randint(2, 20) * 10
                result = Fraction(percent * amount, 100)
                return _final(topic, subtopic, diff, f"{percent}% de {amount}", f"{percent}\\% de {amount}", result, "mental_porcentaje", variant, 1)
            else:
                if r.random() < 0.5:
                    a = r.randint(15, 45)
                    return _final(topic, subtopic, diff, f"doble de {a}", f"2\\times {a}", Fraction(2 * a), "mental_atajos", variant, 1)
                else:
                    a = r.randint(10, 40) * 2
                    return _final(topic, subtopic, diff, f"mitad de {a}", f"{a}\\div 2", Fraction(a // 2), "mental_atajos", variant, 1)

        else:
            variant = r.choice(["cuadrado", "division_signos"])
            if variant == "cuadrado":
                a = r.randint(11, 15)
                return _final(topic, subtopic, diff, f"{a}^2", f"{a}^2", Fraction(a * a), "mental_potencias", variant, 2)
            else:
                dividend, divisor, quotient = _division_exacta(r, 15)
                sign = r.choice([-1, 1])
                result = Fraction(sign * quotient)
                plain = f"{sign * dividend} / {divisor}"
                return _final(topic, subtopic, diff, plain, f"{sign * dividend}\\div {divisor}", result, "mental_directa", variant, 2)


    elif subtopic == "operaciones_mixtas":
        if lv == 1:
            variant = "parentesis_suma"
            a, b, c = r.randint(10, 30), r.randint(5, 15), r.randint(5, 15)
            op1, op2 = r.choice(["+", "-"]), r.choice(["+", "-"])
            if op2 == "-" and c > b: b, c = c, b
            inner = b + c if op2 == "+" else b - c
            if op1 == "-" and inner > a: a, inner = inner, a
            result = Fraction(a + inner if op1 == "+" else a - inner)
            plain = f"{a} {op1} ({b} {op2} {c})"
            return _final(topic, subtopic, diff, plain, plain, result, "mental_dos_pasos", variant, 0)
        elif lv == 2:
            variant = "parentesis_producto"
            a, b, c = r.randint(5, 15), r.randint(2, 9), r.randint(2, 6)
            op = r.choice(["+", "-"])
            if op == "-" and b > a: a, b = b, a
            inner = a + b if op == "+" else a - b
            result = Fraction(inner * c)
            plain = f"({a} {op} {b}) * {c}"
            return _final(topic, subtopic, diff, plain, f"({a}{op}{b})\\times {c}", result, "mental_dos_pasos", variant, 1)
        else:
            variant = "division_mas_algo"
            dividend, divisor, quotient = _division_exacta(r, 12)
            c = r.randint(5, 25)
            op = r.choice(["+", "-"])
            result = Fraction(quotient + c if op == "+" else quotient - c)
            plain = f"{dividend} / {divisor} {op} {c}"
            return _final(topic, subtopic, diff, plain, f"{dividend}\\div {divisor}{op}{c}", result, "mental_cascada", variant, 2)


    elif subtopic == "jerarquia_operaciones":
        if lv == 1:
            variant = "suma_producto"
            a, b, c = r.randint(5, 20), r.randint(2, 6), r.randint(2, 6)
            op = r.choice(["+", "-"])
            # Cebo cognitivo asegurado
            if op == "-" and a < b * c:
                a = r.randint(b * c + 1, b * c + 15)
            result = Fraction(a + b * c if op == "+" else a - b * c)
            plain = f"{a} {op} {b} * {c}"
            return _final(topic, subtopic, diff, plain, f"{a}{op}{b}\\times {c}", result, "jerarquia_un_producto", variant, 0)
        elif lv == 2:
            variant = "dos_productos"
            a, b, c, d = r.randint(3, 8), r.randint(3, 8), r.randint(2, 6), r.randint(2, 6)
            op = r.choice(["+", "-"])
            p1, p2 = a * b, c * d
            if op == "-" and p2 > p1: a, b, c, d, p1, p2 = c, d, a, b, p2, p1
            result = Fraction(p1 + p2 if op == "+" else p1 - p2)
            plain = f"{a} * {b} {op} {c} * {d}"
            return _final(topic, subtopic, diff, plain, f"{a}\\times {b}{op}{c}\\times {d}", result, "jerarquia_mixta", variant, 1)
        else:
            variant = "producto_y_division"
            a, b = r.randint(4, 9), r.randint(3, 8)
            dividend, divisor, quotient = _division_exacta(r, 9)
            op = r.choice(["+", "-"])
            result = Fraction(a * b + quotient if op == "+" else a * b - quotient)
            plain = f"{a} * {b} {op} {dividend} / {divisor}"
            latex = f"{a}\\times {b}{op}{dividend}\\div {divisor}"
            return _final(topic, subtopic, diff, plain, latex, result, "jerarquia_fraccion", variant, 2)


    else: # desafio_4_terminos
        if lv == 1:
            variant = "cuatro_sumas"
            a, b, c, d = r.randint(10, 20), r.randint(5, 15), r.randint(5, 15), r.randint(1, 10)
            result = Fraction(a + b - c + d)
            plain = f"{a} + {b} - {c} + {d}"
            return _final(topic, subtopic, diff, plain, plain, result, "cuatro_terminos_lineal", variant, 1)
        elif lv == 2:
            variant = "parentesis_doble"
            a, b, c, d = r.randint(5, 15), r.randint(2, 9), r.randint(2, 5), r.randint(5, 20)
            result = Fraction((a + b) * c - d)
            plain = f"({a} + {b}) * {c} - {d}"
            return _final(topic, subtopic, diff, plain, f"({a}+{b})\\times {c}-{d}", result, "cuatro_terminos_productos", variant, 2)
        else:
            variant = "cuatro_mixto_exacto"
            a, b = r.randint(15, 35), r.randint(2, 9)
            c = r.randint(2, 9)
            dividend, divisor, quotient = _division_exacta(r, 8)
            sign = r.choice([-1, 1])
            op_frac = "+" if sign > 0 else "-"
            result = Fraction(a - (b * c) + sign * quotient)
            plain = f"{a} - {b} * {c} {op_frac} {dividend}/{divisor}"
            latex = f"{a} - {b}\\times {c} {op_frac} {dividend}\\div {divisor}"
            return _final(topic, subtopic, diff, plain, latex, result, "cuatro_terminos_avanzado", variant, 3)

# ---------------------------------------------------------------------------
# GENERADOR: Conversiones y decimales
# ---------------------------------------------------------------------------

def _decimal_text(fraction: Fraction) -> str:
    dec = Decimal(fraction.numerator) / Decimal(fraction.denominator)
    text = format(dec.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text

def _mixed_fraction_text(value: Fraction) -> str:
    sign = "-" if value < 0 else ""
    f = abs(value)
    whole = f.numerator // f.denominator
    rem = f.numerator % f.denominator
    return f"{sign}{whole} {rem}/{f.denominator}" if rem else f"{sign}{whole}"

def generate_conversions(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    # Denominadores que garantizan decimales exactos
    exact_den = [2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 100]
    # Denominadores que generan periódicos infinitos
    infinite_den = [3, 6, 7, 9, 11, 12, 14, 15, 18, 24]

    if subtopic == "decimal_a_fraccion":
        variant = r.choice(["directo", "medida", "dinero"] if lv < 3 else ["directo", "medida", "negativo", "con_entero"])
        # Diseño inverso: construimos una fracción amigable primero
        den = r.choice(exact_den[:5 + lv * 2])
        whole = r.randint(0, lv + 1) if variant == "con_entero" else 0
        num = r.randint(1, den - 1) + whole * den
        
        if variant == "dinero":
            den = 100
            num = r.randrange(25, 500, 25)
            
        f = Fraction(num, den)
        if variant == "negativo":
            f = -f
            
        dec = _decimal_text(f)
        prompts = {
            "directo": f"Convierte {dec} a fraccion simplificada.",
            "medida": f"Una medida es {dec} m. Escribela como fraccion simplificada.",
            "dinero": f"Escribe ${dec} como fraccion simplificada de unidad.",
            "negativo": f"Convierte el decimal negativo {dec} a fraccion simplificada.",
            "con_entero": f"Convierte {dec} a fraccion impropia simplificada.",
        }
        return draft_question(topic, subtopic, diff, prompts[variant], f, "fraction", template="decimal_a_fraccion", pattern=variant)

    if subtopic == "fraccion_a_decimal":
        variant = r.choice(["exacto", "medida", "redondeado"])
        if variant == "redondeado":
            # Forzamos un denominador infinito para evaluar el redondeo
            den = r.choice(infinite_den)
            num = r.randint(1, den * 2)
            f = Fraction(num, den)
            value = round_nice(f, 4)
            shown = nice(f)
            prompt = f"Convierte {shown} a decimal. Redondea a 4 decimales."
        else:
            # Forzamos un denominador de base 2 y 5 para exigir el decimal exacto
            den = r.choice(exact_den)
            num = r.randint(1, den * 2)
            f = Fraction(num, den)
            if lv == 3 and r.random() < 0.3: f = -f
            value = float(_exact_decimal(f))
            shown = nice(f)
            prompt = f"Convierte {shown} a decimal exacto." if variant == "exacto" else f"Una distancia es {shown} km. Escribela en decimal."
            
        return draft_question(topic, subtopic, diff, prompt, value, "number", template="fraccion_a_decimal", pattern=variant)

    if subtopic in ("mixto_a_decimal", "decimal_a_mixto", "mixto_a_fraccion"):
        # Lógica consolidada para números mixtos asegurando números lógicos
        den = r.choice(exact_den if subtopic != "mixto_a_fraccion" else exact_den + infinite_den)
        whole = r.randint(1, 8 + 2 * lv)
        num = r.randint(1, den - 1)
        f = Fraction(whole * den + num, den)
        if lv == 3 and r.random() < 0.3: f = -f
        
        dec = _decimal_text(f)
        mixed = _mixed_fraction_text(f)
        
        if subtopic == "mixto_a_decimal":
            prompt = f"Convierte {mixed} a decimal exacto."
            return draft_question(topic, subtopic, diff, prompt, float(_exact_decimal(f)), "number", template="mixto_decimal", pattern="directo")
        elif subtopic == "decimal_a_mixto":
            prompt = f"Convierte {dec} a numero mixto simplificado."
            return draft_question(topic, subtopic, diff, prompt, f, "mixed_fraction", answer_plain=mixed, template="decimal_mixto", pattern="directo")
        else: # mixto_a_fraccion
            prompt = f"Convierte {mixed} a fraccion impropia."
            return draft_question(topic, subtopic, diff, prompt, f, "fraction", template="mixto_impropia", pattern="directo")

    if subtopic == "comparacion":
        variant = r.choice(["fracciones", "fraccion_decimal", "negativas", "contra_mitad"])
        hi = 12 + 5 * lv
        
        if variant == "fraccion_decimal":
            a = Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            b = Fraction(r.randint(1, hi), 10 ** r.choice([1, 2]))
            left, right = nice(a), _decimal_text(b)
        elif variant == "negativas":
            a = -Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            b = -Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            left, right = nice(a), nice(b)
        elif variant == "contra_mitad":
            a = Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            b = Fraction(1, 2)
            left, right = nice(a), "1/2"
        else:
            a = Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            b = Fraction(r.randint(1, hi), r.randint(2, 8 + 3 * lv))
            left, right = nice(a), nice(b)
            
        while a == b:
            b += Fraction(1, r.randint(3, 11))
            right = nice(b) if variant != "fraccion_decimal" else _decimal_text(b)
            
        answer = ">" if a > b else "<"
        return draft_question(topic, subtopic, diff, f"Compara usando >, < o =: {left} __ {right}", answer, "symbol", template="comparacion_fracciones", pattern=variant)

    if subtopic == "redondeo":
        variant = r.choice(["decimal", "unidad", "decena"] if lv < 3 else ["decimal", "decimal_limite", "unidad"])
        if "decimal" in variant:
            places = {1: 1, 2: r.choice([2, 3]), 3: r.choice([3, 4])}[lv]
            # Evaluar explícitamente el caso límite (que termina en 5) que falla en round() nativo
            if variant == "decimal_limite" or r.random() < 0.3:
                raw_str = f"{r.randint(10, 99)}.{r.randint(10**(places-1), 10**places - 1)}5"
                raw = Decimal(raw_str)
            else:
                raw = Decimal(r.randint(100, 999999)) / Decimal(10 ** (places + r.choice([1, 2])))
                
            value = round_nice(raw, places)
            return draft_question(topic, subtopic, diff, f"Redondea {raw} a {places} decimales.", value, "number", template="redondeo", pattern=variant)
        
        raw_int = r.randint(25, 9999 if lv == 3 else 999)
        if variant == "decena":
            value = round_nice(Fraction(raw_int, 10), 0) * 10
            return draft_question(topic, subtopic, diff, f"Redondea {raw_int} a la decena mas cercana.", value, "number", template="redondeo", pattern=variant)
        else: # unidad
            raw = Decimal(r.randint(100, 9999)) / Decimal(10)
            value = round_nice(raw, 0)
            return draft_question(topic, subtopic, diff, f"Redondea {raw} a la unidad mas cercana.", value, "number", template="redondeo", pattern=variant)

    if subtopic in ("periodico_puro", "periodico_mixto"):
        max_rep_len = 1 if lv == 1 else 2 if lv == 2 else 3
        rep_digits = r.randint(1, 10**max_rep_len - 1)
        rep = str(rep_digits)
        
        whole = r.randint(1, 4 + lv) if r.random() < 0.5 else 0
        sign = -1 if lv == 3 and r.random() < 0.3 else 1
        
        if subtopic == "periodico_puro":
            frac_part = Fraction(int(rep), 10 ** len(rep) - 1)
            txt = f"{whole}.{rep}{rep}..." if whole else f"0.{rep}{rep}..."
            latex = f"{whole}.\\overline{{{rep}}}" if whole else f"0.\\overline{{{rep}}}"
        else:
            fixed = str(r.randint(1, 9 if lv == 1 else 99))
            frac_part = Fraction(int(fixed + rep) - int(fixed), (10 ** len(fixed)) * (10 ** len(rep) - 1))
            txt = f"{whole}.{fixed}{rep}{rep}..." if whole else f"0.{fixed}{rep}{rep}..."
            latex = f"{whole}.{fixed}\\overline{{{rep}}}" if whole else f"0.{fixed}\\overline{{{rep}}}"
            
        f = sign * (Fraction(whole, 1) + frac_part)
        if sign < 0:
            txt = f"-{txt}"
            latex = f"-{latex}"
            
        return draft_question(topic, subtopic, diff, f"Convierte {txt} a fraccion.", f, "fraction", latex=latex, template="periodico", pattern=subtopic)

    if subtopic == "operacion":
        # Cruce de decimales exactos con fracciones para devolver un resultado algebraico riguroso
        variant = r.choice(["suma", "resta", "multiplicacion"])
        f1 = Fraction(r.randint(1, 100), 10 ** r.choice([1, 2]))  # Decimal con 1 o 2 posiciones
        f2 = Fraction(r.randint(1, 25), r.choice(exact_den[:6]))  # Fracción amigable
        
        if lv == 3 and r.random() < 0.3: f1 = -f1
        
        op_map = {"suma": "+", "resta": "-", "multiplicacion": "*"}
        op = op_map[variant]
        
        result = calc(f1, op, f2)
        plain = f"{_decimal_text(f1)} {op} {nice(f2)}"
        
        return draft_question(topic, subtopic, diff, f"Calcula y responde con fraccion simplificada: {plain}", result, "fraction", plain=plain, template="fraccion_decimal_operacion", pattern=variant)
# ---------------------------------------------------------------------------
# GENERADOR: Expresiones algebraicas
# ---------------------------------------------------------------------------

def _poly_diff(r, diff: str, *, min_terms: int | None = None) -> tuple[tuple[sp.Symbol, ...], sp.Expr]:
    lv = level(diff)
    if lv == 1:
        variables = (x,)
        poly = random_univar_poly(r, degree=r.randint(2, 4), terms=min_terms or r.randint(2, 3), coeff_hi=7, allow_negative=False)
    elif lv == 2:
        variables = tuple(r.sample((x, y), 2))
        poly = random_multivar_poly(r, variables, terms=min_terms or r.randint(3, 5), max_degree=r.randint(2, 4), coeff_hi=9, allow_negative=True, require_constant_gap=True)
    else:
        variables = choose_variables(r, diff)
        poly = random_multivar_poly(r, variables, terms=min_terms or r.randint(4, 6), max_degree=r.randint(3, 6), coeff_hi=12, allow_negative=True, require_constant_gap=True)
    return variables, sp.expand(poly)


def _degree_question(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    variables, poly = _poly_diff(r, diff)
    poly_obj = sp.Poly(poly, *variables)
    total = int(poly_obj.total_degree())
    relative = {str(var): int(poly_obj.degree(var)) for var in variables}
    terms_by_degree = sorted(poly_obj.terms(), key=lambda item: sum(item[0]), reverse=True)
    max_degree = sum(terms_by_degree[0][0])
    leading_coeffs = [c for monom, c in terms_by_degree if sum(monom) == max_degree]
    unique_leading_term = len(leading_coeffs) == 1

    variants = ["grado_total", "grado_termino_lider"]
    if unique_leading_term:
        variants.append("coeficiente_lider")
    if lv >= 2:
        variants.append("grado_relativo")
    variant = r.choice(variants)
    if variant == "grado_relativo":
        var = r.choice(variables)
        prompt = f"Indica el grado relativo respecto a {var}."
        answer = relative[str(var)]
        pattern = f"grado_relativo_{var}"
    elif variant == "coeficiente_lider":
        prompt = "Indica el coeficiente del termino de mayor grado."
        answer = int(leading_coeffs[0])
        pattern = "coeficiente_lider"
    elif variant == "grado_termino_lider":
        prompt = "Indica el grado del termino de mayor grado."
        answer = total
        pattern = "grado_termino_lider"
    else:
        prompt = "Indica el grado total del polinomio."
        answer = total
        pattern = "grado_total"

    return draft_question(
        topic,
        subtopic,
        diff,
        prompt,
        answer,
        "number",
        plain=f"{prompt} {sp.sstr(poly)}",
        latex=sp.latex(poly),
        instructions="Responde solo el numero solicitado.",
        template="grado_polinomio",
        pattern=f"{pattern}_lv{lv}",
        meta={"degree": {"total": total, "relative": relative}, "variables": [str(v) for v in variables]},
    )


def _sum_rest(topic: str, subtopic: str, diff: str, r, sign: int):
    lv = level(diff)
    variables, p = _poly_diff(r, diff)
    _, q = _poly_diff(r, diff, min_terms=2 + lv)
    _, h = _poly_diff(r, diff, min_terms=2 + (lv > 1))
    op = "+" if sign > 0 else "-"
    variant = r.choice(["dos_polinomios", "tres_polinomios", "parentesis_anidado"])

    if variant == "tres_polinomios":
        if sign > 0:
            result = sp.expand(p + q + h)
            plain, latex = shuffled_sum_latex([p, q, h], "+")
        else:
            result = sp.expand(p - q - h)
            plain = f"({sp.sstr(p)}) - ({sp.sstr(q)}) - ({sp.sstr(h)})"
            latex = f"\\left({sp.latex(p)}\\right)-\\left({sp.latex(q)}\\right)-\\left({sp.latex(h)}\\right)"
    elif variant == "parentesis_anidado":
        result = sp.expand(p + sign * (q - h))
        plain = f"({sp.sstr(p)}) {op} (({sp.sstr(q)}) - ({sp.sstr(h)}))"
        latex = f"\\left({sp.latex(p)}\\right) {op} \\left(\\left({sp.latex(q)}\\right)-\\left({sp.latex(h)}\\right)\\right)"
    else:
        result = sp.expand(p + sign * q)
        plain, latex = shuffled_sum_latex([p, q], op)

    return draft_question(
        topic,
        subtopic,
        diff,
        "Simplifica la operacion de polinomios.",
        result,
        "expression",
        plain=plain,
        latex=latex,
        instructions="Combina terminos semejantes y ordena la expresion.",
        template="suma_resta_polinomios",
        pattern=f"{variant}_{op}_vars{len(variables)}_lv{lv}",
        meta={"variables": [str(v) for v in variables], "knobs": {"terms": 3 + lv, "op": op, "variant": variant}},
    )


def _one_term_poly(r, variables: tuple[sp.Symbol, ...], coeff_hi: int, max_degree: int, allow_negative: bool) -> sp.Expr:
    return random_multivar_poly(r, variables, terms=1, max_degree=max_degree, coeff_hi=coeff_hi, allow_negative=allow_negative)


def _multiplication(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    if lv == 1:
        variables = (x,)
        variant = r.choice(["monomio_binomio", "binomio_por_monomio", "factor_comun"])
        if variant == "binomio_por_monomio":
            p = random_univar_poly(r, degree=1, terms=2, coeff_hi=7, allow_negative=False)
            q = signed_coeff(r, 6, False) * x ** r.randint(1, 2)
        elif variant == "factor_comun":
            a, b, c = r.randint(2, 7), r.randint(2, 7), r.randint(1, 8)
            p = a * x
            q = b * x + c
        else:
            p = _one_term_poly(r, variables, 6, 3, False)
            q = random_univar_poly(r, degree=1, terms=2, coeff_hi=8, allow_negative=False)
    elif lv == 2:
        variables = tuple(r.sample((x, y), 1 if r.random() < 0.6 else 2))
        variant = r.choice(["binomio_binomio", "cuadrado_binomio", "diferencia_cuadrados"])
        var = variables[0]
        if variant == "cuadrado_binomio":
            a, b = signed_coeff(r, 6), signed_coeff(r, 8)
            p = a * var + b
            q = p
        elif variant == "diferencia_cuadrados":
            a, b = signed_coeff(r, 5), signed_coeff(r, 8)
            p = a * var + b
            q = a * var - b
        else:
            p = random_multivar_poly(r, variables, terms=2, max_degree=2, coeff_hi=7, allow_negative=True)
            q = random_multivar_poly(r, variables, terms=2, max_degree=2, coeff_hi=7, allow_negative=True)
    else:
        variables = tuple(r.sample(VARIABLES, r.randint(2, 3)))
        variant = r.choice(["trinomios_multivariable", "cuadrado_binomio_multi", "diferencia_cuadrados_multi"])
        a, b = variables[0], variables[1]
        if variant == "cuadrado_binomio_multi":
            p = signed_coeff(r, 5) * a + signed_coeff(r, 5) * b
            q = p
        elif variant == "diferencia_cuadrados_multi":
            p = signed_coeff(r, 5) * a + signed_coeff(r, 5) * b
            q = sp.expand(p.subs(b, -b))
        else:
            p = random_multivar_poly(r, variables, terms=3, max_degree=3, coeff_hi=8, allow_negative=True)
            q = random_multivar_poly(r, variables, terms=r.randint(2, 3), max_degree=3, coeff_hi=8, allow_negative=True)
    result = sp.expand(p * q)
    plain = f"({sp.sstr(p)}) * ({sp.sstr(q)})"
    latex = f"\\left({sp.latex(p)}\\right)\\left({sp.latex(q)}\\right)"
    return draft_question(
        topic,
        subtopic,
        diff,
        "Multiplica y simplifica.",
        result,
        "expression",
        plain=plain,
        latex=latex,
        instructions="Distribuye todos los terminos y combina semejantes.",
        template="multiplicacion_polinomios",
        pattern=variant,
        meta={"variables": [str(v) for v in variables], "knobs": {"leftTerms": len(sp.Poly(p, *variables).terms()), "rightTerms": len(sp.Poly(q, *variables).terms())}},
    )


def _division(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    if lv == 1:
        variant = r.choice(["lineal_exacta", "monomio_exacta", "factor_comun"])
        if variant == "monomio_exacta":
            divisor = r.randint(2, 6) * x
            quotient = random_univar_poly(r, degree=r.randint(1, 2), terms=2, coeff_hi=5, allow_negative=False)
            remainder = sp.Integer(0)
        elif variant == "factor_comun":
            a = r.randint(2, 6)
            divisor = a * x
            quotient = r.randint(2, 6) * x + r.randint(1, 8)
            remainder = sp.Integer(0)
        else:
            a = r.randint(1, 6)
            divisor = x - a
            quotient = random_univar_poly(r, degree=r.randint(1, 2), terms=2, coeff_hi=6, allow_negative=False)
            remainder = sp.Integer(0)
    elif lv == 2:
        variant = r.choice(["monomio_exacta", "sintetica_exacta", "sintetica_con_residuo"])
        if variant == "monomio_exacta":
            divisor = signed_coeff(r, 7) * x ** r.randint(1, 2)
            quotient = random_univar_poly(r, degree=r.randint(2, 3), terms=3, coeff_hi=8, allow_negative=True)
            remainder = sp.Integer(0)
        else:
            a = r.randint(-5, 5) or 2
            divisor = x - a
            quotient = random_univar_poly(r, degree=r.randint(2, 3), terms=3, coeff_hi=8, allow_negative=True)
            remainder = sp.Integer(r.randint(-6, 6)) if variant == "sintetica_con_residuo" else sp.Integer(0)
            if variant == "sintetica_con_residuo" and remainder == 0:
                remainder = sp.Integer(3)
    else:
        variant = r.choice(["lineal_larga_residuo", "cuadratica_larga_residuo", "lineal_exacta_densa"])
        if variant == "cuadratica_larga_residuo":
            a, b = r.randint(-4, 4) or 1, r.randint(-5, 5) or 2
            divisor = sp.expand((x - a) * (x - b))
        else:
            a = r.randint(-7, 7) or 2
            divisor = x - a
        quotient = random_univar_poly(r, degree=r.randint(3, 4), terms=r.randint(3, 5), coeff_hi=10, allow_negative=True)
        if variant == "lineal_exacta_densa":
            remainder = sp.Integer(0)
        else:
            rem_degree = max(0, sp.degree(divisor, x) - 1)
            remainder = random_univar_poly(r, degree=rem_degree, terms=1 if rem_degree == 0 else 2, coeff_hi=7, allow_negative=True)
    dividend = sp.expand(divisor * quotient + remainder)
    combined = sp.simplify(sp.apart(sp.together(quotient + remainder / divisor), x))
    plain = f"({sp.sstr(dividend)}) / ({sp.sstr(divisor)})"
    latex = f"\\dfrac{{{sp.latex(dividend)}}}{{{sp.latex(divisor)}}}"
    return draft_question(
        topic,
        subtopic,
        diff,
        "Divide el polinomio.",
        combined,
        "expression",
        plain=plain,
        latex=latex,
        instructions="Puedes responder como expresion simplificada o como cociente y residuo: q=..., r=....",
        template="division_polinomios",
        pattern=variant,
        meta={
            "validation": "polynomial_division",
            "method": "sintetica" if sp.degree(divisor, x) == 1 else "larga",
            "dividend": sp.sstr(dividend),
            "divisor": sp.sstr(divisor),
            "quotient": sp.sstr(sp.expand(quotient)),
            "remainder": sp.sstr(sp.expand(remainder)),
            "knobs": {"divisorDegree": int(sp.degree(divisor, x)), "hasRemainder": bool(remainder != 0), "variant": variant},
        },
    )


def _algebraic_simplification(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    if lv == 1:
        # Fácil (Reconocimiento directo): a*x*(b*x +/- c)
        variant = "monomio_por_binomio"
        a, b, c = r.randint(2, 9), r.randint(2, 9), r.randint(1, 9)
        op = r.choice(["+", "-"])
        expr = sp.expand(a * x * (b * x + (c if op == "+" else -c)))
        plain = f"{a}*x * ({b}*x {op} {c})"
        latex = f"{a}x \\left({b}x {op} {c}\\right)"
        
    elif lv == 2:
        # Normal (Patrones notables): (x^2 - (r1+r2)x + r1*r2) / (x - r1) = x - r2
        variant = "division_exacta_notable"
        r1, r2 = r.randint(1, 8), r.randint(1, 8)
        num = x**2 - (r1 + r2)*x + (r1 * r2)
        den = x - r1
        expr = x - r2
        plain = f"(x**2 - {r1+r2}*x + {r1*r2}) / (x - {r1})"
        latex = f"\\dfrac{{x^2 - {r1+r2}x + {r1*r2}}}{{x - {r1}}}"
        
    else:
        # Avanzado (Cancelación multinivel): ((a^2 x^2 - b^2 y^2) / (ax + by)) - ax -> -by
        variant = "cancelacion_multinivel"
        a, b = r.randint(2, 6), r.randint(2, 6)
        num = (a**2) * x**2 - (b**2) * y**2
        den = a * x + b * y
        expr = sp.simplify((num / den) - a * x)
        plain = f"(({a**2}*x**2 - {b**2}*y**2) / ({a}*x + {b}*y)) - {a}*x"
        latex = f"\\dfrac{{{a**2}x^2 - {b**2}y^2}}{{{a}x + {b}y}} - {a}x"
        
    return draft_question(
        topic, subtopic, diff, "Simplifica la expresion algebraica.", expr, "expression",
        plain=plain, latex=latex,
        instructions="Reduce usando propiedades algebraicas y productos notables.",
        template="simplificacion_algebraica", pattern=variant,
        meta={"knobs": {"level": lv, "variant": variant}},
    )


def generate_algebra(topic: str, subtopic: str, diff: str, r):
    if subtopic == "grado_polinomio":
        return _degree_question(topic, subtopic, diff, r)
    if subtopic == "suma_polinomios":
        return _sum_rest(topic, subtopic, diff, r, 1)
    if subtopic == "resta_polinomios":
        return _sum_rest(topic, subtopic, diff, r, -1)
    if subtopic == "multiplicacion_polinomios":
        return _multiplication(topic, subtopic, diff, r)
    if subtopic == "division_polinomios":
        return _division(topic, subtopic, diff, r)
    return _algebraic_simplification(topic, subtopic, diff, r)

# ---------------------------------------------------------------------------
# GENERADOR: Razones y porcentajes
# ---------------------------------------------------------------------------

def generate_percentages(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    hi = {1: 60, 2: 180, 3: 420}[lv]

    if subtopic == "proporcion_inversa":
        variant = r.choice(["simbolica_simple", "obreros_dias", "maquinas_horas"] if lv < 3 else ["producto_compuesto", "obreros_dias", "maquinas_horas", "rendimiento"])
        if variant == "obreros_dias":
            workers = r.randint(3, 12 + lv * 4)
            days = r.randint(2, 12 + lv * 4)
            new_workers = r.randint(3, 16 + lv * 6)
            value = Fraction(workers * days, new_workers)
            text = f"{workers} obreros terminan una tarea en {days} dias. Cuantos dias tardan {new_workers} obreros al mismo ritmo?"
        elif variant == "maquinas_horas":
            machines = r.randint(2, 10 + lv * 3)
            hours = r.randint(3, 18 + lv * 5)
            new_machines = r.randint(2, 14 + lv * 4)
            value = Fraction(machines * hours, new_machines)
            text = f"{machines} maquinas hacen un trabajo en {hours} horas. Cuantas horas necesitan {new_machines} maquinas?"
        elif variant == "rendimiento":
            a, b, c, d = r.randint(2, hi), r.randint(2, hi), r.randint(2, hi), r.randint(2, 18)
            value = Fraction(a * b * d, c)
            text = f"Despeja x: ({a} * {b} * {d}) / {c} = x"
        else:
            a, b, c = r.randint(2, hi), r.randint(2, hi), r.randint(2, hi)
            value = Fraction(a * b, c)
            text = f"Despeja x: {a} * {b} = {c} * x"
        return draft_question(topic, subtopic, diff, text, round_nice(value), "number", template="proporcion", pattern=variant)

    if subtopic == "proporcion_directa":
        variant = r.choice(["simbolica", "precio_unitario", "escala"] if lv < 3 else ["simbolica", "precio_unitario", "escala", "tasa_compuesta"])
        if variant == "precio_unitario":
            units = r.randint(2, 12 + lv * 5)
            price = r.randint(10, hi)
            target = r.randint(2, 18 + lv * 7)
            value = Fraction(price * target, units)
            text = f"Si {units} unidades cuestan {price}, cuanto cuestan {target} unidades?"
        elif variant == "escala":
            model = r.randint(2, 25 + lv * 10)
            real = model * r.randint(2, 12 + lv * 4)
            target = r.randint(2, 25 + lv * 10)
            value = Fraction(real * target, model)
            text = f"En una escala, {model} cm representan {real} cm reales. Cuanto representan {target} cm?"
        elif variant == "tasa_compuesta":
            a, b, c, d = [r.randint(2, 25 + lv * 12) for _ in range(4)]
            value = Fraction(b * c * d, a)
            text = f"Despeja x: {a}/{b} = ({c} * {d})/x"
        else:
            a, b, c = r.randint(2, hi), r.randint(2, hi), r.randint(2, hi)
            if lv == 1:
                c = r.randint(2, 20)
                a = r.randint(2, 10)
                b = a * r.randint(2, 10)
            value = Fraction(b * c, a)
            text = f"Despeja x: {a}/{b} = {c}/x"
        return draft_question(topic, subtopic, diff, text, round_nice(value), "number", template="proporcion", pattern=variant)

    if subtopic == "regla_3_compuesta":
        variant = r.choice(["directa_contexto", "directa_simbolica", "precio_compuesto"] if lv == 1 else ["compuesta_inversa", "compuesta_contexto", "simbolica_equivalente"] if lv == 2 else ["compuesta_inversa", "compuesta_contexto", "despeje_compuesto"])
        if variant == "directa_contexto":
            a, b, c = r.randint(2, 12), r.randint(2, 12), r.randint(2, 12)
            text = f"Si {a} unidades cuestan {b}, cuanto cuestan {c} unidades?"
            value = Fraction(b * c, a)
        elif variant == "directa_simbolica":
            a, b, c = r.randint(2, 12), r.randint(2, 12), r.randint(2, 12)
            text = f"Despeja x: {a}/{b} = {c}/x"
            value = Fraction(b * c, a)
        elif variant == "precio_compuesto":
            packs, price, target = r.randint(2, 8), r.randint(10, 80), r.randint(3, 16)
            text = f"{packs} paquetes cuestan {price}. Cuanto cuestan {target} paquetes?"
            value = Fraction(price * target, packs)
        elif variant == "compuesta_contexto":
            people = r.randint(2, 12 + lv * 4)
            days = r.randint(2, 12 + lv * 4)
            hours = r.randint(2, 8 + lv * 2)
            new_people = r.randint(2, 14 + lv * 5)
            new_hours = r.randint(2, 9 + lv * 2)
            value = Fraction(people * days * hours, new_people * new_hours)
            text = f"{people} personas trabajando {hours} h/dia terminan en {days} dias. Cuantos dias tardan {new_people} personas a {new_hours} h/dia?"
        elif variant == "despeje_compuesto":
            a, b, c, d, e, f = [r.randint(2, 35) for _ in range(6)]
            text = f"Despeja x: ({a} * {b})/({c} * x) = ({d} * {e})/{f}"
            value = Fraction(a * b * f, c * d * e)
        else:
            a, b, c, d, e = [r.randint(2, 18 + lv * 15) for _ in range(5)]
            text = f"Despeja x: ({a} * {b})/{c} = ({d} * {e})/x"
            value = Fraction(d * e * c, a * b)
        return draft_question(topic, subtopic, diff, text, round_nice(value), "number", template="regla_3", pattern=variant)

    if subtopic == "descuento_doble":
        # Uso de Fraction para evitar errores de flotantes sucesivos
        if lv == 1:
            price = r.randrange(100, 1000, 50)
            d1 = r.choice([10, 20, 25, 50])
            text = f"Un producto cuesta ${price}. Aplica descuento de {d1}%. Valor final?"
            value = price * (1 - Fraction(d1, 100))
            variant = "descuento_simple"
        elif lv == 2:
            price = r.randrange(200, 2000, 100)
            d1 = r.choice([10, 20, 25])
            tax = r.choice([5, 10, 15])
            text = f"Un producto cuesta ${price}. Primero baja {d1}% y luego sube {tax}% de impuesto. Valor final?"
            value = price * (1 - Fraction(d1, 100)) * (1 + Fraction(tax, 100))
            variant = "descuento_impuesto"
        else:
            # Diseño Inverso para calcular el original a partir de un valor final limpio
            original = r.choice([200, 400, 500, 800, 1000])
            d1, d2 = r.choice([10, 20]), r.choice([10, 25])
            final = original * (1 - Fraction(d1, 100)) * (1 - Fraction(d2, 100))
            text = f"Despues de aplicar descuentos sucesivos de {d1}% y {d2}%, el precio final es ${round_nice(final, 2)}. Cual era el original?"
            value = original
            variant = "descuento_inverso"
        return draft_question(topic, subtopic, diff, text, round_nice(value, 2), "number", template="porcentaje_compuesto", pattern=variant)

    if subtopic == "que_porcentaje":
        # Uso de bases amigables para cálculo limpio
        nice_bases = [10, 20, 25, 40, 50, 100]
        if lv == 1:
            total = r.choice(nice_bases)
            part = r.randint(1, total)
            text = f"Que porcentaje es {part} de {total}?"
            value = Fraction(part, total) * 100
            variant = "base_amigable"
        elif lv == 2:
            total = r.choice(nice_bases) * r.choice([2, 3, 5])
            part = r.randint(1, total)
            text = f"De {total} estudiantes, {part} aprobaron. Que porcentaje aprobo?"
            value = Fraction(part, total) * 100
            variant = "contexto_amigable"
        else:
            old = r.choice(nice_bases) * r.choice([2, 5])
            new = old + r.randint(1, old // 2)
            text = f"De {old} a {new}, cual es el porcentaje de aumento?"
            value = Fraction(new - old, old) * 100
            variant = "cambio_relativo"
        return draft_question(topic, subtopic, diff, text, round_nice(value, 2), "percent", template="que_porcentaje", pattern=variant)

    if subtopic == "cuanto_es_porcentaje_de":
        variant = r.choice(["directo", "descuento_contexto", "grupo"] if lv < 3 else ["directo", "descuento_contexto", "grupo", "aumento_contexto"])
        percent = r.choice([5, 10, 15, 20, 25, 50, 75]) if lv == 1 else r.randint(1, 99)
        amount = r.randrange(20, 300, 10) if lv == 1 else r.randrange(50, 1000 + lv * 2500, 10)
        if variant == "descuento_contexto":
            text = f"Un descuento es de {percent}% sobre ${amount}. Cuanto dinero representa?"
        elif variant == "grupo":
            text = f"En un grupo de {amount} personas, el {percent}% participa. Cuantas personas son?"
        elif variant == "aumento_contexto":
            text = f"Un valor de {amount} aumenta {percent}%. Cuanto vale solo el aumento?"
        else:
            text = f"Calcula el {percent}% de {amount}."
        return draft_question(topic, subtopic, diff, text, round_nice(Fraction(percent, 100) * amount, 2), "number", template="porcentaje_de", pattern=variant)

    variant = r.choice(["doble_directo", "contexto", "fraccion_porcentual"] if lv < 3 else ["triple", "contexto_triple", "doble_directo"])
    a, b = r.randint(1, 99), r.randint(1, 99)
    if variant == "triple":
        c = r.randint(1, 99)
        text = f"Calcula el {a}% del {b}% del {c}%."
        value = Fraction(a * b * c, 10000)
    elif variant == "contexto_triple":
        c = r.randint(1, 99)
        text = f"De un grupo queda el {a}%, luego el {b}% de ese grupo y despues el {c}%. Que porcentaje del original queda?"
        value = Fraction(a * b * c, 10000)
    elif variant == "contexto":
        text = f"De un grupo queda el {a}%, y de ese grupo se toma el {b}%. Que porcentaje del original es?"
        value = Fraction(a * b, 100)
    elif variant == "fraccion_porcentual":
        a = r.choice([25, 50, 75])
        b = r.choice([20, 40, 60, 80])
        text = f"Calcula mentalmente el {a}% de {b}%."
        value = Fraction(a * b, 100)
    else:
        text = f"Calcula el {a}% de {b}%."
        value = Fraction(a * b, 100)
    return draft_question(topic, subtopic, diff, text, round_nice(value, 4 if lv == 3 else 2), "percent", template="porcentaje_de_porcentaje", pattern=variant)

# ---------------------------------------------------------------------------
# GENERADOR: Simplificacion
# ---------------------------------------------------------------------------

def _fraction_kind(value):
    if isinstance(value, Fraction):
        return "number" if value.denominator == 1 else "fraction"
    expr = sp.simplify(value)
    if expr.is_Rational:
        return "fraction" if not expr.is_Integer else "number"
    return "expression"


def _num_frac(r, hi=18):
    return Fraction(r.randint(1, hi), r.randint(2, hi))


def _sp_fraction(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def _has_variables(expr) -> bool:
    try:
        return bool(set(sp.sympify(expr).free_symbols) & set(VARIABLES))
    except Exception:
        return False


def _has_radical(expr) -> bool:
    try:
        e = sp.sympify(expr)
        return any(isinstance(p, sp.Pow) and p.exp.is_Rational and not p.exp.is_Integer for p in e.atoms(sp.Pow))
    except Exception:
        return False


def _is_integer_result(expr) -> bool:
    if isinstance(expr, Fraction):
        return expr.denominator == 1
    try:
        e = sp.simplify(expr)
        return bool(e.is_Rational and e.is_Integer)
    except Exception:
        return False


def _display_has_variable(question: dict[str, Any]) -> bool:
    display = question.get("display") or {}
    text = f"{display.get('plain', '')} {display.get('latex', '')}"
    return any(name in text for name in ("x", "y", "z"))


def _question_too_simple(question: dict[str, Any]) -> bool:
    meta = question.get("metadata") or {}
    template = meta.get("template")
    pattern = meta.get("pattern")
    value = question.get("value")
    allowed_simple = {"exp_cero", "exp_uno"}
    if pattern in allowed_simple:
        return False
    if template == "simplificacion_fracciones" and not _has_variables(value):
        return _is_integer_result(value)
    if template == "potencias":
        e = sp.simplify(value)
        if e in (0, 1):
            return True
        return not _has_variables(e) and len(str(e)) <= 2
    if template == "radicales" and not _has_variables(value):
        return bool(sp.simplify(value).is_number and not _has_radical(value))
    if level(question.get("difficulty")) == 3 and not _display_has_variable(question):
        return True
    return False


def _latex_expr(expr) -> str:
    return sp.latex(expr)


def _fraction_question(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    op = {
        "fracciones_suma": "+",
        "fracciones_resta": "-",
        "fracciones_multiplicacion": "*",
        "fracciones_division": "/",
    }[subtopic]
    sym = {"*": "\\times", "/": "\\div", "+": "+", "-": "-"}[op]
    disp = {"*": "*", "/": "/", "+": "+", "-": "-"}[op]

    while True:
        try:
            if lv == 1:
                variant = r.choice(["numerica_individual", "mixta_con_entero", "denominador_controlado"])
                hi = 11
                if variant == "mixta_con_entero":
                    whole = r.randint(1, 5)
                    f1, f2 = _num_frac(r, hi), _num_frac(r, hi)
                    left = Fraction(whole, 1) + f1
                    result = calc(left, op, f2)
                    plain = f"({whole} + {nice(f1)}) {disp} {nice(f2)}"
                    latex = f"\\left({whole}+{sp.latex(_sp_fraction(f1))}\\right) {sym} {sp.latex(_sp_fraction(f2))}"
                elif variant == "denominador_controlado":
                    if op in ("+", "-"):
                        den = r.randint(3, 14)
                        a, b = r.sample(range(1, den * 2), 2)
                        f1, f2 = Fraction(a, den), Fraction(b, den)
                    elif op == "*":
                        a, b = r.randint(2, 12), r.randint(2, 12)
                        f1, f2 = Fraction(a, b), Fraction(b, r.randint(3, 14))
                    else:
                        a, b, c = r.randint(2, 12), r.randint(2, 12), r.randint(3, 14)
                        f1, f2 = Fraction(a, b), Fraction(c, b)
                    result = calc(f1, op, f2)
                    plain = f"{nice(f1)} {disp} {nice(f2)}"
                    latex = f"{sp.latex(_sp_fraction(f1))} {sym} {sp.latex(_sp_fraction(f2))}"
                else:
                    f1, f2 = _num_frac(r, hi), _num_frac(r, hi)
                    result = calc(f1, op, f2)
                    plain = f"{nice(f1)} {disp} {nice(f2)}"
                    latex = f"{sp.latex(_sp_fraction(f1))} {sym} {sp.latex(_sp_fraction(f2))}"
                kind = "fraction"
                pattern = f"{variant}_{op}"

            elif lv == 2:
                variant = r.choice(["tres_terminos", "parentesis_misma_operacion", "mixto_y_fraccion"])
                hi = 20
                f1, f2, f3 = _num_frac(r, hi), _num_frac(r, hi), _num_frac(r, hi)
                if variant == "mixto_y_fraccion":
                    f1 = Fraction(r.randint(1, 5), 1) + f1
                if op == "+":
                    result = f1 + f2 + f3
                    plain = f"{nice(f1)} + {nice(f2)} + {nice(f3)}"
                    latex = f"{sp.latex(_sp_fraction(f1))}+{sp.latex(_sp_fraction(f2))}+{sp.latex(_sp_fraction(f3))}"
                elif op == "-":
                    result = f1 - f2 - f3
                    plain = f"{nice(f1)} - {nice(f2)} - {nice(f3)}"
                    latex = f"{sp.latex(_sp_fraction(f1))}-{sp.latex(_sp_fraction(f2))}-{sp.latex(_sp_fraction(f3))}"
                elif op == "*":
                    result = f1 * f2 * f3
                    plain = f"{nice(f1)} * {nice(f2)} * {nice(f3)}"
                    latex = f"{sp.latex(_sp_fraction(f1))}\\times {sp.latex(_sp_fraction(f2))}\\times {sp.latex(_sp_fraction(f3))}"
                else:
                    result = f1 / f2 / f3
                    plain = f"{nice(f1)} / {nice(f2)} / {nice(f3)}"
                    latex = f"{sp.latex(_sp_fraction(f1))}\\div {sp.latex(_sp_fraction(f2))}\\div {sp.latex(_sp_fraction(f3))}"
                if variant == "parentesis_misma_operacion":
                    plain = f"({plain.rsplit(f' {disp} ', 1)[0]}) {disp} {nice(f3)}" if op in ("+", "-", "*", "/") else plain
                kind = "fraction"
                pattern = f"{variant}_{op}"

            else:
                a, b, c, d = r.sample(range(1, 9), 4)
                m, n, p = r.randint(1, 5), r.randint(1, 5), r.randint(1, 5)
                if op in ("+", "-"):
                    variant = r.choice(["racionales_mismo_denominador", "racionales_distinto_denominador", "dos_variables_denominador"])
                    if variant == "racionales_mismo_denominador":
                        den = (x + a) * (x + b)
                        left = (m * x + c) / den
                        right = (n * x - a) / den
                    elif variant == "dos_variables_denominador":
                        left = (m * x + n * y) / (x * y)
                        right = (p * x - a * y) / (x * y)
                    else:
                        left = sp.Rational(m, 1) / (x + a)
                        right = sp.Rational(n, 1) / (y + b)
                    shown = left + right if op == "+" else left - right
                elif op == "*":
                    variant = r.choice(["producto_racional_con_cancelacion", "producto_dos_variables_denominador", "producto_tres_factores"])
                    if variant == "producto_dos_variables_denominador":
                        left = (m * x) / (y * (x + a))
                        right = (n * y * (x + a)) / (x * (y + b))
                        shown = left * right
                    elif variant == "producto_tres_factores":
                        left = (m * x**2) / (y + a)
                        right = (n * (y + a)) / (z + b)
                        third = (p * (z + b)) / (x * y)
                        shown = left * right * third
                    else:
                        left = (m * x * y) / ((x + a) * z)
                        right = (n * z * (x + a)) / (y * (x + b))
                        shown = left * right
                else:
                    variant = r.choice(["division_racional_con_cancelacion", "division_dos_variables_denominador", "division_compuesta"])
                    if variant == "division_dos_variables_denominador":
                        left = (m * x * (y + a)) / (y * (x + b))
                        right = (n * (y + a)) / (x * (x + c))
                        shown = left / right
                    elif variant == "division_compuesta":
                        left = ((x + a) / (y + b)) / ((x + a) / (z + c))
                        right = sp.Rational(m, n) * y / z
                        shown = left / right
                    else:
                        left = (m * x * (y + a)) / (z * (x + b))
                        right = (n * (y + a)) / (z * (x + c))
                        shown = left / right
                
                result = sp.cancel(sp.together(shown))
                if result.has(sp.zoo, sp.oo, sp.nan): continue
                plain = sp.sstr(shown)
                latex = sp.latex(shown)
                kind = "expression"
                pattern = variant

            return draft_question(
                topic, subtopic, diff, f"Calcula y simplifica: {plain}", result, kind,
                plain=plain, latex=latex, instructions="Responde en la forma mas simplificada equivalente.",
                template="simplificacion_fracciones", pattern=pattern
            )
        except Exception:
            continue


def _powers_didactic(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    while True:
        try:
            if lv == 1:
                arquetipo = r.choice(['exp_cero', 'exp_uno', 'exp_negativo', 'prod_igual_base', 'cociente_igual_base'])
            elif lv == 2:
                arquetipo = r.choice(['potencia_potencia', 'potencia_producto', 'potencia_cociente', 'exp_fraccionario', 'coef_potencia'])
            else:
                arquetipo = r.choice(['prod_cociente_mix', 'neg_mix_simplifica', 'tres_variables', 'doble_fraccion_exp'])

            b = r.choice(['x', 'y', 'z', '2', '3', '5'])
            
            if arquetipo == 'exp_cero':
                s = f"({b} * {r.randint(2,20)})**0"
            elif arquetipo == 'exp_uno':
                coef = r.randint(2, 12)
                s = f"{coef} * ({b})**1"
            elif arquetipo == 'exp_negativo':
                exp = r.randint(2, 9)
                s = f"{b}**(-{exp})"
            elif arquetipo == 'prod_igual_base':
                exps = [r.randint(2, 7) for _ in range(r.choice([2, 3]))]
                s = " * ".join([f"{b}**{e}" for e in exps])
            elif arquetipo == 'cociente_igual_base':
                e1, e2 = r.randint(5, 14), r.randint(1, 4)
                s = f"{b}**{e1} / {b}**{e2}"
            
            elif arquetipo == 'potencia_potencia':
                e1, e2 = r.randint(2, 5), r.randint(2, 5)
                s = f"({b}**{e1})**{e2}"
                if r.random() < 0.3: s = f"(({b}**{e1})**{e2})**{r.randint(2,3)}"
            elif arquetipo == 'potencia_producto':
                b2 = r.choice([v for v in ['x','y','z','2','3'] if v != b])
                e = r.randint(2, 6)
                s = f"({b} * {b2})**{e}"
            elif arquetipo == 'potencia_cociente':
                b2 = r.choice([v for v in ['x','y','z','2','3'] if v != b])
                e = r.randint(2, 6)
                s = f"({b} / {b2})**{e}"
            elif arquetipo == 'exp_fraccionario':
                num, den = r.randint(1, 5), r.randint(2, 5)
                s = f"({b}**{num})**(1/{den})"
            elif arquetipo == 'coef_potencia':
                k = r.randint(2, 5)
                e = r.randint(2, 5)
                s = f"({k} * {b})**{e}"

            elif arquetipo == 'prod_cociente_mix':
                e1, e2, e3 = r.randint(3, 7), r.randint(3, 7), r.randint(1, 5)
                s = f"({b}**{e1} * {b}**{e2}) / {b}**{e3}"
            elif arquetipo == 'neg_mix_simplifica':
                b2 = r.choice([v for v in ['x','y','z'] if v != b])
                m, n, p, q = r.randint(1,5), r.randint(1,5), r.randint(1,5), r.randint(1,5)
                s = f"({b}**-{m} * {b2}**{n}) / ({b}**{p} * {b2}**-{q})"
            elif arquetipo == 'tres_variables':
                e1, e2, e3 = r.randint(1,4), r.randint(1,4), r.randint(1,4)
                n = r.randint(2, 4)
                s = f"(x**{e1} * y**{e2} * z**{e3})**{n}"
            elif arquetipo == 'doble_fraccion_exp':
                c, d = r.randint(1, 5), r.randint(2, 5)
                if c == d: c = c + 1
                m, n = r.randint(1, 4), r.randint(1, 4)
                s = f"({c}/{d})**{m} * ({c}/{d})**{n}"
            
            ej_vis = sp.sympify(s, evaluate=False)
            result = sp.simplify(sp.sympify(s))
            if result.has(sp.zoo, sp.oo, sp.nan, sp.I): continue
            
            plain = str(s).replace('**', '^').replace('*', '·')
            latex = sp.latex(ej_vis)
            return draft_question(
                topic, subtopic, diff, "Simplifica usando leyes de potencias.", result, "expression", 
                plain=plain, latex=latex, template="potencias", pattern=arquetipo
            )
        except Exception:
            continue


def _radicals_didactic(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    while True:
        try:
            if lv == 1:
                arquetipo = r.choice(['raiz_producto', 'raiz_cociente_basico', 'simplificacion_directa', 'suma_semejantes_simple'])
            elif lv == 2:
                arquetipo = r.choice(['raiz_de_raiz', 'potencia_raiz', 'introducir_factor', 'extraccion_factor', 'suma_con_simplif'])
            else:
                arquetipo = r.choice(['racionalizacion_simple', 'racionalizacion_binomio', 'coef_en_radicando', 'radical_fraccionario_mixto'])

            if arquetipo == 'raiz_producto':
                n = r.choice([2, 3])
                a, b = r.randint(2, 6)**n, r.randint(2, 6)**n
                s = f"root({a}, {n}) * root({b}, {n})"
            elif arquetipo == 'raiz_cociente_basico':
                n = r.choice([2, 3])
                a, b = r.randint(2, 6)**n, r.randint(2, 4)**n
                s = f"root({a}/{b}, {n})"
            elif arquetipo == 'simplificacion_directa':
                n, p = r.randint(2, 5), r.randint(2, 5)
                s = f"root(x**{p}, {n*p})"
            elif arquetipo == 'suma_semejantes_simple':
                n = r.choice([2, 3, 5, 7])
                a, b = r.randint(2, 8), r.randint(1, 8)
                s = f"{a}*sqrt({n}) + {b}*sqrt({n})"
            
            elif arquetipo == 'raiz_de_raiz':
                n1, n2 = r.randint(2, 4), r.randint(2, 4)
                s = f"root(root(x, {n2}), {n1})"
            elif arquetipo == 'potencia_raiz':
                n, p = r.randint(2, 5), r.randint(2, 6)
                s = f"(root(x, {n}))**{p}"
            elif arquetipo == 'introducir_factor':
                n, a, b = r.choice([2,3]), r.randint(2, 5), r.randint(2, 10)
                s = f"{a} * root({b}, {n})"
            elif arquetipo == 'extraccion_factor':
                n, a, b = r.choice([2,3]), r.randint(2, 4), r.randint(2, 7)
                s = f"root({(a**n) * b}, {n})"
            elif arquetipo == 'suma_con_simplif':
                n = r.choice([2, 3, 5])
                k, m = r.randint(2, 6), r.randint(2, 6)
                s = f"sqrt({k*k*n}) + sqrt({m*m*n})"

            elif arquetipo == 'racionalizacion_simple':
                a, b = r.randint(2, 10), r.choice([2, 3, 5, 7, 11])
                s = f"{a} / sqrt({b})"
            elif arquetipo == 'racionalizacion_binomio':
                a, b, c = r.randint(2, 8), r.choice([2, 3, 5, 7]), r.randint(1, 5)
                s = f"{a} / (sqrt({b}) + {c})"
            elif arquetipo == 'coef_en_radicando':
                k = r.choice([4, 9, 16, 25, 36])
                exp = 2 * r.randint(1, 5)
                s = f"sqrt({k} * x**{exp})"
            elif arquetipo == 'radical_fraccionario_mixto':
                a, p, q = r.randint(1, 5), r.randint(1, 8), r.choice([4, 9, 16])
                b, rr, ss = r.randint(1, 5), r.randint(1, 8), r.choice([4, 9, 16])
                s = f"{a} * sqrt({p}/{q}) + {b} * sqrt({rr}/{ss})"

            ej_vis = sp.sympify(s, evaluate=False)
            result = sp.simplify(sp.sympify(s))
            if result.has(sp.zoo, sp.oo, sp.nan, sp.I): continue

            plain = str(s).replace('**', '^').replace('*', '·').replace('root', 'raiz')
            latex = sp.latex(ej_vis)
            return draft_question(
                topic, subtopic, diff, "Simplifica completamente el radical.", result, "expression", 
                plain=plain, latex=latex, template="radicales", pattern=arquetipo
            )
        except Exception:
            continue


def _mixed_cascade(topic: str, subtopic: str, diff: str, r):
    lv = level(diff)
    lo, hi = 2, 6 + lv*2
    
    while True:
        try:
            if lv == 1:
                arquetipo = r.choice([
                    'castillo_basico', 'radical_anidado_basico', 
                    'potencias_cruzadas_basico', 'fraccion_compuesta_basica'
                ])
            elif lv == 2:
                arquetipo = r.choice([
                    'castillo_medio', 'radical_anidado_medio', 
                    'negativo_anidado_medio', 'triple_fraccion_media', 
                    'triple_piso_numerico'
                ])
            else:
                arquetipo = r.choice([
                    'castillo_complejo', 'radical_anidado_complejo', 
                    'cascada_fraccionaria_letras', 'gran_simplificacion', 
                    'simbolico_potencia_radical', 'triple_piso_letras', 
                    'operacion_combinada_extrema'
                ])
            
            if arquetipo == 'castillo_basico':
                a, b, c, d = r.randint(1, 5), r.randint(1, 5), r.randint(2, 6), r.randint(2, 6)
                s = f"{a} - {b}/({c} - 1/{d})"
            elif arquetipo == 'radical_anidado_basico':
                c = r.randint(1, 5)**2
                b = r.randint(1, 4)**2
                s = f"sqrt({b} + sqrt({c}))"
            elif arquetipo == 'potencias_cruzadas_basico':
                ba, bb = r.sample([2, 3, 5], 2)
                m, p = r.randint(3, 6), r.randint(1, 2)
                s = f"({ba}**{m} * {bb}**3) / ({ba}**{p} * {bb}**2)"
            elif arquetipo == 'fraccion_compuesta_basica':
                a, b, c, d = r.randint(1, 5), r.randint(2, 5), r.randint(1, 5), r.randint(2, 5)
                if Fraction(a, b) == Fraction(c, d): continue
                s = f"({a}/{b} + {c}/{d}) / ({a}/{b} - {c}/{d})"

            elif arquetipo == 'castillo_medio':
                a, b, c, d = r.randint(lo, hi), r.randint(lo, hi), r.randint(lo, hi), r.randint(lo, hi)
                s = f"{a} - 1/({b} - 1/({c} - 1/{d}))"
            elif arquetipo == 'radical_anidado_medio':
                c = r.randint(1, 10)
                raiz_c = math.isqrt(c)
                b = (r.randint(2,5)**2 - raiz_c) if raiz_c**2==c else r.choice([1,4,9,16])
                if b < 1: b = 4
                a = r.randint(2,5)**2
                s = f"sqrt({a} + sqrt({b} + sqrt({c})))"
            elif arquetipo == 'negativo_anidado_medio':
                a, b, c = r.randint(2, 6), r.randint(2, 6), r.randint(2, 10)
                s = f"((1/{a})**-1 + (1/{b})**-1)**-1 * {c}"
            elif arquetipo == 'triple_fraccion_media':
                def fr(): return r.randint(1,5), r.randint(2,5)
                a,b=fr(); c,d=fr(); e,f=fr(); g,h=fr()
                if sp.Rational(e,f) == sp.Rational(g,h): continue
                s = f"(({a}/{b} + {c}/{d}) / ({e}/{f} - {g}/{h}))"
            elif arquetipo == 'triple_piso_numerico':
                a, b, c = r.randint(1, hi), r.randint(1, hi), r.randint(1, hi)
                d, e, f = r.randint(1, hi), r.randint(1, hi), r.randint(2, hi)
                s = f"{a} / ({b} + {c} / ({d} + {e}/{f}))"

            elif arquetipo == 'castillo_complejo':
                a, b, c, d, e = [r.randint(lo, hi) for _ in range(5)]
                s = f"{a} - 1/({b} - 1/({c} - 1/({d} - 1/{e})))"
                if r.random() < 0.5: s = s.replace(str(e), "x") 
            elif arquetipo == 'radical_anidado_complejo':
                s = f"sqrt(x**8 + sqrt(x**4 + sqrt(x**2)))" if r.random() < 0.5 else f"sqrt(x**4 * sqrt(y**8)) / (x * y)"
            elif arquetipo == 'cascada_fraccionaria_letras':
                s = f"((1/x - 1/y) / (1/x**2 - 1/y**2)) * (x*y)"
            elif arquetipo == 'gran_simplificacion':
                a,b = r.randint(2,6), r.randint(2,6)
                if a == b: continue
                e1  = r.randint(2,5)
                c, d = r.sample(range(2, 8), 2)
                m   = r.randint(2,5)
                e2  = r.choice([2,4])
                s = f"({a}**{e1} - {b}**{e1}) / (1/(1/{c} - 1/{d})) + ({a}/{b})**-{m} - (-{a})**{e2}"
            elif arquetipo == 'simbolico_potencia_radical':
                a, b = r.choice([2,4,6,8]), r.choice([2,4,6])
                c, d = r.randint(0,3), r.randint(0,3)
                s = f"sqrt(x**{a} * y**{b}) / (x**{c} * y**{d})"
            elif arquetipo == 'triple_piso_letras':
                a, b, c = r.randint(1, 5), r.randint(1, 5), r.randint(1, 5)
                s = f"x / ({a} + y / ({b} + z/{c}))"
            elif arquetipo == 'operacion_combinada_extrema':
                a,b = r.randint(2,5), r.randint(2,5)
                e1,e2 = r.randint(2,4), r.randint(2,4)
                bn = r.choice([-2,-3,-4])
                e3 = r.choice([2,4])
                div = r.randint(2,5)
                s_exp = r.randint(2, 4)
                s = f"({a}**{e1} - {b}**{e2}) / 1**-5 - ({bn})**{e3} / {div}**2 + (1/{div})**-{s_exp}"

            ej_vis = sp.sympify(s, evaluate=False)
            result = sp.simplify(sp.sympify(s))
            if result.has(sp.zoo, sp.oo, sp.nan, sp.I): continue

            plain = str(s).replace('**', '^').replace('*', '·')
            latex = sp.latex(ej_vis)
            
            return draft_question(
                topic, subtopic, diff, 
                "Simplifica la expresion compleja en cascada.", 
                result, "expression", 
                plain=plain, latex=latex, 
                template="simplificacion_mixta", pattern=arquetipo
            )
        except Exception:
            continue


def generate_simplification(topic: str, subtopic: str, diff: str, r):
    last = None
    for _ in range(36):
        if subtopic.startswith("fracciones_"):
            q = _fraction_question(topic, subtopic, diff, r)
        elif subtopic == "potencias":
            q = _powers_didactic(topic, subtopic, diff, r)
        elif subtopic == "radicales":
            q = _radicals_didactic(topic, subtopic, diff, r)
        else:
            q = _mixed_cascade(topic, subtopic, diff, r)
            
        last = q
        if not _question_too_simple(q):
            return q
    return last


# ---------------------------------------------------------------------------
# REGISTRO DE TEMAS
# ---------------------------------------------------------------------------

TOPICS = {
    "agilidad_mental": {
        "label": "Agilidad mental",
        "gen": generate_mental,
        "subs": {
            "calculo_rapido": {"label": "Calculo rapido"},
            "operaciones_mixtas": {"label": "Operaciones mixtas"},
            "jerarquia_operaciones": {"label": "Jerarquia de operaciones"},
            "desafio_4_terminos": {"label": "Desafio de 4 terminos"},
        },
    },
    "conversiones_decimales": {
        "label": "Conversiones y decimales",
        "gen": generate_conversions,
        "subs": {
            "decimal_a_fraccion": {"label": "Decimal a fraccion"},
            "fraccion_a_decimal": {"label": "Fraccion a decimal"},
            "mixto_a_decimal": {"label": "Mixto a decimal"},
            "decimal_a_mixto": {"label": "Decimal a mixto"},
            "mixto_a_fraccion": {"label": "Mixto a fraccion"},
            "comparacion": {"label": "Comparacion"},
            "redondeo": {"label": "Redondeo"},
            "periodico_puro": {"label": "Periodico puro"},
            "periodico_mixto": {"label": "Periodico mixto"},
            "operacion": {"label": "Operacion con decimales/fracciones"},
        },
    },
    "expresiones_algebraicas": {
        "label": "Expresiones algebraicas",
        "gen": generate_algebra,
        "subs": {
            "grado_polinomio": {"label": "Grado de polinomio"},
            "suma_polinomios": {"label": "Suma de polinomios"},
            "resta_polinomios": {"label": "Resta de polinomios"},
            "multiplicacion_polinomios": {"label": "Multiplicacion de polinomios"},
            "division_polinomios": {"label": "Division de polinomios"},
            "simplificacion_algebraica": {"label": "Simplificacion algebraica"},
        },
    },
    "razones_porcentajes": {
        "label": "Razones, proporciones y porcentajes",
        "gen": generate_percentages,
        "subs": {
            "proporcion_inversa": {"label": "Proporcion inversa"},
            "proporcion_directa": {"label": "Proporcion directa"},
            "regla_3_compuesta": {"label": "Regla de 3 compuesta"},
            "descuento_doble": {"label": "Descuento doble"},
            "que_porcentaje": {"label": "Que porcentaje es/de"},
            "cuanto_es_porcentaje_de": {"label": "Cuanto es el porcentaje de"},
            "porcentaje_de_porcentaje": {"label": "Porcentaje de porcentaje"},
        },
    },
    "simplificacion": {
        "label": "Simplificacion",
        "gen": generate_simplification,
        "subs": {
            "fracciones_suma": {"label": "Fracciones: suma"},
            "fracciones_resta": {"label": "Fracciones: resta"},
            "fracciones_multiplicacion": {"label": "Fracciones: multiplicacion"},
            "fracciones_division": {"label": "Fracciones: division"},
            "potencias": {"label": "Propiedades de potencias"},
            "radicales": {"label": "Propiedades de radicales"},
            "mixta_avanzada": {"label": "Simplificacion mixta"},
        },
    },
}


def topic_label(topic: str) -> str:
    return TOPICS[topic]["label"]


def subtopic_label(topic: str, subtopic: str) -> str:
    return TOPICS[topic]["subs"][subtopic]["label"]


# ---------------------------------------------------------------------------
# ESTADISTICAS
# ---------------------------------------------------------------------------

def attempt_record(question: dict[str, Any], result: dict[str, Any], elapsed: float | None = None) -> dict[str, Any]:
    return {
        "topic": question.get("topic"),
        "subtopic": question.get("subtopic"),
        "difficulty": question.get("difficulty"),
        "time": elapsed,
        "correct": bool(result.get("correcto") or result.get("correct")),
        "pattern": (question.get("metadata") or {}).get("pattern"),
        "template": (question.get("metadata") or {}).get("template"),
        "questionId": question.get("id") or result.get("questionId"),
    }


def summarize_attempts(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    by_subtopic: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "time": []})
    for rec in records:
        key = rec.get("subtopic") or "desconocido"
        bucket = by_subtopic[key]
        bucket["total"] += 1
        bucket["correct"] += int(bool(rec.get("correct")))
        if rec.get("time") is not None:
            bucket["time"].append(float(rec["time"]))
    weak = []
    detail = {}
    for key, bucket in by_subtopic.items():
        accuracy = bucket["correct"] / bucket["total"] if bucket["total"] else 0
        avg_time = sum(bucket["time"]) / len(bucket["time"]) if bucket["time"] else None
        detail[key] = {
            "total": bucket["total"],
            "correct": bucket["correct"],
            "accuracy": round(accuracy * 100, 2),
            "averageTime": round(avg_time, 2) if avg_time is not None else None,
        }
        if bucket["total"] >= 2 and accuracy < 0.7:
            weak.append(key)
    return {
        "total": total,
        "correct": correct,
        "score": round(correct / total * 100, 2) if total else 0,
        "averageTime": round(
            sum(float(r["time"]) for r in records if r.get("time") is not None)
            / max(1, sum(1 for r in records if r.get("time") is not None)),
            2,
        )
        if any(r.get("time") is not None for r in records)
        else None,
        "weakSubtopics": weak,
        "bySubtopic": detail,
        "adaptiveReady": [
            {
                "subtopic": r.get("subtopic"),
                "difficulty": r.get("difficulty"),
                "time": r.get("time"),
                "correct": r.get("correct"),
            }
            for r in records
        ],
    }


# ---------------------------------------------------------------------------
# EXAMEN
# ---------------------------------------------------------------------------

MAX_EXAM_QUESTIONS = 50


def _signature(question: dict[str, Any]) -> str:
    meta = question.get("metadata") or {}
    return "|".join(
        str(x)
        for x in (
            question.get("topic"),
            question.get("subtopic"),
            question.get("difficulty"),
            meta.get("antiRepeatSignature"),
        )
    )


def _build_exam_core(
    question_factory: Callable[..., dict[str, Any]],
    *,
    topic: str | None = None,
    subtopic: str | None = None,
    difficulty: str = "normal",
    count: int = 10,
    seed: Any = None,
    include_answers: bool = False,
    public_fn: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    count = max(1, min(MAX_EXAM_QUESTIONS, int(count)))
    diff = difficulty_key(difficulty)
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern_counts: dict[str, int] = {}
    tries = 0
    while len(questions) < count and tries < count * 80:
        tries += 1
        q = question_factory(topic=topic, subtopic=subtopic, difficulty=diff, seed=(str(seed) + str(tries) if seed is not None else None), include_answer=True)
        sig = _signature(q)
        display_sig = hashlib.sha1(str(q.get("display")).encode("utf-8")).hexdigest()[:16]
        meta = q.get("metadata") or {}
        pattern_key = "|".join(str(x) for x in (q.get("subtopic"), meta.get("template"), meta.get("pattern")))
        pattern_limit = max(2, count // 3)
        if sig in seen or display_sig in seen:
            continue
        if pattern_counts.get(pattern_key, 0) >= pattern_limit and tries < count * 50:
            continue
        seen.add(sig)
        seen.add(display_sig)
        pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
        q["exam"] = True
        questions.append(public_fn(q, include_answers) if public_fn else q)
    return {
        "id": hashlib.sha1(f"{seed}:{topic}:{subtopic}:{diff}:{time.time()}".encode("utf-8")).hexdigest()[:14],
        "topic": topic or "mixed",
        "subtopic": subtopic or "mixed",
        "difficulty": diff,
        "count": len(questions),
        "createdAt": int(time.time()),
        "questions": questions,
        "antiRepeat": {"tries": tries, "uniqueSignatures": len(seen)},
    }


def _grade_exam_core(items: list[dict[str, Any]], validate_fn: Callable[[Any, Any], dict[str, Any]]) -> dict[str, Any]:
    results = []
    records = []
    correct = 0
    for item in items:
        result = validate_fn(
            item.get("token") or item.get("questionToken") or item.get("question"),
            str(item.get("answer", item.get("respuesta", item.get("respuesta_usuario", "")))),
        )
        results.append(result)
        correct += int(result["correcto"])
        if isinstance(item.get("question"), dict):
            records.append(attempt_record(item["question"], result, item.get("time")))
    total = len(results)
    summary = summarize_attempts(records) if records else {}
    return {
        "correctas": correct,
        "total": total,
        "score": round(correct / total * 100, 2) if total else 0,
        "results": results,
        "stats": summary,
    }

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _render_math_png(latex: str, fontsize: int = 16, dpi: int = 150):
    if not latex: # Asumiendo que HAS_MATPLOTLIB está definido
        return None
    try:
        fig = plt.figure(figsize=(8, 1.25))
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        # Se usa el color --ink exacto del frontend: #18202f
        ax.text(0.02, 0.5, f"${latex}$", fontsize=fontsize, va="center", ha="left", color="#18202f")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.06)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def pdf_bytes(exam: dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas
    import textwrap

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 42
    content_w = width - 2 * margin
    
    # Paleta de colores exacta de Teorema (basada en el CSS)
    brand = (18 / 255, 75 / 255, 104 / 255)     # --brand: #124b68
    accent = (194 / 255, 122 / 255, 44 / 255)   # --accent: #c27a2c
    soft = (244 / 255, 247 / 255, 251 / 255)    # --soft: #f4f7fb
    line = (217 / 255, 225 / 255, 234 / 255)    # --line: #d9e1ea
    ink = (24 / 255, 32 / 255, 47 / 255)        # --ink: #18202f
    muted = (102 / 255, 112 / 255, 133 / 255)   # --muted: #667085

    def fill(rgb):
        c.setFillColorRGB(*rgb)

    def stroke(rgb):
        c.setStrokeColorRGB(*rgb)

    def header(title: str):
        # Fondo del header simulando la barra superior (topbar)
        fill(brand)
        c.rect(0, height - 58, width, 58, fill=1, stroke=0)
        # Borde inferior del header con el color accent
        fill(accent)
        c.rect(0, height - 58, width, 4, fill=1, stroke=0)
        
        fill((1, 1, 1))
        c.setFont("Helvetica-Bold", 17) # Tamaño de título ajustado
        c.drawString(margin, height - 35, title)
        c.setFont("Helvetica", 9)
        fill((0.85, 0.88, 0.92))
        meta = f"{exam.get('moduleLabel', 'Modulo 0')} | Dificultad: {str(exam.get('difficulty', '')).capitalize()} | Preguntas: {exam.get('count', len(exam.get('questions', [])))}"
        c.drawRightString(width - margin, height - 35, meta)

    def wrap(text: str, font: str, size: int, max_w: float):
        words = str(text).split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and stringWidth(candidate, font, size) > max_w:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    def ensure_space(y: float, needed: float, title: str) -> float:
        if y - needed < 42:
            c.showPage()
            header(title)
            return height - 84
        return y

    header("Examen de práctica")
    y = height - 86
    c.setFont("Helvetica-Bold", 9)
    fill(muted)
    c.drawString(margin, y, "ESTUDIANTE:")
    stroke(line)
    c.line(margin + 72, y - 2, margin + 270, y - 2)
    c.drawString(margin + 300, y, "FECHA:")
    c.line(margin + 342, y - 2, width - margin, y - 2)
    y -= 26

    # Panel de instrucciones con bordes redondeados (simulando --radius-sm)
    fill(soft)
    stroke(line)
    c.roundRect(margin, y - 34, content_w, 34, 8, fill=1, stroke=1)
    fill(ink)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin + 10, y - 12, "INSTRUCCIONES")
    c.setFont("Helvetica", 8)
    c.drawString(margin + 10, y - 25, "Resuelve con procedimiento claro. Se aceptan formas algebraicamente equivalentes cuando estén simplificadas.")
    y -= 52

    for idx, question in enumerate(exam.get("questions", []), 1):
        display = question.get("display") or {}
        latex = display.get("latex")
        plain = display.get("plain") or question.get("prompt") or ""
        math_buf = _render_math_png(latex) if latex else None
        math_h = 0
        math_w = 0
        if math_buf:
            try:
                img = ImageReader(math_buf)
                iw, ih = img.getSize()
                math_w = iw * 72 / 150
                math_h = ih * 72 / 150
                if math_w > content_w - 16:
                    scale = (content_w - 16) / math_w
                    math_w *= scale
                    math_h *= scale
            except Exception:
                math_buf = None
                
        text_lines = wrap(plain, "Helvetica", 9, content_w - 18)
        box_h = max(38, (math_h + 18) if math_buf else len(text_lines) * 12 + 18)
        y = ensure_space(y, box_h + 54, "Examen de práctica")
        c.setFont("Helvetica-Bold", 10)
        
        # Color del enumerador ajustado a brand
        fill(brand)
        c.drawString(margin, y, f"{idx:02d}.")
        y -= 12
        
        # Caja de la pregunta con bordes redondeados (--radius-md)
        fill(soft)
        stroke(line)
        c.roundRect(margin, y - box_h, content_w, box_h, 12, fill=1, stroke=1)
        
        if math_buf:
            math_buf.seek(0)
            c.drawImage(ImageReader(math_buf), margin + 8, y - box_h + (box_h - math_h) / 2, math_w, math_h, mask="auto")
        else:
            fill(ink)
            c.setFont("Helvetica", 9)
            ty = y - 14
            for line_text in text_lines:
                c.drawString(margin + 8, ty, line_text)
                ty -= 12
        y -= box_h + 12
        fill(muted)
        c.setFont("Helvetica", 9)
        c.drawString(margin, y, "Respuesta:")
        stroke(line)
        c.line(margin + 62, y - 2, width - margin, y - 2)
        y -= 20
        c.line(margin, y, width - margin, y)
        y -= 24

    c.showPage()
    header("Solucionario y resumen")
    y = height - 84
    summary = exam.get("summary") or {}
    if summary:
        c.setFont("Helvetica-Bold", 10)
        fill(ink)
        c.drawString(margin, y, f"Resultado: {summary.get('correct', 0)}/{summary.get('total', 0)} | Nota: {summary.get('score', 0)}%")
        y -= 18
        if summary.get("averageTime") is not None:
            c.setFont("Helvetica", 9)
            fill(muted)
            c.drawString(margin, y, f"Tiempo promedio: {summary.get('averageTime')} s")
            y -= 18

    for idx, question in enumerate(exam.get("questions", []), 1):
        answer = question.get("answer") or {}
        answer_text = answer.get("plain", "(sin respuesta)")
        answer_latex = answer.get("latex")
        sol_buf = _render_math_png(answer_latex, fontsize=14) if answer_latex else None
        y = ensure_space(y, 34, "Solucionario")
        c.setFont("Helvetica-Bold", 9)
        fill(brand) # Cambiado de teal a brand
        c.drawString(margin, y, f"{idx:02d}.")
        if sol_buf:
            try:
                img = ImageReader(sol_buf)
                iw, ih = img.getSize()
                w_pt, h_pt = iw * 72 / 150, ih * 72 / 150
                if w_pt > content_w - 30:
                    scale = (content_w - 30) / w_pt
                    w_pt *= scale
                    h_pt *= scale
                sol_buf.seek(0)
                c.drawImage(ImageReader(sol_buf), margin + 22, y - h_pt + 4, w_pt, h_pt, mask="auto")
                y -= max(18, h_pt + 8)
            except Exception:
                sol_buf = None
                
        if not sol_buf:
            c.setFont("Helvetica", 9)
            fill(ink)
            for line_txt in textwrap.wrap(answer_text, 92) or [""]:
                c.drawString(margin + 22, y, line_txt)
                y -= 12
        stroke(line)
        c.line(margin, y, width - margin, y)
        y -= 8

    c.save()
    buf.seek(0)
    return buf.read()


def export_exam_pdf(exam: dict[str, Any], output_path: str | None = None) -> str:
    if not output_path:
        output_path = os.path.join(os.path.expanduser("~"), "Downloads", f"modulo0_examen_{exam.get('id', int(time.time()))}.pdf")
    if not output_path.lower().endswith(".pdf"):
        output_path += ".pdf"
    with open(output_path, "wb") as fh:
        fh.write(pdf_bytes(exam))
    return os.path.abspath(output_path)

# ---------------------------------------------------------------------------
# BACKEND / SERVIDOR
# ---------------------------------------------------------------------------

MODULE = "modulo_0"
MODULE_LABEL = "Modulo 0 - Mate basica"
SECRET = os.environ.get("MODULO0_SECRET") or secrets.token_hex(32)
RECENT_SIGNATURES: dict[tuple[str, str, str], deque[str]] = defaultdict(lambda: deque(maxlen=10))


class BackendError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def new_rng(seed: Any = None) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random(secrets.randbits(64))


def jb(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def ub64(data: str) -> bytes:
    return base64.urlsafe_b64decode((data + "=" * (-len(data) % 4)).encode("ascii"))


def sign(payload: dict[str, Any]) -> str:
    body = b64(jb(payload))
    sig = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def unsign(token: str) -> dict[str, Any]:
    try:
        body, sig = token.split(".", 1)
    except ValueError as exc:
        raise BackendError("Token invalido", 400) from exc
    expected = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise BackendError("Token invalido o expirado", 400)
    return json.loads(ub64(body).decode("utf-8"))


def _normalize_difficulty(difficulty: Any = None, default: str = "normal") -> str:
    value = difficulty if difficulty not in (None, "") else default
    if str(value).strip().lower() in {"medio", "media"}:
        value = "normal"
    if str(value).strip().lower() in {"dificil", "difícil"}:
        value = "avanzado"
    return difficulty_key(str(value))


def _finalize_question(draft: dict[str, Any], include_answer: bool = True) -> dict[str, Any]:
    topic = draft["topic"]
    subtopic = draft["subtopic"]
    diff = difficulty_key(draft["difficulty"])
    answer = answer_obj(draft["value"], draft["kind"], draft.get("answerPlain"), draft.get("answerLatex"))
    metadata = copy.deepcopy(draft.get("metadata") or {})
    q = {
        "module": MODULE,
        "moduleLabel": MODULE_LABEL,
        "topic": topic,
        "topicLabel": topic_label(topic),
        "subtopic": subtopic,
        "subtopicLabel": subtopic_label(topic, subtopic),
        "difficulty": diff,
        "difficultyLabel": difficulty_label(diff),
        "exam": False,
        "prompt": draft.get("prompt") or "",
        "instructions": draft.get("instructions") or "Responde con el resultado simplificado.",
        "display": draft.get("display") or {"plain": draft.get("prompt") or "", "format": "text"},
        "answer": answer,
        "metadata": metadata,
    }
    raw = jb({"t": topic, "s": subtopic, "d": diff, "p": q["display"], "a": answer, "m": metadata})
    q["id"] = hashlib.sha1(raw).hexdigest()[:14]
    q["token"] = sign(
        {
            "id": q["id"],
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": diff,
            "answer": answer,
            "metadata": metadata,
            "createdAt": int(time.time()),
        }
    )
    return q if include_answer else public(q, False)


def public(question: dict[str, Any], include_answer: bool = False) -> dict[str, Any]:
    q = copy.deepcopy(question)
    if not include_answer:
        q.pop("answer", None)
    return q


def choose_topic_sub(r: random.Random, topic: str | None = None, subtopic: str | None = None) -> tuple[str, str]:
    tk = (topic or "mixed").strip().lower()
    if tk in ("mixed", "mixto", "aleatorio", "random", ""):
        tk = r.choice(list(TOPICS))
    if tk not in TOPICS:
        raise BackendError(f"Tema no encontrado: {topic}", 404)
    sk = (subtopic or "random").strip().lower()
    if sk in ("mixed", "mixto", "aleatorio", "random", ""):
        sk = r.choice(list(TOPICS[tk]["subs"]))
    if sk not in TOPICS[tk]["subs"]:
        raise BackendError(f"Subtema no encontrado: {subtopic}", 404)
    return tk, sk


def _remember_or_reject(question: dict[str, Any]) -> bool:
    meta = question.get("metadata") or {}
    sig = meta.get("antiRepeatSignature") or question.get("id")
    key = (question["topic"], question["subtopic"], question["difficulty"])
    if sig in RECENT_SIGNATURES[key]:
        return False
    RECENT_SIGNATURES[key].append(sig)
    return True


def generate_question(
    topic: str | None = None,
    subtopic: str | None = None,
    difficulty: str = "normal",
    seed: Any = None,
    include_answer: bool = True,
    rng: random.Random | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    r = rng or new_rng(seed)
    diff = _normalize_difficulty(difficulty)
    tk, sk = choose_topic_sub(r, topic, subtopic)
    last = None
    for _ in range(24):
        draft = TOPICS[tk]["gen"](tk, sk, diff, r)
        q = _finalize_question(draft, include_answer=True)
        last = q
        if _remember_or_reject(q):
            return q if include_answer else public(q, False)
    return last if include_answer else public(last, False)


def generate_exam(
    topic: str | None = None,
    subtopic: str | None = None,
    difficulty: str = "normal",
    count: int = 10,
    seed: Any = None,
    include_answers: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    diff = _normalize_difficulty(difficulty)
    exam = _build_exam_core(
        generate_question,
        topic=topic,
        subtopic=subtopic,
        difficulty=diff,
        count=count,
        seed=seed,
        include_answers=include_answers,
        public_fn=public,
    )
    exam.update({"module": MODULE, "moduleLabel": MODULE_LABEL, "difficultyLabel": difficulty_label(diff), "level": level(diff)})
    return exam


def _payload_from_token_or_question(token_or_question: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(token_or_question, str):
        return unsign(token_or_question)
    if isinstance(token_or_question, dict):
        if token_or_question.get("answer") or token_or_question.get("expected"):
            return token_or_question
        if token_or_question.get("token"):
            return unsign(token_or_question["token"])
    raise BackendError("No hay respuesta para validar", 400)


def validate_answer(token_or_question: str | dict[str, Any], user_answer: Any) -> dict[str, Any]:
    try:
        return validate_payload(_payload_from_token_or_question(token_or_question), user_answer)
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(str(exc), 400) from exc


def generate_explanation(token_or_question: str | dict[str, Any], user_answer: Any) -> dict[str, Any]:
    return _explain_from_payload(_payload_from_token_or_question(token_or_question), user_answer)


def grade_exam(items: list[dict[str, Any]]) -> dict[str, Any]:
    return _grade_exam_core(items, validate_answer)


def get_options() -> dict[str, Any]:
    quick = [
        {"key": topic, "label": data["label"], "topic": topic, "difficulty": "avanzado"}
        for topic, data in TOPICS.items()
    ]
    return {
        "module": MODULE,
        "moduleLabel": MODULE_LABEL,
        "difficulties": [{"key": key, "label": difficulty_label(key), "level": level(key)} for key in DIFFICULTIES],
        "topics": [
            {
                "key": tk,
                "label": data["label"],
                "subtopics": [
                    {"key": sk, "label": sv["label"], "exam": True, "difficulties": list(DIFFICULTIES)}
                    for sk, sv in data["subs"].items()
                ],
            }
            for tk, data in TOPICS.items()
        ],
        "defaults": {"topic": "agilidad_mental", "difficulty": "normal", "examCount": 10},
        "api": {
            "options": "/api/options",
            "question": "/api/question",
            "exam": "/api/exam",
            "validate": "/api/validate",
            "explain": "/api/explain",
            "grade": "/api/exam/grade",
            "pdf": "/api/exam/pdf",
        },
        "quickTopics": quick,
    }


def boolp(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "si", "yes", "on")


def intp(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _request_diff(data: dict[str, Any], default: str = "normal") -> str:
    return _normalize_difficulty(data.get("difficulty", data.get("level", None)), default)


class Handler(BaseHTTPRequestHandler):
    server_version = "Modulo0Backend/3.0"

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_pdf(self, data: bytes, name: str = "modulo0_examen.pdf"):
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or 0)
        return {} if n <= 0 else json.loads(self.rfile.read(n).decode("utf-8"))

    def query(self) -> tuple[str, dict[str, str]]:
        parsed = urllib.parse.urlparse(self.path)
        return (parsed.path.rstrip("/") or "/"), {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        try:
            path, params = self.query()
            if path in ("/", "/health", "/api/health"):
                self.send_json({"ok": True, "module": MODULE, "topics": len(TOPICS), "difficulties": list(DIFFICULTIES)})
            elif path in ("/api/options", "/api/menu", "/menu"):
                self.send_json(get_options())
            elif path == "/api/question":
                self.send_json(
                    generate_question(
                        params.get("topic"),
                        params.get("subtopic"),
                        _request_diff(params),
                        params.get("seed"),
                        boolp(params.get("include_answer"), True),
                    )
                )
            elif path in ("/api/exam", "/examen"):
                self.send_json(
                    generate_exam(
                        params.get("topic"),
                        params.get("subtopic"),
                        _request_diff(params, "normal"),
                        intp(params.get("count"), 10),
                        params.get("seed"),
                        boolp(params.get("include_answers"), False),
                    )
                )
            else:
                raise BackendError("Ruta no encontrada", 404)
        except DifficultyError as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)
        except BackendError as exc:
            self.send_json({"ok": False, "error": str(exc)}, exc.status)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, 500)

    def do_POST(self):
        try:
            path, _params = self.query()
            data = self.body()
            if path == "/api/question":
                self.send_json(
                    generate_question(
                        data.get("topic"),
                        data.get("subtopic"),
                        _request_diff(data),
                        data.get("seed"),
                        boolp(data.get("include_answer"), True),
                    )
                )
            elif path == "/api/exam":
                self.send_json(
                    generate_exam(
                        data.get("topic"),
                        data.get("subtopic"),
                        _request_diff(data, "normal"),
                        intp(data.get("count", data.get("quantity", 10)), 10),
                        data.get("seed"),
                        boolp(data.get("include_answers"), False),
                    )
                )
            elif path in ("/api/validate", "/validar"):
                token = data.get("token") or data.get("questionToken")
                if not token:
                    raise BackendError("Falta token de pregunta", 400)
                self.send_json(validate_answer(token, str(data.get("answer", data.get("respuesta", "")))))
            elif path == "/api/explain":
                token = data.get("token") or data.get("questionToken")
                if not token:
                    raise BackendError("Falta token de pregunta", 400)
                self.send_json(generate_explanation(token, str(data.get("answer", ""))))
            elif path == "/api/exam/grade":
                self.send_json(grade_exam(data.get("answers", [])))
            elif path == "/api/exam/pdf":
                exam = data.get("exam") or generate_exam(
                    data.get("topic"),
                    data.get("subtopic"),
                    _request_diff(data, "normal"),
                    intp(data.get("count"), 10),
                    include_answers=True,
                )
                self.send_pdf(pdf_bytes(exam), f"modulo0_examen_{exam.get('id', 'nuevo')}.pdf")
            else:
                raise BackendError("Ruta no encontrada", 404)
        except DifficultyError as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)
        except BackendError as exc:
            self.send_json({"ok": False, "error": str(exc)}, exc.status)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, 500)

    def log_message(self, *args):
        return


def smoke_test() -> dict[str, Any]:
    failures = []
    generated = 0
    pattern_coverage: dict[str, int] = {}
    for topic, data in TOPICS.items():
        for subtopic in data["subs"]:
            for diff in DIFFICULTIES:
                try:
                    q = generate_question(topic, subtopic, diff, include_answer=True)
                    generated += 1
                    if not validate_answer(q["token"], q["answer"]["plain"])["correcto"]:
                        failures.append(f"{topic}/{subtopic}/{diff}: no valida su respuesta")
                except Exception as exc:
                    failures.append(f"{topic}/{subtopic}/{diff}: {exc}")
                patterns = set()
                for idx in range(18):
                    try:
                        sample = generate_question(topic, subtopic, diff, seed=f"{topic}:{subtopic}:{diff}:{idx}", include_answer=True)
                        patterns.add((sample.get("metadata") or {}).get("pattern"))
                        if topic == "simplificacion" and diff == "avanzado" and not _display_has_variable(sample):
                            failures.append(f"{topic}/{subtopic}/{diff}: pregunta avanzada sin variables")
                    except Exception as exc:
                        failures.append(f"{topic}/{subtopic}/{diff}: muestreo de variedad fallo: {exc}")
                        break
                pattern_coverage[f"{topic}/{subtopic}/{diff}"] = len(patterns)
                if len(patterns) < 3:
                    failures.append(f"{topic}/{subtopic}/{diff}: variedad insuficiente ({len(patterns)} patrones)")
    exam = generate_exam("expresiones_algebraicas", difficulty="avanzado", count=8, include_answers=True)
    try:
        pdf_ok = pdf_bytes(exam).startswith(b"%PDF")
    except Exception as exc:
        pdf_ok = False
        failures.append(f"pdf: {exc}")
    return {
        "ok": not failures and pdf_ok,
        "generated": generated,
        "failures": failures,
        "examCount": exam["count"],
        "pdf": pdf_ok,
        "minPatternCoverage": min(pattern_coverage.values()) if pattern_coverage else 0,
        "patternCoverage": pattern_coverage,
    }


def run_server(host: str = "127.0.0.1", port: int = 8765):
    print(f"Modulo 0 backend listo en http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Backend HTTP para Modulo 0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test:
        print(json.dumps(smoke_test(), ensure_ascii=False, indent=2, default=str))
        return
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()