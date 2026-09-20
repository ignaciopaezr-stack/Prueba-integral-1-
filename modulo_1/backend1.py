# =============================================================================
# BACKEND UNIFICADO — BASE GENERAL DEL PROYECTO
# =============================================================================
# Este archivo une lo mejor de las dos pruebas que existian:
#   - El MOTOR de generacion/validacion/PDF de "modulo1_backend (1).py"
#     (7 temas, subtipos con dificultad easy/normal/hard, deteccion de
#     ejercicios repetidos, examenes con PDF en memoria) — se mantiene
#     practicamente intacto porque es la version mas completa y probada.
#   - La CAPA HTTP de "modulo0_backend.py" / "prueba_de_aplicacion_con_modulo1.py"
#     (http.server + ThreadingHTTPServer, manejo de CORS, respuestas JSON
#     consistentes, --smoke-test por CLI, y descarga de PDF por token de un
#     solo uso — el patron que ya usabas para que funcione en iOS Safari/PWA)
#     adaptada aqui para servir el motor de Modulo 1 sin depender de Flask.
#
# Con esto, este archivo sirve como plantilla unica para:
#   - Correrlo en local:            python backend.py --auto-port
#   - Desplegarlo en Render/Railway: python backend.py (usa $PORT si existe)
#   - Probar el motor sin servidor:  python backend.py --smoke-test
#
# El frontend correspondiente (frontend.html) ya llama exactamente a las
# rutas que este servidor expone (ver seccion "SERVIDOR HTTP" al final del
# archivo): /api/options, /api/exercise, /api/check, /api/exam, /api/exam/pdf.
# =============================================================================

from __future__ import annotations

import matplotlib
matplotlib.use('Agg') # Fuerza a Matplotlib a no usar una interfaz gráfica (modo servidor)
import matplotlib.pyplot as plt
import argparse
import base64
import io
import json
import os
import random
import re
import socket
import tempfile
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

# matplotlib.pyplot no es thread-safe (pila global de figuras); el servidor
# HTTP de este archivo es multi-hilo (ThreadingHTTPServer), asi que toda
# llamada de render pasa por este lock. Ver _render_mathtext_png().
_MATPLOTLIB_LOCK = threading.Lock()

import sympy as sp
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as rl_canvas
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.core.relational import Relational


# =============================================================================
# MODULO 1 - BACKEND  —  SEGUNDA PASADA COMPLETA
# =============================================================================
#
# [FIX-1] validate_answer/inecuaciones: .as_set() en lugar de simplify_logic.
# [FIX-2] lineal_parentesis: loop de reintento limpio (sin "right += x").
# [FIX-3] trinomio_inspeccion: leading=1 siempre (AC es un subtype separado).
# [FIX-4] agrupacion: base2 ∈ {v2, v2²} para evitar factores trinomiales.
#
# [NEW-1] Subtipos de factorizacion: trinomio_ac, suma_cubos, diferencia_cubos.
#
# [NEW-2] difficulty="normal"|"hard" en generate_exam y generate_exercise.
#
# [NEW-3] cuadratica_analisis: discriminante, eje de simetria, raices.
#
# [NEW-4] lineal_fracciones, cuadratica_radical, inecuacion_racional,
#         valor_absoluto_doble, division_sintetica (ceros hasta grado 7).
#
# [NEW-5] valor_absoluto_anidado: |a*|x+b|+c| op k  (generador no trivial).
#         valor_absoluto_mixto:   |ax+b|+|cx+d| op k  (puntos criticos distintos).
#         PARSER_LOCALS extendido con Interval, Union, oo, EmptySet, S.
#
# [NEW-6] lineal_fracciones hard: (p/q)*(x+r)=(s/t)*x+u — fracciones ambos lados.
#
# [NEW-7] conversiones hard: temperatura encadenada X→Y→Z, cifras significativas.
#
# [NEW-8] Sistema de pesos (generate_exam):
#           NORMAL: 70/25/5%, orden PROGRESIVO (easy→medium→hard).
#           HARD:   15/35/50%, orden MEZCLADO (trampa desde pregunta 1).
#
# [NEW-9] Diferencias estructurales reales hard (SEGUNDA PASADA):
#   ECUACIONES:
#     cuadratica_ambos_lados hard → trampa algebraica: coef x² casi iguales
#     en ambos lados + signo sorpresa en termino lineal.
#   FACTORIZACION:
#     factor_comun hard    → factor comun multivariable (dos variables, pot. altas).
#     agrupacion hard      → 6 terminos, 3 pares (alumno descubre la agrupacion).
#     diferencia_cuadrados hard → anidada: a⁴−b⁴ (dos aplicaciones de la identidad).
#     trinomio_cuadrado_perfecto hard → dos variables: (av+bw)².
#     trinomio_ac hard     → GCF primero, luego AC (dos pasos encadenados).
#     sustitucion hard     → u=(v+c)^n (sustitucion de binomio, forma oculta).
#     suma/diferencia_cubos hard → GCF oculto + cubos (dos pasos).
#     formula_general hard → GCF oculto + formula general (dos pasos).
#     binomio_cubo hard    → GCF oculto + binomio al cubo (dos pasos).
#     completar_cuadrado hard → coeficiente lider != ±1 obligatorio.
#     Statements: NORMAL dice el metodo; HARD dice "Factorice completamente."
#   RACIONALIZACION:
#     doble_racionalizacion (hard-only) → (p√a+q)/(r√b+s): ambos lados con
#       radicales, conjugado del denominador, simplificacion completa.
#     raiz_cuadrada_simple hard → numerador compuesto: (coef+k√v)/√v.
#     binomio_raices_cuadradas hard → solo letras simbolicas, objetivo fijo.
#     raiz_n_esima hard → indice hasta 7.
#     Statements: HARD dice "Identifique la expresion conjugada adecuada."
#   INECUACIONES:
#     Statements hard sin pista de metodo; NORMAL con instruccion explicita.
#
# =============================================================================


MAX_QUESTIONS = 50

TOPICS = {
    "conversiones": "Conversiones de unidades",
    "ecuaciones": "Ecuaciones lineales y cuadraticas",
    "factorizacion": "Factorizacion",
    "racionalizacion": "Racionalizacion",
    "inecuaciones": "Inecuaciones y valor absoluto",
}

TOPIC_TITLES = {
    "conversiones": "EXAMEN DE CONVERSIONES",
    "ecuaciones": "EXAMEN DE ECUACIONES",
    "factorizacion": "EXAMEN DE FACTORIZACION",
    "racionalizacion": "EXAMEN DE RACIONALIZACION",
    "inecuaciones": "EXAMEN DE INECUACIONES Y VALOR ABSOLUTO",
}

LETTERS = list("abcdfghjklmnpqrstuvwxyz")
X = sp.Symbol("x", real=True)

PARSER_LOCALS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "root": sp.root,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "Eq": sp.Eq,
    "Rational": sp.Rational,
    "pi": sp.pi,
    # Conjuntos — necesarios para validar soluciones de valor_absoluto_mixto / anidado
    "Interval": sp.Interval,
    "Union": sp.Union,
    "Intersection": sp.Intersection,
    "oo": sp.oo,
    "EmptySet": sp.EmptySet,
    "S": sp.S,
}
PARSER_LOCALS.update({letter: sp.Symbol(letter, real=True) for letter in LETTERS + ["x", "y", "z"]})

POSITIVE_LOCALS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "root": sp.root,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "Eq": sp.Eq,
    "Rational": sp.Rational,
    "pi": sp.pi,
}
POSITIVE_LOCALS.update({letter: sp.Symbol(letter, positive=True) for letter in LETTERS + ["x", "y", "z"]})

PARSER_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================


def _make_rng(seed: int | str | None = None) -> random.Random:
    return random.Random(seed)


def _weighted_choice(rng: random.Random, pool: list, weights: list[float]) -> Any:
    """Selecciona un elemento con probabilidad ponderada.
    
    pool y weights deben tener la misma longitud.
    Permite distribuciones no uniformes entre subtipos (ver NORMAL vs HARD).
    """
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(pool, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return pool[-1]


def _weighted_subtype(rng: random.Random, topic: str, hard: bool) -> str:
    """Elige un subtipo usando pesos segun dificultad.
    
    NORMAL: 70% facil / 25% media / 5% dificil
    HARD:   15% facil / 35% media / 50% dificil

    Subtipos marcados como hard-exclusivos (en HARD_SUBTYPES_BY_TOPIC pero NO
    en SUBTYPES_BY_TOPIC) jamas aparecen en modo normal, ni siquiera con el
    5% de peso asignado a la categoria 'hard'.
    """
    cats = DIFFICULTY_CATEGORIES.get(topic)
    if cats is None:
        pool = HARD_SUBTYPES_BY_TOPIC[topic] if hard else SUBTYPES_BY_TOPIC[topic]
        return rng.choice(pool)

    easy_pool   = cats["easy"]
    medium_pool = cats["medium"]
    hard_pool   = cats["hard"]

    # En modo normal, filtrar subtipos que solo existen en el pool hard
    if not hard:
        normal_all = set(SUBTYPES_BY_TOPIC.get(topic, []))
        hard_pool = [s for s in hard_pool if s in normal_all]
        if not hard_pool:
            # Si todos los hard son exclusivos, redirigir ese peso a medium
            hard_pool = medium_pool

    if hard:
        weights = [0.15, 0.35, 0.50]
    else:
        weights = [0.70, 0.25, 0.05]

    tier = _weighted_choice(rng, ["easy", "medium", "hard"], weights)
    tier_pool = {"easy": easy_pool, "medium": medium_pool, "hard": hard_pool}[tier]
    return rng.choice(tier_pool)


def _limit_quantity(quantity: int, maximum: int = MAX_QUESTIONS) -> int:
    try:
        quantity = int(quantity)
    except Exception as exc:
        raise ValueError("La cantidad debe ser un entero.") from exc
    if quantity <= 0:
        raise ValueError("La cantidad debe ser positiva.")
    return min(quantity, maximum)


def _parse_math(text: str, positive_symbols: bool = False) -> Any:
    text = str(text).strip()
    text = text.replace("^", "**")
    text = text.replace(" raiz ", " sqrt ")
    text = re.sub(r"\braiz\(([^)]+)\)", r"sqrt(\1)", text)
    text = re.sub(r"(\d+)\s*raiz\(([^)]+)\)", r"\1*sqrt(\2)", text)
    text = text.replace(" o ", " | ").replace(" y ", " & ")
    text = text.replace("∞", "oo").replace("−", "-")
    text = re.sub(r"\bU\b", "Union", text)
    locals_map = POSITIVE_LOCALS if positive_symbols else PARSER_LOCALS
    return parse_expr(
        text,
        local_dict=locals_map,
        transformations=PARSER_TRANSFORMATIONS,
        evaluate=True,
    )


def _parse_math_shape(text: str, positive_symbols: bool = False) -> Any:
    text = str(text).strip()
    text = text.replace("^", "**")
    text = re.sub(r"\braiz\(([^)]+)\)", r"sqrt(\1)", text)
    text = text.replace("∞", "oo").replace("−", "-")
    locals_map = POSITIVE_LOCALS if positive_symbols else PARSER_LOCALS
    return parse_expr(
        text,
        local_dict=locals_map,
        transformations=PARSER_TRANSFORMATIONS,
        evaluate=False,
    )


def _math_equal(left: Any, right: Any) -> bool:
    try:
        return sp.simplify(left - right) == 0
    except Exception:
        try:
            return bool(sp.simplify(left == right))
        except Exception:
            return False


def _plain(value: Any) -> str:
    if value is True:
        return "Todos los numeros reales"
    if value is False:
        return "Sin solucion"
    if value == "none":
        return "Sin solucion real"
    return str(value)


def _latex(value: Any) -> str:
    if value is True:
        return r"\mathbb{R}"
    if value is False or value == "none":
        return r"\varnothing"
    try:
        return sp.latex(value)
    except Exception:
        return str(value)


def _display_payload(kind: str, value: Any) -> dict[str, Any]:
    if kind == "math":
        return {"kind": "math", "plain": _plain(value), "latex": _latex(value)}
    return {"kind": "text", "plain": str(value), "latex": None}


def _answer_payload(value: Any, label: str = "Respuesta correcta") -> dict[str, Any]:
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        return {
            "label": label,
            "kind": "list",
            "plain": [_plain(v) for v in values],
            "latex": [_latex(v) for v in values],
        }
    return {
        "label": label,
        "kind": "math" if not isinstance(value, str) else "text",
        "plain": _plain(value),
        "latex": _latex(value) if not isinstance(value, str) else None,
    }


def _new_exercise(
    topic: str,
    subtype: str,
    statement: str,
    display_kind: str,
    display_value: Any,
    answer_value: Any,
    validation: dict[str, Any],
    *,
    answer_label: str = "Respuesta correcta",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "topic": topic,
        "topic_label": TOPICS[topic],
        "subtype": subtype,
        "statement": statement,
        "display": _display_payload(display_kind, display_value),
        "answer": _answer_payload(answer_value, answer_label),
        "validation": validation,
        "metadata": metadata or {},
    }


def _sorted_solutions(values: list[Any]) -> list[Any]:
    return sorted([sp.simplify(v) for v in values], key=lambda item: (str(type(item)), str(item)))


def _same_solution_set(user_values: list[Any], expected_values: list[Any]) -> bool:
    if len(user_values) != len(expected_values):
        return False

    unmatched = list(expected_values)
    for user_value in user_values:
        for index, expected in enumerate(unmatched):
            if _math_equal(user_value, expected):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def _parse_solution_list(text: str, positive_symbols: bool = False) -> list[Any]:
    raw = str(text).strip()
    if raw.lower() in {"", "none", "sin solucion", "sin solucion real", "[]", "{}"}:
        return []

    try:
        parsed = _parse_math(raw, positive_symbols=positive_symbols)
        if isinstance(parsed, (list, tuple, set, sp.FiniteSet)):
            return list(parsed)
        if isinstance(parsed, sp.Equality):
            return [parsed.rhs]
    except Exception:
        pass

    parts = [part.strip() for part in raw.strip("[]{}()").split(",") if part.strip()]
    return [_parse_math(part, positive_symbols=positive_symbols) for part in parts]


def _has_radical(expr: Any) -> bool:
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_Rational and node.exp.q != 1:
            return True
    return False


def _target_is_rationalized(expr: Any, objective: str, *, preserve_shape: bool = False) -> bool:
    num, den = sp.fraction(expr if preserve_shape else sp.together(expr))
    target = den if objective == "denominador" else num
    return not _has_radical(target)


# =============================================================================
# CONVERSIONES
# =============================================================================


UNIT_TABLES = {
    "distancia_metrica": {"km": 1000, "m": 1, "dm": 0.1, "cm": 0.01, "mm": 0.001},
    "sistema_ingles": {"in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34},
    "volumen": {"m3": 1000, "L": 1, "dL": 0.1, "cL": 0.01, "mL": 0.001, "cm3": 0.001, "gal": 3.78541},
    "masa": {"kg": 1, "g": 0.001, "mg": 0.000001, "lb": 0.45359237, "oz": 0.0283495},
    "tiempo": {"s": 1, "min": 60, "h": 3600, "dia": 86400, "semana": 604800},
    "area": {"km2": 1_000_000, "ha": 10_000, "m2": 1, "dm2": 0.01, "cm2": 0.0001, "mm2": 0.000001},
    "velocidad": {"m/s": 1, "km/h": 1 / 3.6, "ft/s": 0.3048, "mi/h": 0.44704},
    "notacion_cientifica": {
        "T": 10**12,
        "G": 10**9,
        "M": 10**6,
        "k": 10**3,
        "unidad": 1,
        "m": 10**-3,
        "u": 10**-6,
        "n": 10**-9,
        "p": 10**-12,
    },
}

# Nombres completos de cada unidad para enunciados legibles.
# Sin este mapa, el statement mostraria "Convierta 903 u a unidad" sin contexto.
UNIT_DISPLAY_NAMES: dict[str, str] = {
    # distancia_metrica
    "km": "kilómetros (km)", "m": "metros (m)", "dm": "decímetros (dm)",
    "cm": "centímetros (cm)", "mm": "milímetros (mm)",
    # sistema_ingles
    "in": "pulgadas (in)", "ft": "pies (ft)", "yd": "yardas (yd)", "mi": "millas (mi)",
    # volumen
    "m3": "metros cúbicos (m³)", "L": "litros (L)", "dL": "decilitros (dL)",
    "cL": "centilitros (cL)", "mL": "mililitros (mL)", "cm3": "centímetros cúbicos (cm³)",
    "gal": "galones (gal)",
    # masa
    "kg": "kilogramos (kg)", "g": "gramos (g)", "mg": "miligramos (mg)",
    "lb": "libras (lb)", "oz": "onzas (oz)",
    # tiempo
    "s": "segundos (s)", "min": "minutos (min)", "h": "horas (h)",
    "dia": "días (día)", "semana": "semanas (semana)",
    # area
    "km2": "kilómetros cuadrados (km²)", "ha": "hectáreas (ha)",
    "m2": "metros cuadrados (m²)", "dm2": "decímetros cuadrados (dm²)",
    "cm2": "centímetros cuadrados (cm²)", "mm2": "milímetros cuadrados (mm²)",
    # velocidad
    "m/s": "metros por segundo (m/s)", "km/h": "kilómetros por hora (km/h)",
    "ft/s": "pies por segundo (ft/s)", "mi/h": "millas por hora (mi/h)",
    # notacion_cientifica — prefijos SI con nombre completo
    "T": "teras (T, ×10¹²)", "G": "gigas (G, ×10⁹)", "M": "megas (M, ×10⁶)",
    "k": "kilos (k, ×10³)", "unidad": "unidad base",
    "m": "milos (m, ×10⁻³)", "u": "micros (μ, ×10⁻⁶)",
    "n": "nanos (n, ×10⁻⁹)", "p": "picos (p, ×10⁻¹²)",
}


def _unit_label(unit_key: str) -> str:
    """Devuelve el nombre legible de una unidad, o el clave original si no hay mapa."""
    return UNIT_DISPLAY_NAMES.get(unit_key, unit_key)

CONVERSION_SUBTYPES = ["temperatura", *UNIT_TABLES.keys()]

# [NEW-2] Subtipos mas exigentes para modo examen: unidades menos intuitivas.
HARD_CONVERSION_SUBTYPES = ["area", "velocidad", "notacion_cientifica", "sistema_ingles", "volumen", "masa"]

# Categorias para seleccion ponderada (Paso 8 — sistema de pesos)
CONVERSION_DIFFICULTY_CATEGORIES = {
    "easy":   ["temperatura", "distancia_metrica", "tiempo"],
    "medium": ["volumen", "masa", "area"],
    "hard":   ["velocidad", "notacion_cientifica", "sistema_ingles"],
}


def convert_units(value: float, origin: str, target: str, table: dict[str, float]) -> float:
    return value * table[origin] / table[target]


def convert_temperature(value: float, origin: str, target: str) -> float:
    if origin == target:
        return value
    conversions: dict[tuple[str, str], Any] = {
        ("C", "F"): lambda v: v * 1.8 + 32,
        ("C", "K"): lambda v: v + 273.15,
        ("F", "C"): lambda v: (v - 32) / 1.8,
        ("F", "K"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("K", "C"): lambda v: v - 273.15,
        ("K", "F"): lambda v: (v - 273.15) * 1.8 + 32,
    }
    return conversions[(origin, target)](value)


def generate_conversion_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
) -> dict[str, Any]:
    # [NEW-2] En modo hard se prefieren subtipos menos intuitivos.
    default_pool = HARD_CONVERSION_SUBTYPES if hard else CONVERSION_SUBTYPES
    subtype = subtype or rng.choice(default_pool)
    if subtype not in CONVERSION_SUBTYPES:
        raise ValueError(f"Subtipo de conversion desconocido: {subtype}")

    if subtype == "temperatura":
        units = ["C", "F", "K"]
        origin = rng.choice(units)
        target = rng.choice([u for u in units if u != origin])
        # [HARD] Valores con decimales o rangos mas extremos; conversion doble (X→Y→Z).
        if hard:
            value = round(rng.uniform(-60 if origin == "C" else 0, 300), 1)
            # Conversion encadenada (trampa): convertir a intermediario primero
            intermediate = rng.choice([u for u in units if u not in (origin, target)])
            inter_val = round(convert_temperature(value, origin, intermediate), 2)
            answer = round(convert_temperature(inter_val, intermediate, target), 2)
            statement = (
                f"Convierta {value} °{origin} a °{intermediate}, "
                f"y el resultado conviertalo a °{target}."
            )
        else:
            value = rng.randint(-40, 120) if origin == "C" else rng.randint(0, 420)
            answer = round(convert_temperature(value, origin, target), 2)
            statement = f"Convierta {value} grados {origin} a grados {target}."
    else:
        table = UNIT_TABLES[subtype]
        units = list(table.keys())
        origin = rng.choice(units)
        target = rng.choice([u for u in units if u != origin])
        origin_label = _unit_label(origin)
        target_label = _unit_label(target)
        # [HARD] Magnitudes extremas, decimales finos, o conversion con escala inversa.
        if hard:
            value = rng.choice([
                rng.randint(1, 9999),
                round(rng.uniform(0.0001, 0.999), 5),
                round(rng.uniform(100, 50000), 3),
            ])
            sig_figs = rng.choice([3, 4, 5])
            answer = round(convert_units(value, origin, target, table), 6)
            statement = (
                f"Convierta {value} {origin_label} a {target_label}. "
                f"Exprese la respuesta con {sig_figs} cifras significativas."
            )
        else:
            value = rng.choice([rng.randint(1, 900), round(rng.uniform(1, 250), 2)])
            answer = round(convert_units(value, origin, target, table), 6)
            statement = f"Convierta {value} {origin_label} a {target_label}."

    return _new_exercise(
        "conversiones",
        subtype,
        statement,
        "text",
        statement,
        f"{answer} {_unit_label(target)}",
        {
            "type": "numeric",
            "answer": answer,
            "tolerance": max(0.01, abs(answer) * 0.0005),
        },
        answer_label="Conversion correcta",
        metadata={"origin": origin, "target": target, "value": value, "hard": hard},
    )


# =============================================================================
# ECUACIONES
# =============================================================================


EQUATION_SUBTYPES = [
    "lineal_basica",
    "lineal_parentesis",
    "lineal_fracciones",          # [NEW-4] Coeficientes racionales no enteros
    "cuadratica_raices_enteras",
    "cuadratica_raices_racionales",
    "cuadratica_sin_reales",
    "cuadratica_ambos_lados",
    "cuadratica_analisis",        # [NEW-3] Discriminante, eje o raíces
    "cuadratica_radical",         # [NEW-4] sqrt(ax+b) = c  (hard exclusivo)
    "sistemas_ecuaciones",
]

# [NEW-2] Para examen: subtipos que requieren mas pasos o discriminante no trivial.
HARD_EQUATION_SUBTYPES = [
    "lineal_parentesis",
    "lineal_fracciones",          # [NEW-4]
    "cuadratica_raices_racionales",
    "cuadratica_sin_reales",
    "cuadratica_ambos_lados",
    "cuadratica_analisis",
    "cuadratica_radical",         # [NEW-4]
    "sistemas_ecuaciones",
]

# Categorias para seleccion ponderada (Paso 8)
EQUATION_DIFFICULTY_CATEGORIES = {
    "easy":   ["lineal_basica"],
    "medium": ["lineal_parentesis", "cuadratica_raices_enteras"],
    "hard":   ["lineal_fracciones", "cuadratica_raices_racionales",
               "cuadratica_sin_reales", "cuadratica_ambos_lados",
               "cuadratica_analisis", "cuadratica_radical"],
}


def _nonzero_int(rng: random.Random, start: int, stop: int) -> int:
    value = 0
    while value == 0:
        value = rng.randint(start, stop)
    return value


def generate_equation_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
) -> dict[str, Any]:
    default_pool = HARD_EQUATION_SUBTYPES if hard else EQUATION_SUBTYPES
    subtype = subtype or rng.choice(default_pool)
    if subtype not in EQUATION_SUBTYPES:
        raise ValueError(f"Subtipo de ecuacion desconocido: {subtype}")

    x = X

    # [NEW-2] Rangos de coeficientes segun dificultad.
    coef_range = (-20, 20) if hard else (-12, 12)
    const_range = (-50, 50) if hard else (-25, 25)

    # ── [NEW-4] lineal_fracciones ──────────────────────────────────────
    if subtype == "lineal_fracciones":
        # NORMAL: (p/q)*x + (r/s) = t   —  un solo término racional, forma directa
        # HARD:   (p/q)*(x + r) = (s/t)*x + u  — fracciones en ambos lados + paréntesis
        for _ in range(40):
            p = _nonzero_int(rng, -10 if not hard else -15, 10 if not hard else 15)
            q = rng.randint(2, 8 if not hard else 14)
            r = rng.randint(-8 if not hard else -15, 8 if not hard else 15)
            s = rng.randint(2, 6 if not hard else 10)
            t = rng.randint(-12 if not hard else -25, 12 if not hard else 25)
            coef = sp.Rational(p, q)
            const = sp.Rational(r, s)
            if coef.q == 1:          # fraccion reducida a entero → saltear
                continue
            if hard:
                # Forma: (p/q)*(x + r) = (s2/t2)*x + u  — mas pasos, dos fracciones
                s2 = rng.randint(1, 5)
                t2 = rng.randint(2, 6)
                u  = rng.randint(-10, 10)
                coef2 = sp.Rational(s2, t2)
                if coef2.q == 1:
                    continue
                lhs = coef * (x + r)
                rhs = coef2 * x + u
                eq = sp.Eq(lhs, rhs)
                sol = sp.solve(eq, x)
                if sol and sol[0].is_Rational and not sol[0].is_Integer:
                    break
            else:
                lhs = coef * x + const
                eq = sp.Eq(lhs, t)
                sol = sp.solve(eq, x)
                if sol:
                    break
        else:
            if hard:
                eq = sp.Eq(sp.Rational(3, 4) * (x + 2), sp.Rational(1, 2) * x - 1)
            else:
                eq = sp.Eq(sp.Rational(3, 4) * x - sp.Rational(1, 2), sp.Integer(5))
            sol = sp.solve(eq, x)
        solutions = _sorted_solutions(sol)
        stmt = (
            "Resuelva la ecuacion con fracciones en ambos lados. Despeje x."
            if hard else
            "Resuelva la ecuacion con coeficientes racionales."
        )
        return _new_exercise(
            "ecuaciones", subtype, stmt,
            "math", eq, solutions,
            {"type": "solution_set", "variable": "x", "solutions": [str(s) for s in solutions]},
            answer_label="Soluciones",
            metadata={"hard": hard},
        )

    # ── [NEW-4] cuadratica_radical ─────────────────────────────────────
    if subtype == "cuadratica_radical":
        # Forma: sqrt(a*x + b) = c  (c >= 1).
        # Solucion: x = (c^2 - b) / a  (siempre satisface a*sol+b = c^2 >= 0).
        for _ in range(50):
            c = rng.randint(1, 8 if not hard else 15)
            a = _nonzero_int(rng, -12 if not hard else -20, 12 if not hard else 20)
            b = rng.randint(-25 if not hard else -50, 25 if not hard else 50)
            sol_expr = sp.Rational(c ** 2 - b, a)
            # Verificar que el radicando en la solucion sea exactamente c^2 (no extraneous)
            if (a * sol_expr + b) == c ** 2:
                break
        else:
            a, b, c = 3, -3, 3   # 3x-3=9 → x=4
            sol_expr = sp.Integer(4)
        display_eq = sp.Eq(sp.sqrt(a * x + b), c)
        solutions = [sol_expr]
        return _new_exercise(
            "ecuaciones", subtype,
            "Resuelva la ecuacion radical (verifique que la solucion satisfaga el dominio).",
            "math", display_eq, solutions,
            {"type": "solution_set", "variable": "x", "solutions": [str(s) for s in solutions]},
            answer_label="Soluciones",
            metadata={"hard": hard},
        )

    if subtype == "lineal_basica":
        a = _nonzero_int(rng, *coef_range)
        d = _nonzero_int(rng, *coef_range)
        while a == d:
            d = _nonzero_int(rng, *coef_range)
        b = rng.randint(*const_range)
        e = rng.randint(*const_range)
        left, right = a * x + b, d * x + e

    elif subtype == "lineal_parentesis":
        # [FIX-2] Sustituido el parche "right += x" por un loop de reintento.
        # Se garantiza que el coeficiente de x no se cancele sin alterar la
        # estructura algebraica del ejercicio.
        for _ in range(30):
            a = _nonzero_int(rng, -8 if not hard else -15, 8 if not hard else 15)
            b = rng.randint(-10 if not hard else -20, 10 if not hard else 20)
            c = _nonzero_int(rng, -8 if not hard else -15, 8 if not hard else 15)
            d = rng.randint(-10 if not hard else -20, 10 if not hard else 20)
            e = rng.randint(-20 if not hard else -40, 20 if not hard else 40)
            f = rng.randint(-20 if not hard else -40, 20 if not hard else 40)
            left = a * (x + b) + e
            right = c * (x + d) + f
            # Rechazar si los terminos en x se cancelan (identidad o sin solucion trivial).
            if sp.expand(left - right).coeff(x) != 0:
                break
        else:
            # Fallback garantizado: ecuacion sencilla con coeficientes distintos.
            left, right = 3 * x + 5, x - 7

    elif subtype == "cuadratica_raices_enteras":
        pool = [n for n in range(-9 if not hard else -15, 10 if not hard else 16) if n != 0]
        r1, r2 = rng.sample(pool, 2)
        a = _nonzero_int(rng, -5 if not hard else -8, 5 if not hard else 8)
        left = sp.expand(a * (x - r1) * (x - r2))
        right = 0

    elif subtype == "cuadratica_raices_racionales":
        # [NEW-2] Hard: denominadores y numeradores mas grandes.
        num_range = (-15, 15) if hard else (-8, 8)
        den_range = (2, 9) if hard else (2, 7)
        r1 = sp.Rational(rng.randint(*num_range), rng.randint(*den_range))
        r2 = sp.Rational(rng.randint(*num_range), rng.randint(*den_range))
        while r1 == r2:
            r2 = sp.Rational(rng.randint(*num_range), rng.randint(*den_range))
        a = _nonzero_int(rng, 1, 6 if not hard else 10)
        left = sp.expand(a * (x - r1) * (x - r2))
        right = 0

    elif subtype == "cuadratica_sin_reales":
        # Nota: a > 0 (rango 1..8) y k > 0 (rango 1..12) garantizan que
        # a*(x-h)^2 + k > 0 para todo x real => sin raices reales. Correcto.
        a = rng.randint(1, 8 if not hard else 15)   # siempre positivo
        h = rng.randint(-8 if not hard else -15, 8 if not hard else 15)
        k = rng.randint(1, 12 if not hard else 25)  # siempre positivo
        left = sp.expand(a * (x - h) ** 2 + k)
        right = 0

    else:  # cuadratica_ambos_lados
        if not hard:
            # NORMAL: ambos lados simples, diferencia clara de coeficientes
            pool = [n for n in range(-8, 9) if n != 0]
            r1, r2 = rng.sample(pool, 2)
            a = _nonzero_int(rng, -4, 4)
            diff = sp.expand(a * (x - r1) * (x - r2))
            right = (
                rng.randint(-4, 4) * x**2
                + rng.randint(-10, 10) * x
                + rng.randint(-12, 12)
            )
            left = sp.expand(right + diff)
        else:
            # HARD: trampa algebraica — se construyen ambos lados para que:
            # 1. El coeficiente de x² sea casi igual en ambos lados (tentación de cancelar mal)
            # 2. El coeficiente del término lineal tenga signo sorpresa
            # 3. Se necesite pasar todos los términos y reordenar antes de factorizar
            for _ in range(50):
                # Construcción: left = A*x^2 + B*x + C, right = D*x^2 + E*x + F
                # donde A-D != 0 (no se cancela x²) pero |A-D| es pequeño (trampa)
                A = rng.randint(2, 8)
                D = rng.randint(1, A - 1) if A > 1 else 1  # D < A pero cercano
                # Coeficientes con signos opuestos en x (trampa de signo)
                B = rng.randint(-20, -5)
                E = rng.randint(5, 20)
                C = rng.randint(-30, 30)
                F = rng.randint(-30, 30)
                left  = A * x**2 + B * x + C
                right = D * x**2 + E * x + F
                eq_test = sp.Eq(left, right)
                sols = sp.solve(eq_test, x)
                real_sols = [s for s in sols if sp.im(s) == 0]
                if real_sols:
                    break
            else:
                # Fallback garantizado
                left  = 3 * x**2 - 7 * x + 2
                right = x**2 + 3 * x - 10

    # [NEW-3] cuadratica_analisis: pregunta sobre propiedades de la cuadratica.
    if subtype == "cuadratica_analisis":
        # Genera una cuadratica con raices reales garantizadas (para discriminante/raices).
        for _ in range(60):
            a_c = _nonzero_int(rng, -8 if hard else -6, 8 if hard else 6)
            b_c = rng.randint(-18 if hard else -12, 18 if hard else 12)
            c_c = rng.randint(-25 if hard else -15, 25 if hard else 15)
            disc_val = b_c ** 2 - 4 * a_c * c_c
            if disc_val >= 0:
                break
        else:
            a_c, b_c, c_c = 1, -5, 6
            disc_val = 1

        quad_expr = a_c * x ** 2 + b_c * x + c_c
        equation_q = sp.Eq(quad_expr, 0)
        disc_sym = sp.Integer(disc_val)
        axis_sym = sp.Rational(-b_c, 2 * a_c)
        real_sols = _sorted_solutions(sp.solve(equation_q, x))

        prop = rng.choice(["discriminante", "eje_simetria", "raices"])

        if prop == "discriminante":
            return _new_exercise(
                "ecuaciones", "cuadratica_analisis",
                "Calcula el discriminante (D = b^2 - 4ac) de la ecuacion cuadratica.",
                "math", equation_q, disc_sym,
                {"type": "math_equal", "answer_expr": str(disc_sym)},
                answer_label="Discriminante",
                metadata={"hard": hard, "property": "discriminante"},
            )
        elif prop == "eje_simetria":
            return _new_exercise(
                "ecuaciones", "cuadratica_analisis",
                "Halla el eje de simetria (x = -b / 2a) de la parabola.",
                "math", equation_q, axis_sym,
                {"type": "math_equal", "answer_expr": str(axis_sym)},
                answer_label="Eje de simetria",
                metadata={"hard": hard, "property": "eje_simetria"},
            )
        else:
            return _new_exercise(
                "ecuaciones", "cuadratica_analisis",
                "Encuentra las raices reales x1 y x2 de la ecuacion cuadratica.",
                "math", equation_q, real_sols,
                {"type": "solution_set", "variable": "x", "solutions": [str(s) for s in real_sols]},
                answer_label="Raices reales (x1, x2)",
                metadata={"hard": hard, "property": "raices"},
            )

    equation = sp.Eq(left, right)
    solutions = _sorted_solutions(sp.solve(equation, x))
    real_solutions = [sol for sol in solutions if sp.im(sol).simplify() == 0]
    if subtype == "cuadratica_sin_reales":
        real_solutions = []

    # HARD: enunciado sin pista de metodo (el alumno debe descubrirlo)
    if hard and subtype == "cuadratica_ambos_lados":
        stmt = "Resuelva la ecuacion. Pase todos los terminos a un lado antes de factorizar."
    elif hard:
        stmt = "Resuelva la ecuacion."
    else:
        stmt = "Resuelva la ecuacion."

    return _new_exercise(
        "ecuaciones",
        subtype,
        stmt,
        "math",
        equation,
        real_solutions,
        {
            "type": "solution_set",
            "variable": "x",
            "solutions": [str(sol) for sol in real_solutions],
        },
        answer_label="Soluciones",
        metadata={"hard": hard},
    )


# =============================================================================
# FACTORIZACION
# =============================================================================


# [NEW-1] Agregados trinomio_ac, suma_cubos y diferencia_cubos.
FACTORIZATION_SUBTYPES = [
    "factor_comun",
    "agrupacion",
    "diferencia_cuadrados",
    "trinomio_cuadrado_perfecto",
    "trinomio_inspeccion",
    "trinomio_ac",          # [NEW-1] Metodo AC (coeficiente lider != 1)
    "sustitucion",
    "suma_cubos",           # [NEW-1] a^3 + b^3
    "diferencia_cubos",     # [NEW-1] a^3 - b^3
    "binomio_cubo",         # [NEW-4] (a+b)^3 o (a-b)^3
    "formula_general",
    "division_sintetica",
    "completar_cuadrado",
]

# [NEW-2] Subtipos mas exigentes para modo examen.
HARD_FACTORIZATION_SUBTYPES = [
    "trinomio_ac",
    "sustitucion",
    "suma_cubos",
    "diferencia_cubos",
    "binomio_cubo",         # [NEW-4]
    "formula_general",
    "division_sintetica",
    "completar_cuadrado",
]

# Categorias para seleccion ponderada (Paso 8)
FACTORIZATION_DIFFICULTY_CATEGORIES = {
    "easy":   ["factor_comun", "diferencia_cuadrados",
               "trinomio_cuadrado_perfecto", "trinomio_inspeccion"],
    "medium": ["agrupacion", "trinomio_ac", "suma_cubos",
               "diferencia_cubos", "binomio_cubo"],
    "hard":   ["sustitucion", "formula_general",
               "division_sintetica", "completar_cuadrado"],
}

PRODUCT_NOTABLE_NAMES = [
    "Diferencia de cuadrados",
    "Trinomio cuadrado perfecto",
    "Suma de cubos",
    "Diferencia de cubos",
]

# Subtipos de productos notables que admiten ejercicios de EXPANSION (direction="expandir").
# Todos los demas subtipos solo factorizan; en modo examen y modo aleatorio SIEMPRE se reduce.
EXPANDABLE_SUBTYPES: set[str] = {
    "diferencia_cuadrados",
    "trinomio_cuadrado_perfecto",
    "suma_cubos",
    "diferencia_cubos",
    "binomio_cubo",
}


def _random_symbol(rng: random.Random) -> sp.Symbol:
    return sp.Symbol(rng.choice(LETTERS), real=True)


def _random_symbols(rng: random.Random, count: int) -> list[sp.Symbol]:
    return [sp.Symbol(letter, real=True) for letter in rng.sample(LETTERS, count)]


def _nonzero_values(rng: random.Random, count: int, start: int = -20, stop: int = 20) -> list[int]:
    values = []
    while len(values) < count:
        value = rng.randint(start, stop)
        if value != 0:
            values.append(value)
    return values


def generate_factorization_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
    direction: str | None = None,   # "expandir" | "reducir" | None (aleatorio)
) -> dict[str, Any]:
    """Genera un ejercicio de factorizacion o productos notables.

    direction:
      "reducir"  — dado el polinomio expandido, pedir la forma factorizada (default).
      "expandir" — dado la forma factorizada, pedir el polinomio expandido.
      None       — se elige aleatoriamente; solo los subtipos de productos notables
                   tienen sentido en modo expandir (diferencia_cuadrados,
                   trinomio_cuadrado_perfecto, suma_cubos, diferencia_cubos, binomio_cubo).
    """
    default_pool = HARD_FACTORIZATION_SUBTYPES if hard else FACTORIZATION_SUBTYPES
    subtype = subtype or rng.choice(default_pool)
    if subtype not in FACTORIZATION_SUBTYPES:
        raise ValueError(f"Subtipo de factorizacion desconocido: {subtype}")

    # Subtipos que tienen sentido en modo "expandir" (productos notables con identidad clara)
    EXPANDABLE = EXPANDABLE_SUBTYPES

    # Elegir direccion si no se especifico:
    # Por defecto SIEMPRE se factoriza. "expandir" debe pedirse explicitamente.
    # (El modo aleatorio y el modo examen jamas expandiran.)
    if direction is None:
        direction = "reducir"

    # Si se pidio expandir en un subtipo no expandable, forzar reducir
    if direction == "expandir" and subtype not in EXPANDABLE:
        direction = "reducir"

    allow_additive = False

    # [NEW-2] Rangos segun dificultad.
    coef_max = 20 if hard else 12
    root_max = 12 if hard else 9

    # ── Paso 10: statements NORMAL vs HARD ─────────────────────────────
    # NORMAL: enuncia el metodo Y la identidad para que el alumno la reconozca.
    # HARD:   solo dice "Factorice completamente" — el alumno debe descubrir el metodo.
    STMT_HINTS = {
        "factor_comun":
            "Extraiga el factor comun maximo (MCF).",
        "agrupacion":
            "Factorice agrupando terminos en pares con factor comun.",
        "diferencia_cuadrados":
            "Diferencia de cuadrados: a² − b² = (a+b)(a−b).",
        "trinomio_cuadrado_perfecto":
            "Trinomio cuadrado perfecto: a²±2ab+b² = (a±b)².",
        "trinomio_inspeccion":
            "Factorice el trinomio por inspeccion (FOIL inverso): x²+bx+c = (x+r₁)(x+r₂).",
        "trinomio_ac":
            "Metodo AC: multiplique a·c, encuentre factores que sumen b, luego agrupe.",
        "sustitucion":
            "Factorice usando sustitucion (u = expresion): identifique el patron oculto.",
        "suma_cubos":
            "Suma de cubos: a³+b³ = (a+b)(a²−ab+b²).",
        "diferencia_cubos":
            "Diferencia de cubos: a³−b³ = (a−b)(a²+ab+b²).",
        "binomio_cubo":
            "Binomio al cubo: (a+b)³ = a³+3a²b+3ab²+b³  |  (a−b)³ = a³−3a²b+3ab²−b³.",
        "formula_general":
            "Formula general: x = (−b ± √(b²−4ac)) / 2a.",
        "division_sintetica":
            "Division sintetica: divide el polinomio entre (x−r) para encontrar ceros y factores.",
        "completar_cuadrado":
            "Completar cuadrado: ax²+bx+c = a(x+h)²+k  donde h = b/(2a).",
    }

    # Variantes de enunciado para modo expandir
    STMT_EXPAND = {
        "diferencia_cuadrados":
            "Expanda usando la identidad Diferencia de cuadrados: (a+b)(a−b) = a²−b².",
        "trinomio_cuadrado_perfecto":
            "Expanda usando Trinomio cuadrado perfecto: (a±b)² = a²±2ab+b².",
        "suma_cubos":
            "Expanda usando Suma de cubos: (a+b)(a²−ab+b²) = a³+b³.",
        "diferencia_cubos":
            "Expanda usando Diferencia de cubos: (a−b)(a²+ab+b²) = a³−b³.",
        "binomio_cubo":
            "Expanda el Binomio al cubo: (a+b)³ = a³+3a²b+3ab²+b³  |  (a−b)³ = a³−3a²b+3ab²−b³.",
    }

    if direction == "expandir":
        stmt_normal = STMT_EXPAND.get(subtype, "Expanda la expresion.")
        stmt_hard   = "Expanda completamente la expresion aplicando la identidad correspondiente."
    else:
        stmt_normal = STMT_HINTS.get(subtype, "Factorice o reescriba segun el metodo indicado.")
        stmt_hard   = "Factorice completamente. Determine el metodo mas adecuado."

    if subtype == "factor_comun":
        if not hard:
            # NORMAL: un factor comun numerico simple + variable
            term_count = rng.randint(2, 4)
            common_number = rng.choice([2, 3, 4, 5, 6, 7, 10])
            v = _random_symbol(rng)
            common_power = rng.randint(1, 3)
            terms = []
            for coef in _nonzero_values(rng, term_count, -coef_max, coef_max):
                power = common_power + rng.randint(0, 3)
                terms.append(coef * v ** (power - common_power))
            answer = common_number * v**common_power * sp.Add(*terms)
            question = sp.expand(answer)
            answer = sp.factor(question)
        else:
            # HARD: factor comun multivariable con potencias altas + coeficiente no obvio.
            # Dos variables, factor comun incluye ambas (trampa: hay que buscar el MCF en 2 vars)
            v1, v2 = _random_symbols(rng, 2)
            common_number = rng.choice([2, 3, 5, 6, 7, 10, 12])
            p1 = rng.randint(2, 4)   # potencia comun de v1
            p2 = rng.randint(1, 3)   # potencia comun de v2
            term_count = rng.randint(3, 5)
            terms = []
            for coef in _nonzero_values(rng, term_count, -coef_max, coef_max):
                extra1 = rng.randint(0, 3)
                extra2 = rng.randint(0, 2)
                terms.append(coef * v1**extra1 * v2**extra2)
            # Multiplica por el factor comun para construir el polinomio
            question = sp.expand(common_number * v1**p1 * v2**p2 * sp.Add(*terms))
            answer = sp.factor(question)

    elif subtype == "agrupacion":
        if not hard:
            # NORMAL: construccion polinomial real — (v1+A)*(v2+B) expandido a 4 terminos.
            # El alumno agrupa: v2*(v1+A) + B*(v1+A) = (v1+A)*(v2+B).
            # Garantiza shape_ok: ambos factores son binomios no numericos.
            for _ in range(30):
                v1, v2 = _random_symbols(rng, 2)
                A = _nonzero_int(rng, -8, 8)
                B = _nonzero_int(rng, -8, 8)
                if A == B or v1 == v2:
                    continue
                question = sp.expand((v1 + A) * (v2 + B))
                answer = sp.factor(question)
                if answer.is_Mul and len(answer.args) >= 2:
                    break
            else:
                v1, v2 = sp.Symbol('x'), sp.Symbol('y')
                question = sp.expand((v1 + 3) * (v2 + 5))
                answer = sp.factor(question)
        else:
            # HARD: 6 terminos, 3 pares — el alumno debe descubrir como agrupar.
            # Construccion: (v1+c)*(a*v2 + b*v3 + d)
            for _ in range(30):
                v1, v2, v3 = _random_symbols(rng, 3)
                if len({v1, v2, v3}) < 3:
                    continue
                c = _nonzero_int(rng, -8, 8)
                a = _nonzero_int(rng, -coef_max, coef_max)
                b = _nonzero_int(rng, -coef_max, coef_max)
                d = _nonzero_int(rng, -coef_max, coef_max)
                question = sp.expand((v1 + c) * (a * v2 + b * v3 + d))
                answer = sp.factor(question)
                if answer.is_Mul and len(answer.args) >= 2:
                    break
            else:
                v1, v2, v3 = sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')
                question = sp.expand((v1 + 2) * (3 * v2 - v3 + 4))
                answer = sp.factor(question)

    elif subtype == "diferencia_cuadrados":
        v1, v2 = _random_symbols(rng, 2)
        a = rng.randint(2, coef_max)
        b = rng.randint(2, coef_max)
        p, q = rng.randint(1, 3), rng.randint(1, 3)
        if hard:
            # HARD: diferencia de cuadrados anidada — (a²-b²)(a²+b²) = a^4 - b^4
            # El alumno debe aplicar la identidad dos veces
            question = (a * v1**p) ** 4 - (b * v2**q) ** 4
        else:
            question = (a * v1**p) ** 2 - (b * v2**q) ** 2
        answer = sp.factor(question)

    elif subtype == "trinomio_cuadrado_perfecto":
        v = _random_symbol(rng)
        if not hard:
            a = _nonzero_int(rng, -coef_max, coef_max)
            b = _nonzero_int(rng, -coef_max, coef_max)
            answer = (a * v + b) ** 2
            question = sp.expand(answer)
        else:
            # HARD: TCP con dos variables (a*v1 + b*v2)^2 — forma menos obvia
            v2 = _random_symbol(rng)
            while v2 == v:
                v2 = _random_symbol(rng)
            a = _nonzero_int(rng, -coef_max, coef_max)
            b = _nonzero_int(rng, -coef_max, coef_max)
            answer = (a * v + b * v2) ** 2
            question = sp.expand(answer)
        answer = sp.factor(question)

    elif subtype == "trinomio_inspeccion":
        # [FIX-3] Restringido a leading=1 para que sea verdadera inspeccion.
        v = _random_symbol(rng)
        pool = [n for n in range(-root_max, root_max + 1) if n != 0]
        r1, r2 = rng.sample(pool, 2)
        answer_raw = (v + r1) * (v + r2)
        question = sp.expand(answer_raw)
        answer = sp.factor(question)

    elif subtype == "trinomio_ac":
        v = _random_symbol(rng)
        pool = [n for n in range(-root_max, root_max + 1) if n != 0]
        r1, r2 = rng.sample(pool, 2)
        if not hard:
            # NORMAL: AC directo, leading > 1
            leading = rng.choice([2, 3, 5])
            factor2_leading = rng.choice([1, 2, 3])
            answer_raw = (leading * v + r1) * (factor2_leading * v + r2)
            question = sp.expand(answer_raw)
            answer = sp.factor(question)
        else:
            # HARD: GCF primero, luego AC — dos pasos que el alumno debe descubrir.
            # Estructura: gcf * (leading*v^2 + B*v + C)  donde el trinomio interior es AC
            gcf = rng.choice([2, 3, 5, 6, 7])
            leading = rng.choice([2, 3, 5, 7])
            factor2_leading = rng.choice([1, 2, 3, 5])
            inner = sp.expand((leading * v + r1) * (factor2_leading * v + r2))
            question = sp.expand(gcf * inner)
            answer = sp.factor(question)

    elif subtype == "sustitucion":
        v = _random_symbol(rng)
        if not hard:
            # NORMAL: u = v^n, sustitucion directa de monomio
            power = rng.choice([2, 3, 4])
            pool = [n for n in range(-8, 9) if n != 0]
            r1, r2 = rng.sample(pool, 2)
            question = sp.expand((v**power + r1) * (v**power + r2))
            answer = sp.factor(question)  # forma canónica; garantiza que answer_expr coincida con validate_answer
        else:
            # HARD: u = (v + c)^n — sustitucion de binomio, forma oculta.
            # Se expande (v+c)^power como u, resultando en una estructura más compleja.
            c = _nonzero_int(rng, -5, 5)
            power = rng.choice([2, 3])
            pool = [n for n in range(-12, 13) if n != 0]
            r1, r2 = rng.sample(pool, 2)
            u_expr = (v + c)**power
            answer = (u_expr + r1) * (u_expr + r2)
            question = sp.expand(answer)
            answer = sp.factor(question)

    elif subtype == "suma_cubos":
        # [NEW-1] a^3 + b^3 = (a+b)(a^2 - ab + b^2)
        v1, v2 = _random_symbols(rng, 2)
        a_c = rng.randint(1, 5 if not hard else 8)
        b_c = rng.randint(1, 5 if not hard else 8)
        p = rng.choice([1, 3])
        q = rng.choice([1, 3])
        if hard:
            # HARD: GCF antes de aplicar suma de cubos
            gcf = rng.choice([2, 3, 4, 5])
            inner = (a_c * v1**p) ** 3 + (b_c * v2**q) ** 3
            question = sp.expand(gcf * inner)
        else:
            question = (a_c * v1**p) ** 3 + (b_c * v2**q) ** 3
        answer = sp.factor(question)

    elif subtype == "diferencia_cubos":
        # [NEW-1] a^3 - b^3 = (a-b)(a^2 + ab + b^2)
        v1, v2 = _random_symbols(rng, 2)
        a_c = rng.randint(1, 5 if not hard else 8)
        b_c = rng.randint(1, 5 if not hard else 8)
        p = rng.choice([1, 3])
        q = rng.choice([1, 3])
        if hard:
            # HARD: GCF antes de aplicar diferencia de cubos
            gcf = rng.choice([2, 3, 4, 5])
            inner = (a_c * v1**p) ** 3 - (b_c * v2**q) ** 3
            question = sp.expand(gcf * inner)
        else:
            question = (a_c * v1**p) ** 3 - (b_c * v2**q) ** 3
        answer = sp.factor(question)

    elif subtype == "formula_general":
        v = _random_symbol(rng)
        a_range = (-8, 8) if hard else (-6, 6)
        b_range = (-20, 20) if hard else (-15, 15)
        c_range = (-30, 30) if hard else (-20, 20)
        for _ in range(100):
            a = _nonzero_int(rng, *a_range)
            b = rng.randint(*b_range)
            c = rng.randint(*c_range)
            if b**2 - 4 * a * c >= 0:
                break
        else:
            a, b, c = 1, 0, -1
        if hard:
            # HARD: GCF oculto primero, luego formula general — dos pasos
            gcf = rng.choice([2, 3, 5])
            question = gcf * (a * v**2 + b * v + c)
        else:
            question = a * v**2 + b * v + c
        answer = sp.factor(question, extension=sp.sqrt(b**2 - 4 * a * c))

    elif subtype == "division_sintetica":
        v = _random_symbol(rng)
        # [NEW-4] Hard: grados 4-7 para mayor exigencia.
        degree_choices = list(range(4, 8)) if hard else [3, 4]
        num_roots = rng.choice(degree_choices)
        root_pool = list(range(-10 if hard else -8, 11 if hard else 9))
        roots = [rng.choice(root_pool) for _ in range(num_roots)]
        leading = sp.Integer(rng.choice([1, 1, 2, -1]))
        poly = leading
        for root in roots:
            poly *= v - root
        question = sp.expand(poly)
        factored = sp.factor(question)
        unique_roots = _sorted_solutions(list(set(roots)))

        # SIEMPRE calcular ambas: ceros y factorizacion.
        # La pregunta alterna entre pedir uno u otro, pero la respuesta correcta
        # incluye los dos para que el alumno pueda verificar ambas representaciones.
        prop = rng.choice(["factorizacion", "zeros", "zeros"])
        roots_str = ", ".join(str(r) for r in unique_roots)

        if prop == "zeros":
            stmt = ("Encuentra los ceros del polinomio usando division sintetica. "
                    "Indica todos los ceros (con multiplicidad si corresponde)."
                    if hard else
                    "Encuentra los ceros (raices) del polinomio usando division sintetica.")
            answer_label = f"Ceros: {roots_str}"
            return _new_exercise(
                "factorizacion", subtype,
                stmt,
                "math", question, unique_roots,
                {"type": "solution_set", "variable": str(v),
                 "solutions": [str(r) for r in unique_roots]},
                answer_label=answer_label,
                metadata={"method": subtype, "hard": hard, "property": "zeros",
                          "factored": str(factored), "roots": [str(r) for r in unique_roots]},
            )
        else:
            stmt = ("Factorice completamente el polinomio usando division sintetica."
                    if hard else
                    "Factorice el polinomio usando division sintetica.")
            answer_label = f"Factorizado: {_plain(factored)}"
            return _new_exercise(
                "factorizacion", subtype,
                stmt,
                "math", question, factored,
                {"type": "factorization", "answer_expr": str(factored), "allow_additive": False},
                answer_label=answer_label,
                metadata={"method": subtype, "hard": hard, "property": "factorizacion",
                          "factored": str(factored), "roots": [str(r) for r in unique_roots]},
            )

    elif subtype == "binomio_cubo":
        # (a*v ± b)^3 expandido => factorizar como binomio al cubo.
        v = _random_symbol(rng)
        a = _nonzero_int(rng, 1, 4 if not hard else 6)
        b = _nonzero_int(rng, 1, 6 if not hard else 10)
        sign = rng.choice([1, -1])
        if hard:
            # HARD: GCF antes del binomio al cubo — dos pasos
            gcf = rng.choice([2, 3, 4])
            raw = (a * v + sign * b) ** 3
            question = sp.expand(gcf * raw)
        else:
            raw = (a * v + sign * b) ** 3
            question = sp.expand(raw)
        answer = sp.factor(question)

    else:  # completar_cuadrado
        v = _random_symbol(rng)
        a_range = (-8, 8) if hard else (-5, 5)
        b_range = (-20, 20) if hard else (-12, 12)
        a = _nonzero_int(rng, *a_range)
        b = rng.choice([n for n in range(*b_range, 2) if n != 0] or [-4, -2, 2, 4])
        c = rng.randint(*((-20, 20) if hard else (-15, 15)))
        if hard:
            # HARD: coeficiente lider != 1, requiere sacar a(x+h)^2 + k
            # con h racional no obvio
            while a in (1, -1, 0):
                a = _nonzero_int(rng, *a_range)
        question = a * v**2 + b * v + c
        h = sp.Rational(b, 2 * a)
        k = sp.simplify(c - a * h**2)
        answer = sp.Add(a * (v + h) ** 2, k, evaluate=False)
        allow_additive = True

    stmt = stmt_hard if hard else stmt_normal

    # ── Intercambio expandir/reducir ───────────────────────────────────
    # En modo "expandir": el enunciado muestra la forma FACTORIZADA y pide la expandida.
    # En modo "reducir":  el enunciado muestra la forma EXPANDIDA  y pide la factorizada.
    if direction == "expandir":
        display_expr = answer          # mostrar: forma factorizada
        expected_expr = question       # esperar: forma expandida
    else:
        display_expr = question        # mostrar: forma expandida
        expected_expr = answer         # esperar: forma factorizada

    return _new_exercise(
        "factorizacion",
        subtype,
        stmt,
        "math",
        display_expr,
        expected_expr,
        {
            "type": "factorization" if direction == "reducir" else "math_equal",
            "answer_expr": str(expected_expr),
            "allow_additive": allow_additive,
        },
        answer_label="Resultado",
        metadata={"method": subtype, "hard": hard, "direction": direction},
    )


# =============================================================================
# RACIONALIZACION
# =============================================================================


RATIONALIZATION_SUBTYPES = [
    "raiz_cuadrada_simple",
    "raiz_n_esima",
    "binomio_raices_cuadradas",
    "binomio_raices_cubicas",
    "trinomio_raices_cuadradas",
    "doble_racionalizacion",    # [NEW-6] Numerador Y denominador con radicales — solo hard
]

# [NEW-2] Subtipos con denominador/numerador mas complejo.
HARD_RATIONALIZATION_SUBTYPES = [
    "raiz_n_esima",
    "binomio_raices_cubicas",
    "trinomio_raices_cuadradas",
    "doble_racionalizacion",    # [NEW-6]
]

# Categorias para seleccion ponderada (Paso 8)
# doble_racionalizacion es HARD-EXCLUSIVO — no aparece en ninguna categoria normal.
RATIONALIZATION_DIFFICULTY_CATEGORIES = {
    "easy":   ["raiz_cuadrada_simple"],
    "medium": ["raiz_n_esima", "binomio_raices_cuadradas"],
    "hard":   ["binomio_raices_cubicas", "trinomio_raices_cuadradas", "doble_racionalizacion"],
}

# Pool NORMAL excluye doble_racionalizacion explicitamente
NORMAL_RATIONALIZATION_SUBTYPES = [
    s for s in RATIONALIZATION_SUBTYPES if s != "doble_racionalizacion"
]


def _not_square(n: int) -> bool:
    return int(n**0.5) ** 2 != n


def _not_cube(n: int) -> bool:
    return round(abs(n) ** (1 / 3)) ** 3 != abs(n)


def _root(value: Any, index: int) -> Any:
    return sp.Pow(value, sp.Rational(1, index), evaluate=False)


def _unevaluated_fraction(numerator: Any, denominator: Any) -> Any:
    return sp.Mul(numerator, sp.Pow(denominator, -1, evaluate=False), evaluate=False)


def rationalize_expression(expr: Any, objective: str = "denominador") -> Any:
    if objective == "denominador":
        return sp.factor(sp.radsimp(expr))
    if objective == "numerador":
        return sp.factor(1 / sp.radsimp(1 / expr))
    raise ValueError("El objetivo debe ser 'denominador' o 'numerador'.")


def generate_rationalization_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
) -> dict[str, Any]:
    default_pool = HARD_RATIONALIZATION_SUBTYPES if hard else RATIONALIZATION_SUBTYPES
    subtype = subtype or rng.choice(default_pool)
    if subtype not in RATIONALIZATION_SUBTYPES:
        raise ValueError(f"Subtipo de racionalizacion desconocido: {subtype}")

    # [NEW-6] Hard: coeficientes mas grandes, siempre letras (mas abstracto).
    coef_range = (8, 20) if hard else (2, 12)
    use_letters = True if hard else rng.choice([True, False])
    if subtype in ("trinomio_raices_cuadradas", "doble_racionalizacion"):
        use_letters = False

    objective = rng.choice(["denominador", "numerador"])
    coef = rng.randint(*coef_range)

    if use_letters:
        symbols = [sp.Symbol(name, positive=True) for name in rng.sample(["x", "y", "z", "a", "b", "m", "n"], 3)]
        v1, v2, v3 = symbols
    else:
        nums = [n for n in range(2, 31) if _not_square(n)]
        v1, v2, v3 = rng.sample(nums, 3)

    # ── [NEW-6] doble_racionalizacion ───────────────────────────────────
    # HARD-EXCLUSIVO: expresion donde tanto numerador como denominador tienen
    # radicales. El alumno debe racionalizar los dos y simplificar.
    # Forma: (p*sqrt(a) + q) / (r*sqrt(b) + s)
    # Proceso: (1) multiplicar por conjugado del denominador, (2) verificar
    #          que el resultado tambien tenga numerador sin radical.
    if subtype == "doble_racionalizacion":
        sq_nums = [n for n in range(2, 40) if _not_square(n)]
        for _ in range(50):
            a_n = rng.choice(sq_nums)
            b_n = rng.choice([n for n in sq_nums if n != a_n])
            p = rng.randint(1, 6)
            q = rng.randint(1, 8)
            r = rng.randint(1, 6)
            s = rng.randint(1, 8)
            sign_n = rng.choice([1, -1])
            sign_d = rng.choice([1, -1])
            # Numerador: p*sqrt(a) + sign_n*q
            # Denominador: r*sqrt(b) + sign_d*s
            num_expr = p * sp.sqrt(a_n) + sign_n * q
            den_expr = r * sp.sqrt(b_n) + sign_d * s
            # Conjugado del denominador: r*sqrt(b) - sign_d*s
            conj = r * sp.sqrt(b_n) - sign_d * s
            denom_product = sp.expand(den_expr * conj)  # r²*b - s²  (entero)
            if denom_product == 0:
                continue
            # Resultado numerador: (p*sqrt(a)+sign_n*q)*(r*sqrt(b)-sign_d*s)
            full_num = sp.expand(num_expr * conj)
            result = sp.simplify(full_num / denom_product)
            # Verificar que el resultado no tenga sqrt en denominador
            r_num, r_den = sp.fraction(sp.together(result))
            if not _has_radical(r_den):
                break
        else:
            # Fallback: (sqrt(2)+1)/(sqrt(3)+1)
            num_expr = sp.sqrt(2) + 1
            den_expr = sp.sqrt(3) + 1
            conj = sp.sqrt(3) - 1
            denom_product = sp.Integer(2)
            result = sp.simplify(num_expr * conj / denom_product)

        expr = _unevaluated_fraction(num_expr, den_expr)
        answer = sp.nsimplify(result, rational=False)
        return _new_exercise(
            "racionalizacion", subtype,
            "Racionalice el denominador multiplicando por el conjugado. "
            "Simplifique completamente la expresion resultante.",
            "math", expr, answer,
            {
                "type": "rationalization",
                "answer_expr": str(answer),
                "objective": "denominador",
                "strict": True,
            },
            answer_label="Expresion racionalizada",
            metadata={"objective": "denominador", "uses_letters": False, "hard": True},
        )

    if subtype == "raiz_cuadrada_simple":
        # [FIX-6] Usar siempre radicando numerico para garantizar forma genuinamente distinta.
        objective = "denominador"
        sq_nums = [n for n in range(2, 60 if hard else 35) if _not_square(n)]
        v1_val = rng.choice(sq_nums)
        coef_val = rng.randint(*coef_range)
        if hard:
            # HARD: ademas agregar un termino entero al numerador: (coef + k) / sqrt(v)
            # para que al racionalizar el numerador quede mas complejo
            k = rng.randint(1, 8)
            num_expr_h = sp.Integer(coef_val) + k * sp.sqrt(sp.Integer(v1_val))
            # (coef + k*sqrt(v)) / sqrt(v)  =>  coef/sqrt(v) + k
            # Racionalizar: coef*sqrt(v)/v + k
            expr_den = _unevaluated_fraction(sp.Integer(coef_val) + k * sp.sqrt(sp.Integer(v1_val)),
                                              sp.sqrt(sp.Integer(v1_val)))
            raw_answer = (sp.Integer(coef_val) * sp.sqrt(sp.Integer(v1_val)) /
                          sp.Integer(v1_val) + k)
            answer_den = sp.nsimplify(raw_answer, rational=False)
            answer_num = answer_den
            v1 = sp.Integer(v1_val)
            coef = coef_val
        else:
            expr_den = _unevaluated_fraction(coef_val, sp.sqrt(v1_val))
            raw_answer = sp.Integer(coef_val) * sp.sqrt(sp.Integer(v1_val)) / sp.Integer(v1_val)
            answer_den = sp.nsimplify(raw_answer, rational=False)
            answer_num = answer_den
            v1 = sp.Integer(v1_val)
            coef = coef_val

    elif subtype == "raiz_n_esima":
        # [FIX-6] Usar siempre radicandos numericos.
        index = rng.randint(4, 7) if hard else rng.randint(3, 6)
        power = rng.randint(1, index - 1)
        ok_nums = [n for n in range(2, 80 if hard else 50) if round(n**(1/index))**index != n]
        v1_val = rng.choice(ok_nums)
        coef_val = rng.randint(*coef_range)
        denom = sp.Integer(v1_val) ** sp.Rational(power, index)
        expr_den = _unevaluated_fraction(coef_val, denom)
        complement = sp.Integer(v1_val) ** sp.Rational(index - power, index)
        answer_den = sp.nsimplify(sp.Integer(coef_val) * complement / sp.Integer(v1_val), rational=False)
        answer_num = _unevaluated_fraction(sp.Integer(v1_val), sp.Integer(coef_val) * complement)
        v1 = sp.Integer(v1_val)
        coef = coef_val

    elif subtype == "binomio_raices_cuadradas":
        sign = rng.choice([1, -1])
        if hard:
            # HARD: siempre usar letras simbolicas (mas abstracto) y objetivo siempre denominador
            # para que no haya ambiguedad
            objective = "denominador"
            use_letters = True
            sym_names = rng.sample(["a", "b", "m", "n", "p", "q"], 2)
            v1 = sp.Symbol(sym_names[0], positive=True)
            v2 = sp.Symbol(sym_names[1], positive=True)
        a_root = _root(v1, 2)
        b_root = _root(v2, 2)
        denom_expr = sp.Add(a_root, sign * b_root, evaluate=False)
        conjugate = sp.sqrt(v1) - sign * sp.sqrt(v2)
        product = sp.simplify(v1 - v2)
        expr_den = _unevaluated_fraction(coef, denom_expr)
        answer_den = sp.factor(coef * conjugate / product)
        raw_num = sp.together(sp.Integer(1) * product / (coef * conjugate))
        num_part, _ = sp.fraction(raw_num)
        if _has_radical(sp.simplify(num_part)):
            objective = "denominador"
        answer_num = _unevaluated_fraction(product, coef * conjugate)

    elif subtype == "binomio_raices_cubicas":
        if not use_letters:
            cube_nums = [n for n in range(2, 35) if _not_cube(n)]
            v1, v2 = rng.sample(cube_nums, 2)
        sign = rng.choice([1, -1])
        a_root = _root(v1, 3)
        b_root = _root(v2, 3)
        denom = sp.Add(a_root, sign * b_root, evaluate=False)
        a_eval = v1 ** sp.Rational(1, 3)
        b_eval = v2 ** sp.Rational(1, 3)
        factor_expr = a_eval**2 - sign * a_eval * b_eval + b_eval**2
        product = v1 + sign * v2
        expr_den = _unevaluated_fraction(coef, denom)
        answer_den = sp.factor(coef * factor_expr / product)
        answer_num = _unevaluated_fraction(product, coef * factor_expr)

    else:  # trinomio_raices_cuadradas
        denom = sp.Add(_root(v1, 2), _root(v2, 2), _root(v3, 2), evaluate=False)
        expr_den = _unevaluated_fraction(coef, denom)
        answer_den = sp.factor(sp.radsimp(coef / (sp.sqrt(v1) + sp.sqrt(v2) + sp.sqrt(v3))))
        answer_num = sp.factor(sp.together(1 / answer_den))

    if objective == "numerador":
        expr = _unevaluated_fraction(denom_expr if subtype == "binomio_raices_cuadradas" else denom, coef)
        answer = answer_num
    else:
        expr = expr_den
        answer = answer_den

    # HARD: statement sin pista de metodo
    if hard:
        statement = (
            f"Racionalice el {objective}. Identifique la expresion conjugada adecuada "
            f"y simplifique completamente."
        )
    else:
        statement = f"Racionalice el {objective} y simplifique al maximo."

    return _new_exercise(
        "racionalizacion",
        subtype,
        statement,
        "math",
        expr,
        answer,
        {
            "type": "rationalization",
            "answer_expr": str(answer),
            "objective": objective,
            "strict": True,
        },
        answer_label="Resultado racionalizado",
        metadata={"objective": objective, "uses_letters": use_letters, "hard": hard},
    )


# =============================================================================
# INECUACIONES Y VALOR ABSOLUTO
# =============================================================================


INEQUALITY_SUBTYPES = [
    "inecuacion_lineal",
    "inecuacion_cuadratica",
    "inecuacion_racional",        # [NEW-4] (x+a)/(x+b) op c
    "valor_absoluto_ecuacion",
    "valor_absoluto_inecuacion",
    "valor_absoluto_doble",       # [NEW-4] ||x+a| - b| op c
    "valor_absoluto_anidado",     # [NEW-5] |a*|x+b| + c| op k
    "valor_absoluto_mixto",       # [NEW-5] |ax+b| + |cx+d| op k
    "comparacion_valor_absoluto",
]

# [NEW-2] Subtipos que exigen manejo de intervalos o casos multiples.
HARD_INEQUALITY_SUBTYPES = [
    "inecuacion_cuadratica",
    "inecuacion_racional",        # [NEW-4]
    "valor_absoluto_inecuacion",
    "valor_absoluto_doble",       # [NEW-4]
    "valor_absoluto_anidado",     # [NEW-5]
    "valor_absoluto_mixto",       # [NEW-5]
    "comparacion_valor_absoluto",
]

# Categorias para seleccion ponderada (Paso 8)
INEQUALITY_DIFFICULTY_CATEGORIES = {
    "easy":   ["inecuacion_lineal", "valor_absoluto_ecuacion"],
    "medium": ["inecuacion_cuadratica", "valor_absoluto_inecuacion",
               "comparacion_valor_absoluto"],
    "hard":   ["inecuacion_racional", "valor_absoluto_doble",
               "valor_absoluto_anidado", "valor_absoluto_mixto"],
}


@lru_cache(maxsize=512)
def _solve_inequality_cached(rel: Relational) -> Any:
    try:
        return sp.solve_univariate_inequality(rel, X, relational=True)
    except Exception:
        return sp.solve(rel, X)


def _solve_inequality(rel: Relational) -> Any:
    try:
        return _solve_inequality_cached(rel)
    except TypeError:
        # Fallback por si la expresion no es hasheable (caso raro): resolver sin cache.
        try:
            return sp.solve_univariate_inequality(rel, X, relational=True)
        except Exception:
            return sp.solve(rel, X)


def generate_inequality_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
) -> dict[str, Any]:
    default_pool = HARD_INEQUALITY_SUBTYPES if hard else INEQUALITY_SUBTYPES
    subtype = subtype or rng.choice(default_pool)
    if subtype not in INEQUALITY_SUBTYPES:
        raise ValueError(f"Subtipo de inecuacion desconocido: {subtype}")

    x = X
    op = rng.choice(["<", "<=", ">", ">="])

    # [NEW-2] Rangos ampliados para modo hard.
    coef_range = (-20, 20) if hard else (-12, 12)
    const_range = (-30, 30) if hard else (-20, 20)

    # ── [NEW-4] inecuacion_racional ────────────────────────────────────
    if subtype == "inecuacion_racional":
        # (x + a) / (x + b) op c  con a ≠ b para evitar denominador=0 en la solucion.
        for _ in range(30):
            a = rng.randint(-8 if not hard else -14, 8 if not hard else 14)
            b = rng.randint(-8 if not hard else -14, 8 if not hard else 14)
            if a == b:
                continue
            c = rng.choice([-3, -2, -1, 1, 2, 3] if not hard else [-5,-4,-3,-2,-1,1,2,3,4,5])
            rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op]((x + a) / (x + b), c)
            answer = _solve_inequality(rel)
            if answer is not None:
                break
        else:
            a, b, c = 1, -2, 0
            rel = sp.Gt((x + 1) / (x - 2), 0)
            answer = _solve_inequality(rel)
        return _new_exercise(
            "inecuaciones", subtype,
            "Resuelva la inecuacion racional. Recuerde el dominio (denominador ≠ 0).",
            "math", rel, answer,
            {"type": "inequality", "solution": str(answer)},
            answer_label="Conjunto solucion",
            metadata={"hard": hard},
        )

    # ── [NEW-4] valor_absoluto_doble ────────────────────────────────────
    if subtype == "valor_absoluto_doble":
        # ||x + a| - b| op c  donde b >= 1, c >= 1.
        # Equivale a: |x+a| op b±c, lo que genera intervalos no triviales.
        for _ in range(15):
            a = rng.randint(-8 if not hard else -14, 8 if not hard else 14)
            b = rng.randint(1, 8 if not hard else 14)
            c = rng.randint(1, 6 if not hard else 10)
            inner = sp.Abs(x + a) - b
            outer = sp.Abs(inner)
            rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](outer, c)
            answer = _solve_inequality(rel)
            # Evitar soluciones triviales (todo R o vacio)
            if answer not in (True, False, sp.S.Reals, sp.EmptySet):
                break
        else:
            a, b, c = 0, 3, 1
            rel = sp.Lt(sp.Abs(sp.Abs(x) - 3), 1)
            answer = _solve_inequality(rel)
        return _new_exercise(
            "inecuaciones", subtype,
            "Resuelva la inecuacion con valor absoluto doble.",
            "math", rel, answer,
            {"type": "inequality", "solution": str(answer)},
            answer_label="Conjunto solucion",
            metadata={"hard": hard},
        )

    # ── [NEW-5] valor_absoluto_anidado ─────────────────────────────────
    # Forma: |a * |x + b| + c| op k
    # Para que el anidamiento sea no trivial (trampa algebraica), se necesita que
    # la expresion interior a*|x+b|+c pueda cambiar de signo:
    #   - si a > 0: c debe ser negativo (inner < 0 cuando |x+b| < |c|/a)
    #   - si a < 0: inner siempre puede ser negativo (a*|x+b| → -∞)
    # Asi el alumno debe primero simplificar ||...|, no obviar el signo exterior.
    if subtype == "valor_absoluto_anidado":
        for _ in range(20):
            # Garantizar que a y c tengan signos que produzcan cambio de signo interior
            a = rng.choice([-3, -2, 2, 3] if hard else [-2, 2])
            b = rng.randint(-6 if hard else -4, 6 if hard else 4)
            # c con signo opuesto a a, para que inner cambie de signo
            c_abs = rng.randint(1, 8 if hard else 5)
            c = -c_abs if a > 0 else c_abs   # inner = a*|x+b| + c cambia signo
            k = rng.randint(2, 10 if hard else 7)
            inner = a * sp.Abs(x + b) + c
            expr = sp.Abs(inner)
            rel_op = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op]
            rel = rel_op(expr, k)
            try:
                answer = sp.solveset(rel, x, sp.Reals)
                if answer not in (sp.S.Reals, sp.EmptySet, sp.S.EmptySet):
                    break
            except Exception:
                continue
        else:
            # Fallback: |2*|x+3| - 4| < 2  →  |x+3| ∈ (1,3)  →  x ∈ (-6,-2)∪(-4,0)
            rel = sp.Lt(sp.Abs(2 * sp.Abs(x + 3) - 4), 2)
            answer = sp.solveset(rel, x, sp.Reals)
        answer_str = str(answer)
        return _new_exercise(
            "inecuaciones", subtype,
            "Resuelva la inecuacion con valor absoluto anidado. "
            "Primero despeje el valor absoluto interior.",
            "math", rel, answer,
            {"type": "inequality", "solution": answer_str},
            answer_label="Conjunto solucion",
            metadata={"hard": hard},
        )

    # ── [NEW-5] valor_absoluto_mixto ────────────────────────────────────
    # Forma: |ax+b| + |cx+d| op k
    # Ejemplos: |2x+1| + |x-3| < 7
    # Requiere analizar los puntos criticos donde cambia el signo de cada argumento.
    if subtype == "valor_absoluto_mixto":
        for _ in range(20):
            a = rng.choice([1, 1, 2, 2, 3] if hard else [1, 1, 2])
            b = rng.randint(-8 if hard else -5, 8 if hard else 5)
            c_coef = rng.choice([1, 1, 2, 3] if hard else [1, 1, 2])
            d = rng.randint(-8 if hard else -5, 8 if hard else 5)
            k = rng.randint(3, 15 if hard else 10)
            # Evitar que los puntos criticos coincidan (haria trivial el analisis)
            pt1 = sp.Rational(-b, a)
            pt2 = sp.Rational(-d, c_coef)
            if pt1 == pt2:
                continue
            expr = sp.Abs(a * x + b) + sp.Abs(c_coef * x + d)
            rel_op = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op]
            rel = rel_op(expr, k)
            try:
                answer = sp.solveset(rel, x, sp.Reals)
                if answer not in (sp.S.Reals, sp.EmptySet):
                    break
            except Exception:
                continue
        else:
            # Fallback: |2x+1| + |x-3| < 7  (solucion conocida)
            rel = sp.Lt(sp.Abs(2 * x + 1) + sp.Abs(x - 3), 7)
            answer = sp.solveset(rel, x, sp.Reals)
        answer_str = str(answer)
        return _new_exercise(
            "inecuaciones", subtype,
            "Resuelva la inecuacion suma de valores absolutos. "
            "Identifique los puntos criticos y analice cada intervalo.",
            "math", rel, answer,
            {"type": "inequality", "solution": answer_str},
            answer_label="Conjunto solucion",
            metadata={"hard": hard},
        )

    if subtype == "inecuacion_lineal":
        a = _nonzero_int(rng, *coef_range)
        b = rng.randint(*const_range)
        c = rng.randint(*const_range)
        rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](a * x + b, c)
        answer = _solve_inequality(rel)

    elif subtype == "inecuacion_cuadratica":
        pool = [n for n in range(-12 if hard else -9, 13 if hard else 10) if n != 0]
        r1, r2 = sorted(rng.sample(pool, 2))
        a = _nonzero_int(rng, -6 if hard else -4, 6 if hard else 4)
        expr = sp.expand(a * (x - r1) * (x - r2))
        rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](expr, 0)
        answer = _solve_inequality(rel)

    elif subtype == "valor_absoluto_ecuacion":
        a = _nonzero_int(rng, *coef_range)
        b = rng.randint(*const_range)
        c = rng.randint(1, 25 if hard else 20)
        rel = sp.Eq(sp.Abs(a * x + b), c)
        answer = _sorted_solutions(sp.solve(rel, x))
        return _new_exercise(
            "inecuaciones",
            subtype,
            "Resuelva.",
            "math",
            rel,
            answer,
            {
                "type": "solution_set",
                "variable": "x",
                "solutions": [str(sol) for sol in answer],
            },
            answer_label="Soluciones",
            metadata={"hard": hard},
        )

    elif subtype == "valor_absoluto_inecuacion":
        a = _nonzero_int(rng, -10 if hard else -8, 10 if hard else 8)
        b = rng.randint(-15 if hard else -12, 15 if hard else 12)
        c = rng.randint(1, 25 if hard else 20)
        rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](sp.Abs(a * x + b), c)
        answer = _solve_inequality(rel)

    else:  # comparacion_valor_absoluto
        pool = list(range(-15 if hard else -12, 16 if hard else 13))
        a, b = rng.sample(pool, 2)
        rel = rng.choice([sp.Lt, sp.Le])(sp.Abs(x + a), sp.Abs(x + b))
        answer = _solve_inequality(rel)

    # NORMAL: enunciado con instruccion clara del metodo.
    # HARD: enunciado sin pista (el alumno debe reconocer el tipo).
    STMT_MAP = {
        "inecuacion_lineal":          ("Resuelva la inecuacion lineal.",
                                       "Resuelva. Determine el conjunto solucion."),
        "inecuacion_cuadratica":      ("Resuelva la inecuacion cuadratica.",
                                       "Resuelva. Exprese la solucion como intervalo o union de intervalos."),
        "valor_absoluto_inecuacion":  ("Resuelva la inecuacion con valor absoluto.",
                                       "Resuelva la inecuacion. Analice los dos casos del valor absoluto."),
        "comparacion_valor_absoluto": ("Compare los dos valores absolutos y resuelva.",
                                       "Resuelva la inecuacion. Identifique el punto critico."),
    }
    stmt_pair = STMT_MAP.get(subtype, ("Resuelva.", "Resuelva y justifique su respuesta."))
    stmt = stmt_pair[1] if hard else stmt_pair[0]

    return _new_exercise(
        "inecuaciones",
        subtype,
        stmt,
        "math",
        rel,
        answer,
        {
            "type": "inequality",
            "solution": str(answer),
        },
        answer_label="Conjunto solucion",
        metadata={"hard": hard},
    )


# =============================================================================
# GENERADOR UNIFICADO
# =============================================================================


GENERATOR_BY_TOPIC: dict[str, Callable[..., dict[str, Any]]] = {
    "conversiones": generate_conversion_exercise,
    "ecuaciones": generate_equation_exercise,
    "factorizacion": generate_factorization_exercise,
    "racionalizacion": generate_rationalization_exercise,
    "inecuaciones": generate_inequality_exercise,
}

SUBTYPES_BY_TOPIC = {
    "conversiones": CONVERSION_SUBTYPES,
    "ecuaciones": EQUATION_SUBTYPES,
    "factorizacion": FACTORIZATION_SUBTYPES,
    "racionalizacion": NORMAL_RATIONALIZATION_SUBTYPES,   # doble_racionalizacion excluido en normal
    "inecuaciones": INEQUALITY_SUBTYPES,
}

# [NEW-2] Subtipos curados para modo examen (mas exigentes).
HARD_SUBTYPES_BY_TOPIC = {
    "conversiones": HARD_CONVERSION_SUBTYPES,
    "ecuaciones": HARD_EQUATION_SUBTYPES,
    "factorizacion": HARD_FACTORIZATION_SUBTYPES,
    "racionalizacion": HARD_RATIONALIZATION_SUBTYPES,
    "inecuaciones": HARD_INEQUALITY_SUBTYPES,
}

# [NEW-8] Categorias por dificultad para seleccion ponderada.
# NORMAL: 70% facil / 25% media / 5% dificil
# HARD:   15% facil / 35% media / 50% dificil
DIFFICULTY_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "conversiones":    CONVERSION_DIFFICULTY_CATEGORIES,
    "ecuaciones":      EQUATION_DIFFICULTY_CATEGORIES,
    "factorizacion":   FACTORIZATION_DIFFICULTY_CATEGORIES,
    "racionalizacion": RATIONALIZATION_DIFFICULTY_CATEGORIES,
    "inecuaciones":    INEQUALITY_DIFFICULTY_CATEGORIES,
}

DEFAULT_EXAM_SUBTYPES_BY_TOPIC = SUBTYPES_BY_TOPIC


# NOTA: las versiones "reales" de generate_exercise() y generate_exam() estan
# mas abajo (buscar "def generate_exercise" / "def generate_exam" nuevamente);
# esas son las que efectivamente se usan (soportan seen_fingerprints, categorias
# de dificultad por subtipo, etc.). Aqui solo queda declarado el pool de
# subtipos por defecto que ambas versiones comparten.

# =============================================================================
# VALIDACION
# =============================================================================


def validate_answer(exercise: dict[str, Any], user_answer: str) -> dict[str, Any]:
    validation = exercise.get("validation", {})
    validation_type = validation.get("type")
    raw = str(user_answer).strip()

    # [NEW-9] Caso especial: conjuntos solucion vacios (cuadratica_sin_reales).
    # El frontend envia cadena vacia o "sin solucion" cuando no hay raices reales.
    # Aceptar ambas formas cuando el expected es lista vacia.
    if validation_type == "solution_set":
        expected_values = [_parse_math(item) for item in validation.get("solutions", [])]
        if not expected_values:
            # Respuesta esperada es conjunto vacio — aceptar vacio o frases canonicas
            NO_SOL_PHRASES = {"sin solucion", "no tiene solucion", "no hay solucion",
                               "no real solutions", "empty set", "conjunto vacio",
                               "vacio", "∅", "{}"}
            accepted = (not raw) or (raw.lower() in NO_SOL_PHRASES)
            return {"correct": bool(accepted), "expected": []}

    if not raw:
        return {"correct": False, "message": "Respuesta vacia."}

    try:
        if validation_type == "numeric":
            match = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", raw, flags=re.I)
            if not match:
                return {"correct": False, "message": "No se encontro un numero en la respuesta."}
            user_value = float(match.group(0).replace(",", "."))
            expected = float(validation["answer"])
            tolerance = float(validation.get("tolerance", 0.01))
            correct = abs(user_value - expected) <= tolerance
            return {"correct": correct, "expected": expected, "tolerance": tolerance}

        if validation_type == "solution_set":
            user_values = _parse_solution_list(raw)
            expected_values = [_parse_math(item) for item in validation.get("solutions", [])]
            correct = _same_solution_set(user_values, expected_values)
            return {"correct": correct, "expected": [_plain(v) for v in expected_values]}

        if validation_type == "math_equal":
            user_expr = _parse_math(raw)
            expected_expr = _parse_math(validation["answer_expr"])
            return {"correct": _math_equal(user_expr, expected_expr), "expected": _plain(expected_expr)}

        if validation_type == "factorization":
            user_expr = _parse_math(raw)
            # [FIX-5] Usar parse sin evaluacion para verificar forma: evita que
            # "2*(3x+1)" se distribuya a "6x+2" (Add) al parsear con evaluate=True.
            try:
                user_shape = _parse_math_shape(raw)
            except Exception:
                user_shape = user_expr
            expected_expr = _parse_math(validation["answer_expr"])
            equivalent = _math_equal(sp.expand(user_expr), sp.expand(expected_expr))
            additive_ok = bool(validation.get("allow_additive", False))
            factored_shape_ok = additive_ok or not getattr(user_shape, "is_Add", False)
            return {
                "correct": bool(equivalent and factored_shape_ok),
                "expected": _plain(expected_expr),
                "equivalent": bool(equivalent),
                "shape_ok": bool(factored_shape_ok),
            }

        if validation_type == "rationalization":
            user_expr = _parse_math(raw, positive_symbols=True)
            try:
                user_shape = _parse_math_shape(raw, positive_symbols=True)
            except Exception:
                user_shape = user_expr
            expected_expr = _parse_math(validation["answer_expr"], positive_symbols=True)
            equivalent = _math_equal(user_expr, expected_expr)
            objective = validation.get("objective", "denominador")
            rationalized = _target_is_rationalized(user_shape, objective, preserve_shape=True)
            strict = bool(validation.get("strict", True))
            correct = equivalent and (rationalized or not strict)
            message = "Correcto." if correct else "La expresion debe ser equivalente y quedar racionalizada."
            return {
                "correct": bool(correct),
                "expected": _plain(expected_expr),
                "equivalent": bool(equivalent),
                "rationalized": bool(rationalized),
                "message": message,
            }

        if validation_type == "inequality":
            # [FIX-1] Reemplazado sp.simplify_logic (logica booleana proposicional)
            # por comparacion de conjuntos reales usando .as_set().
            # Esto maneja correctamente Union, Intersection, Interval, etc.
            # [NEW-5] Tambien acepta Set objects directos (de solveset) como expected.
            user_expr = _parse_math(raw)
            expected_raw = validation["solution"]
            correct = False
            try:
                # Intentar parsear el expected como expresion sympy
                expected_expr = _parse_math(expected_raw)
                # Obtener la representacion como conjunto para ambos
                if isinstance(user_expr, sp.Set):
                    u_set = user_expr
                else:
                    u_set = user_expr.as_set()
                if isinstance(expected_expr, sp.Set):
                    e_set = expected_expr
                else:
                    e_set = expected_expr.as_set()
                eq_result = u_set.equals(e_set)
                if eq_result is not None:
                    correct = bool(eq_result)
                else:
                    correct = (u_set == e_set)
            except Exception:
                # Ultimo recurso: comparacion de strings de formas simplificadas.
                try:
                    correct = str(sp.simplify(user_expr)) == str(sp.simplify(_parse_math(expected_raw)))
                except Exception:
                    correct = False
            return {"correct": bool(correct), "expected": str(expected_raw)}

    except Exception as exc:
        return {"correct": False, "message": f"No se pudo interpretar la respuesta: {exc}"}

    return {"correct": False, "message": f"Tipo de validacion no soportado: {validation_type}"}


validar_respuesta = validate_answer

# =============================================================================
# EXPORTACION PDF COMPACTA
# =============================================================================

PAGE_W, PAGE_H = A4
MM = 2.83465

# Paleta de colores exacta de Teorema (basada en el frontend)
PDF_COLORS = {
    "brand": (18 / 255, 75 / 255, 104 / 255),   # --brand: #124b68
    "accent": (194 / 255, 122 / 255, 44 / 255), # --accent: #c27a2c
    "ink": (24 / 255, 32 / 255, 47 / 255),      # --ink: #18202f
    "muted": (102 / 255, 112 / 255, 133 / 255), # --muted: #667085
    "line": (217 / 255, 225 / 255, 234 / 255),  # --line: #d9e1ea
    "soft": (244 / 255, 247 / 255, 251 / 255),  # --soft: #f4f7fb
    "white": (1, 1, 1),
}

PDF_MARGIN_X = 15 * MM
PDF_TOP_MARGIN = 18 * MM
PDF_BOTTOM_MARGIN = 16 * MM
PDF_CONTENT_W = PAGE_W - 2 * PDF_MARGIN_X
PDF_BOX_PAD_X = 5 * MM
PDF_BOX_PAD_Y = 3.5 * MM
PDF_LINE_H = 4.8 * MM
PDF_MATH_DPI = 170
PDF_MATH_FONTSIZE = 15
PDF_ANSWER_FONTSIZE = 13
PDF_CASES_BRACKET_W = 6 * MM
PDF_CASES_ROW_GAP = 1.6 * MM

# Expresión regular para sistemas de ecuaciones
_CASES_LATEX_RE = re.compile(r"^\\begin\{cases\}(.*)\\end\{cases\}$", re.DOTALL)


def _cases_rows(latex: str) -> list[str] | None:
    match = _CASES_LATEX_RE.match(latex.strip())
    if not match:
        return None
    rows = [row.strip() for row in re.split(r"\\\\", match.group(1)) if row.strip()]
    return rows or None


def _draw_system_bracket(c: rl_canvas.Canvas, x: float, top_y: float, bottom_y: float) -> None:
    height = top_y - bottom_y
    if height <= 0:
        return
    base_size = 14.0
    c.saveState()
    c.setFont("Helvetica", base_size)
    _set_fill(c, PDF_COLORS["muted"])
    scale_y = max(height / base_size, 0.35)
    c.translate(x, (top_y + bottom_y) / 2)
    c.scale(1.0, scale_y)
    c.drawString(0, -base_size * 0.36, "{")
    c.restoreState()


def _set_fill(c: rl_canvas.Canvas, color: tuple[float, float, float]) -> None:
    c.setFillColorRGB(*color)


def _set_stroke(c: rl_canvas.Canvas, color: tuple[float, float, float]) -> None:
    c.setStrokeColorRGB(*color)


def _wrap_lines(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if current and stringWidth(candidate, font_name, font_size) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or [""]


@lru_cache(maxsize=1024)
def _render_math_bytes(latex: str, fontsize: float = PDF_MATH_FONTSIZE) -> bytes:
    try:
        return _render_mathtext_png(latex, fontsize, math_mode=True)
    except Exception:
        return _render_mathtext_png(latex, fontsize * 0.85, math_mode=False)


def _render_mathtext_png(latex: str, fontsize: float, *, math_mode: bool) -> bytes:
    with _MATPLOTLIB_LOCK:
        fig = plt.figure(figsize=(8, 1.25))
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        text = f"${latex}$" if math_mode else latex
        # Color cambiado a --ink exacto
        ax.text(0.02, 0.5, text, fontsize=fontsize, va="center", ha="left", color="#18202f")
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=PDF_MATH_DPI,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.04,
        )
        plt.close(fig)
        return buffer.getvalue()


def _render_math(latex: str, fontsize: float = PDF_MATH_FONTSIZE) -> io.BytesIO:
    return io.BytesIO(_render_math_bytes(latex, fontsize))


@lru_cache(maxsize=1024)
def _math_natural_size(latex: str, fontsize: float) -> tuple[float, float]:
    buffer = _render_math(latex, fontsize)
    image = ImageReader(buffer)
    width_px, height_px = image.getSize()
    width = width_px * 72 / PDF_MATH_DPI
    height = height_px * 72 / PDF_MATH_DPI
    return width, height


def _math_size(latex: str, fontsize: float, max_width: float) -> tuple[float, float]:
    width, height = _math_natural_size(latex, fontsize)
    if width > max_width:
        scale = max_width / width
        width *= scale
        height *= scale
    return width, height


def _draw_page_header(c: rl_canvas.Canvas, title: str, label: str) -> float:
    header_h = 18 * MM
    # Fondo brand en el header
    _set_fill(c, PDF_COLORS["brand"])
    c.rect(0, PAGE_H - header_h, PAGE_W, header_h, fill=1, stroke=0)
    
    # Borde inferior grueso simulando la topbar
    _set_fill(c, PDF_COLORS["accent"])
    c.rect(0, PAGE_H - header_h, PAGE_W, 4, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 13)
    _set_fill(c, PDF_COLORS["white"])
    c.drawString(PDF_MARGIN_X, PAGE_H - 8 * MM, label)
    c.setFont("Helvetica", 8.5)
    
    # Color de subtítulo ligeramente opaco
    _set_fill(c, (0.85, 0.88, 0.92))
    c.drawRightString(PAGE_W - PDF_MARGIN_X, PAGE_H - 8 * MM, title.upper())
    return PAGE_H - header_h - 8 * MM


def _estimate_display_height(display: dict[str, Any], fontsize: float = PDF_MATH_FONTSIZE) -> float:
    max_width = PDF_CONTENT_W - 2 * PDF_BOX_PAD_X
    if display["kind"] == "math":
        rows = _cases_rows(display["latex"])
        if rows is not None:
            row_max_width = max_width - PDF_CASES_BRACKET_W
            heights = [_math_size(row, fontsize, row_max_width)[1] for row in rows]
            total_h = sum(heights) + PDF_CASES_ROW_GAP * max(0, len(heights) - 1)
            return total_h + 2 * PDF_BOX_PAD_Y
        _, math_h = _math_size(display["latex"], fontsize, max_width)
        return math_h + 2 * PDF_BOX_PAD_Y
    lines = _wrap_lines(display["plain"], "Helvetica", 9.5, max_width)
    return len(lines) * PDF_LINE_H + 2 * PDF_BOX_PAD_Y


def _draw_display_box(
    c: rl_canvas.Canvas,
    display: dict[str, Any],
    y: float,
    *,
    fontsize: float = PDF_MATH_FONTSIZE,
) -> float:
    max_width = PDF_CONTENT_W - 2 * PDF_BOX_PAD_X
    box_h = _estimate_display_height(display, fontsize)
    _set_fill(c, PDF_COLORS["soft"])
    _set_stroke(c, PDF_COLORS["line"])
    c.setLineWidth(0.5)
    
    # Bordes redondeados simulando CSS --radius-sm/md
    c.roundRect(PDF_MARGIN_X, y - box_h, PDF_CONTENT_W, box_h, 8, fill=1, stroke=1)

    if display["kind"] == "math":
        rows = _cases_rows(display["latex"])
        if rows is not None:
            row_max_width = max_width - PDF_CASES_BRACKET_W
            sizes = [_math_size(row, fontsize, row_max_width) for row in rows]
            total_h = sum(h for _, h in sizes) + PDF_CASES_ROW_GAP * max(0, len(sizes) - 1)
            top_y = y - PDF_BOX_PAD_Y
            bottom_y = top_y - total_h
            _draw_system_bracket(c, PDF_MARGIN_X + PDF_BOX_PAD_X, top_y, bottom_y)
            cursor_y = top_y
            for row_latex, (row_w, row_h) in zip(rows, sizes):
                image_y = cursor_y - row_h
                buffer = _render_math(row_latex, fontsize)
                c.drawImage(
                    ImageReader(buffer),
                    PDF_MARGIN_X + PDF_BOX_PAD_X + PDF_CASES_BRACKET_W,
                    image_y,
                    row_w,
                    row_h,
                    mask="auto",
                )
                cursor_y -= row_h + PDF_CASES_ROW_GAP
        else:
            width, height = _math_size(display["latex"], fontsize, max_width)
            image_y = y - PDF_BOX_PAD_Y - height
            buffer = _render_math(display["latex"], fontsize)
            c.drawImage(ImageReader(buffer), PDF_MARGIN_X + PDF_BOX_PAD_X, image_y, width, height, mask="auto")
    else:
        lines = _wrap_lines(display["plain"], "Helvetica", 9.5, max_width)
        c.setFont("Helvetica", 9.5)
        _set_fill(c, PDF_COLORS["ink"])
        text_y = y - PDF_BOX_PAD_Y - 9.5
        for line in lines:
            c.drawString(PDF_MARGIN_X + PDF_BOX_PAD_X, text_y, line)
            text_y -= PDF_LINE_H
    return y - box_h


def _estimate_question_pdf_height(exercise: dict[str, Any], answer_lines: int) -> float:
    statement_lines = _wrap_lines(exercise["statement"], "Helvetica", 9, PDF_CONTENT_W - 18 * MM)
    statement_h = len(statement_lines) * PDF_LINE_H
    display_h = _estimate_display_height(exercise["display"])
    answer_h = max(0, answer_lines) * 6.5 * MM
    return 6 * MM + statement_h + 2 * MM + display_h + 3 * MM + answer_h + 6 * MM


def _draw_question_pdf(c: rl_canvas.Canvas, number: int, exercise: dict[str, Any], y: float, answer_lines: int) -> float:
    c.setFont("Helvetica-Bold", 10.5)
    # Color de enumerador: Brand
    _set_fill(c, PDF_COLORS["brand"])
    c.drawString(PDF_MARGIN_X, y, f"{number:02d}.")

    c.setFont("Helvetica", 9)
    _set_fill(c, PDF_COLORS["ink"])
    statement_lines = _wrap_lines(exercise["statement"], "Helvetica", 9, PDF_CONTENT_W - 18 * MM)
    text_y = y
    for line in statement_lines:
        c.drawString(PDF_MARGIN_X + 12 * MM, text_y, line)
        text_y -= PDF_LINE_H

    y = text_y - 2 * MM
    y = _draw_display_box(c, exercise["display"], y)
    y -= 3 * MM

    if answer_lines > 0:
        c.setFont("Helvetica", 8.5)
        _set_fill(c, PDF_COLORS["muted"])
        c.drawString(PDF_MARGIN_X, y, "Respuesta:")
        y -= 4 * MM
        _set_stroke(c, PDF_COLORS["line"])
        c.setLineWidth(0.5)
        for _ in range(answer_lines):
            c.line(PDF_MARGIN_X, y, PAGE_W - PDF_MARGIN_X, y)
            y -= 6.5 * MM

    _set_stroke(c, PDF_COLORS["line"])
    c.setLineWidth(0.35)
    c.line(PDF_MARGIN_X, y, PAGE_W - PDF_MARGIN_X, y)
    return y - 6 * MM


def _answer_displays(answer: dict[str, Any]) -> list[dict[str, Any]]:
    if answer["kind"] == "list":
        values = answer["latex"] or answer["plain"]
        if not values:
            return [_display_payload("text", "Sin solucion real")]
        return [{"kind": "math", "plain": p, "latex": l} for p, l in zip(answer["plain"], answer["latex"])]
    if answer["kind"] == "math":
        return [{"kind": "math", "plain": answer["plain"], "latex": answer["latex"]}]
    return [{"kind": "text", "plain": answer["plain"], "latex": None}]


def _estimate_answer_pdf_height(exercise: dict[str, Any]) -> float:
    displays = _answer_displays(exercise["answer"])
    content_h = sum(_estimate_display_height(display, PDF_ANSWER_FONTSIZE) for display in displays)
    gaps = max(0, len(displays) - 1) * 2 * MM
    label_lines = _wrap_lines(exercise["answer"].get("label", "Respuesta"), "Helvetica", 9, PDF_CONTENT_W - 18 * MM)
    return 6 * MM + len(label_lines) * PDF_LINE_H + 2 * MM + content_h + gaps + 6 * MM


def _draw_answer_pdf(c: rl_canvas.Canvas, number: int, exercise: dict[str, Any], y: float) -> float:
    c.setFont("Helvetica-Bold", 10.5)
    # Color de enumerador: Brand
    _set_fill(c, PDF_COLORS["brand"])
    c.drawString(PDF_MARGIN_X, y, f"{number:02d}.")

    c.setFont("Helvetica", 9)
    _set_fill(c, PDF_COLORS["ink"])
    label = exercise["answer"].get("label", "Respuesta")
    label_lines = _wrap_lines(label, "Helvetica", 9, PDF_CONTENT_W - 18 * MM)
    text_y = y
    for line in label_lines:
        c.drawString(PDF_MARGIN_X + 12 * MM, text_y, line)
        text_y -= PDF_LINE_H
    y = text_y - 2 * MM

    displays = _answer_displays(exercise["answer"])
    for display in displays:
        y = _draw_display_box(c, display, y, fontsize=PDF_ANSWER_FONTSIZE)
        y -= 2 * MM

    _set_stroke(c, PDF_COLORS["line"])
    c.setLineWidth(0.35)
    c.line(PDF_MARGIN_X, y, PAGE_W - PDF_MARGIN_X, y)
    return y - 6 * MM


def _default_pdf_path(title: str) -> str:
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title.strip()).strip("_") or "Examen"
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return os.path.join(os.path.expanduser("~"), "Downloads", filename)


def export_exam_pdf(
    exam: dict[str, Any],
    output_path: str | None = None,
    *,
    include_answer_key: bool = True,
    answer_lines: int = 1,
) -> str:
    output_path = output_path or _default_pdf_path(exam.get("title", "Examen"))
    folder = os.path.dirname(os.path.abspath(output_path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    title = exam.get("title", "EXAMEN")
    questions = exam.get("questions", [])

    c = rl_canvas.Canvas(output_path, pagesize=A4)
    c.setTitle(title)
    c.setAuthor("Teorema - Backend Modulo 1")

    y = _draw_page_header(c, title, "EXAMEN")
    for index, exercise in enumerate(questions, start=1):
        needed = _estimate_question_pdf_height(exercise, answer_lines)
        if y - needed < PDF_BOTTOM_MARGIN:
            c.showPage()
            y = _draw_page_header(c, title, "EXAMEN")
        y = _draw_question_pdf(c, index, exercise, y, answer_lines)

    if include_answer_key:
        c.showPage()
        y = _draw_page_header(c, title, "SOLUCIONARIO")
        for index, exercise in enumerate(questions, start=1):
            needed = _estimate_answer_pdf_height(exercise)
            if y - needed < PDF_BOTTOM_MARGIN:
                c.showPage()
                y = _draw_page_header(c, title, "SOLUCIONARIO")
            y = _draw_answer_pdf(c, index, exercise, y)

    c.save()
    return output_path


def export_rationalization_exam_pdf(
    quantity: int,
    output_path: str | None = None,
    *,
    seed: int | str | None = None,
    include_answer_key: bool = True,
    difficulty: str = "normal",
) -> str:
    exam = generate_exam("racionalizacion", quantity, seed=seed, difficulty=difficulty)
    return export_exam_pdf(exam, output_path, include_answer_key=include_answer_key, answer_lines=1)


def export_exam_pdf_bytes(
    exam: dict[str, Any],
    *,
    include_answer_key: bool = True,
    answer_lines: int = 1,
) -> bytes:
    import io as _io
    title = exam.get("title", "EXAMEN")
    questions = exam.get("questions", [])

    buf = _io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(title)
    c.setAuthor("Teorema - Backend Modulo 1")

    y = _draw_page_header(c, title, "EXAMEN")
    for index, exercise in enumerate(questions, start=1):
        needed = _estimate_question_pdf_height(exercise, answer_lines)
        if y - needed < PDF_BOTTOM_MARGIN:
            c.showPage()
            y = _draw_page_header(c, title, "EXAMEN")
        y = _draw_question_pdf(c, index, exercise, y, answer_lines)

    if include_answer_key:
        c.showPage()
        y = _draw_page_header(c, title, "SOLUCIONARIO")
        for index, exercise in enumerate(questions, start=1):
            needed = _estimate_answer_pdf_height(exercise)
            if y - needed < PDF_BOTTOM_MARGIN:
                c.showPage()
                y = _draw_page_header(c, title, "SOLUCIONARIO")
            y = _draw_answer_pdf(c, index, exercise, y)

    c.save()
    return buf.getvalue()


exportar_examen_pdf = export_exam_pdf
exportar_examen_racionalizacion_pdf = export_rationalization_exam_pdf
exportar_examen_pdf_bytes = export_exam_pdf_bytes

# =============================================================================
# API 2026: DIFICULTADES REALES, VALOR ABSOLUTO SEPARADO Y VERIFICACION FINAL
# =============================================================================


_LEGACY_GENERATORS = {
    "conversiones": generate_conversion_exercise,
    "ecuaciones": generate_equation_exercise,
    "factorizacion": generate_factorization_exercise,
    "racionalizacion": generate_rationalization_exercise,
    "inecuaciones": generate_inequality_exercise,
}
_LEGACY_VALIDATE_ANSWER = validate_answer

# Snapshot de INEQUALITY_SUBTYPES ANTES de que se redefinan en la sección API 2026.
# Necesario para que generate_absolute_value_exercise pueda pasar subtipos de
# valor_absoluto al generador legacy sin manipular globals() en tiempo de ejecución.
_ORIGINAL_INEQUALITY_SUBTYPES: list[str] = list(INEQUALITY_SUBTYPES)
# Lock para evitar condiciones de carrera al acceder al generador legacy de inecuaciones
# en servidores multi-hilo (ThreadingHTTPServer).
_INEQUALITY_SUBTYPE_LOCK = threading.Lock()

DIFFICULTY_LEVELS = ("easy", "normal", "hard")
DIFFICULTY_LABELS = {
    "easy": "Facil",
    "normal": "Normal",
    "hard": "Avanzado",
}

TOPICS = {
    "conversiones": "Conversiones de unidades",
    "ecuaciones": "Ecuaciones lineales y cuadraticas",
    "factorizacion": "Factorizacion",
    "racionalizacion": "Racionalizacion",
    "inecuaciones": "Inecuaciones",
    "valor_absoluto": "Valor absoluto",
    "binomio_newton": "Binomio de Newton",
}

TOPIC_TITLES = {
    "conversiones": "EXAMEN DE CONVERSIONES",
    "ecuaciones": "EXAMEN DE ECUACIONES",
    "factorizacion": "EXAMEN DE FACTORIZACION",
    "racionalizacion": "EXAMEN DE RACIONALIZACION",
    "inecuaciones": "EXAMEN DE INECUACIONES",
    "valor_absoluto": "EXAMEN DE VALOR ABSOLUTO",
    "binomio_newton": "EXAMEN DE BINOMIO DE NEWTON",
}

ABSOLUTE_VALUE_SUBTYPES = [
    "valor_absoluto_ecuacion",
    "valor_absoluto_inecuacion",
    "comparacion_valor_absoluto",
    "valor_absoluto_doble",
    "valor_absoluto_anidado",
    "valor_absoluto_mixto",
]

INEQUALITY_SUBTYPES = [
    "inecuacion_lineal",
    "inecuacion_cuadratica",
    "inecuacion_racional",
]

FACTORIZATION_SUBTYPES = [
    "factor_comun",
    "agrupacion",
    "diferencia_cuadrados",
    "trinomio_cuadrado_perfecto",
    "trinomio_inspeccion",
    "trinomio_ac",
    "sustitucion",
    "suma_cubos",
    "diferencia_cubos",
    "binomio_cubo",
    "formula_general",
    "division_sintetica",
    "completar_cuadrado",
    "completar_cuadrado_trinomio",
    "completar_cuadrado_binomio",
]

HARD_CONVERSION_SUBTYPES = ["area", "velocidad", "notacion_cientifica", "sistema_ingles", "volumen", "masa"]
HARD_EQUATION_SUBTYPES = [
    "lineal_fracciones",
    "cuadratica_raices_racionales",
    "cuadratica_sin_reales",
    "cuadratica_ambos_lados",
    "cuadratica_analisis",
    "cuadratica_radical",
    "sistemas_ecuaciones",
]
HARD_FACTORIZATION_SUBTYPES = [
    "sustitucion",
    "formula_general",
    "division_sintetica",
    "completar_cuadrado_trinomio",
    "completar_cuadrado_binomio",
]
HARD_RATIONALIZATION_SUBTYPES = [
    "binomio_raices_cubicas",
    "trinomio_raices_cuadradas",
    "doble_racionalizacion",
]
HARD_INEQUALITY_SUBTYPES = ["inecuacion_racional"]
HARD_ABSOLUTE_VALUE_SUBTYPES = [
    "valor_absoluto_doble",
    "valor_absoluto_anidado",
    "valor_absoluto_mixto",
]

BINOMIO_NEWTON_SUBTYPES = [
    "coeficiente_posicion_k",
    "termino_posicion_k",
    "polinomio_completo",
    "terminos_rango",
    "posicion_de_potencia",
]

HARD_BINOMIO_NEWTON_SUBTYPES = [
    "coeficiente_posicion_k",
    "termino_posicion_k",
    "terminos_rango",
    "posicion_de_potencia",
]

BINOMIO_NEWTON_DIFFICULTY_CATEGORIES: dict[str, list[str]] = {
    "easy":   ["coeficiente_posicion_k", "termino_posicion_k", "polinomio_completo"],
    "normal": ["coeficiente_posicion_k", "termino_posicion_k",
               "polinomio_completo", "terminos_rango", "posicion_de_potencia"],
    "hard":   ["coeficiente_posicion_k", "termino_posicion_k",
               "terminos_rango", "posicion_de_potencia"],
}

TOPIC_DIFFICULTY_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "conversiones": {
        "easy": ["temperatura", "distancia_metrica", "tiempo"],
        "normal": ["volumen", "masa", "area"],
        "hard": ["velocidad", "notacion_cientifica", "sistema_ingles"],
    },
    "ecuaciones": {
        "easy": ["lineal_basica", "cuadratica_raices_enteras", "sistemas_ecuaciones"],
        "normal": ["lineal_parentesis", "lineal_fracciones", "cuadratica_analisis", "sistemas_ecuaciones"],
        "hard": ["cuadratica_raices_racionales", "cuadratica_sin_reales", "cuadratica_ambos_lados", "cuadratica_radical", "sistemas_ecuaciones"],
    },
    "factorizacion": {
        "easy": ["factor_comun", "diferencia_cuadrados", "trinomio_cuadrado_perfecto", "trinomio_inspeccion"],
        "normal": ["agrupacion", "trinomio_ac", "suma_cubos", "diferencia_cubos", "binomio_cubo", "completar_cuadrado"],
        "hard": ["sustitucion", "formula_general", "division_sintetica", "completar_cuadrado_trinomio", "completar_cuadrado_binomio"],
    },
    "racionalizacion": {
        "easy": ["raiz_cuadrada_simple"],
        "normal": ["raiz_n_esima", "binomio_raices_cuadradas"],
        "hard": ["binomio_raices_cubicas", "trinomio_raices_cuadradas", "doble_racionalizacion"],
    },
    "inecuaciones": {
        "easy": ["inecuacion_lineal"],
        "normal": ["inecuacion_cuadratica"],
        "hard": ["inecuacion_racional"],
    },
    "valor_absoluto": {
        "easy": ["valor_absoluto_ecuacion"],
        "normal": ["valor_absoluto_inecuacion", "comparacion_valor_absoluto"],
        "hard": ["valor_absoluto_doble", "valor_absoluto_anidado", "valor_absoluto_mixto"],
    },
    "binomio_newton": {
        "easy":   ["coeficiente_posicion_k", "termino_posicion_k", "polinomio_completo"],
        "normal": ["coeficiente_posicion_k", "termino_posicion_k",
                   "polinomio_completo", "terminos_rango", "posicion_de_potencia"],
        "hard":   ["coeficiente_posicion_k", "termino_posicion_k",
                   "terminos_rango", "posicion_de_potencia"],
    },
}

SUBTYPES_BY_TOPIC = {
    "conversiones": CONVERSION_SUBTYPES,
    "ecuaciones": EQUATION_SUBTYPES,
    "factorizacion": FACTORIZATION_SUBTYPES,
    "racionalizacion": RATIONALIZATION_SUBTYPES,
    "inecuaciones": INEQUALITY_SUBTYPES,
    "valor_absoluto": ABSOLUTE_VALUE_SUBTYPES,
    "binomio_newton": BINOMIO_NEWTON_SUBTYPES,
}

HARD_SUBTYPES_BY_TOPIC = {
    "conversiones": HARD_CONVERSION_SUBTYPES,
    "ecuaciones": HARD_EQUATION_SUBTYPES,
    "factorizacion": HARD_FACTORIZATION_SUBTYPES,
    "racionalizacion": HARD_RATIONALIZATION_SUBTYPES,
    "inecuaciones": HARD_INEQUALITY_SUBTYPES,
    "valor_absoluto": HARD_ABSOLUTE_VALUE_SUBTYPES,
    "binomio_newton": HARD_BINOMIO_NEWTON_SUBTYPES,
}

DIFFICULTY_CATEGORIES = TOPIC_DIFFICULTY_CATEGORIES
DEFAULT_EXAM_SUBTYPES_BY_TOPIC = SUBTYPES_BY_TOPIC


def _normalize_difficulty(difficulty: Any = None, hard: bool | None = None) -> str:
    if hard is True:
        return "hard"
    raw = "" if difficulty is None else str(difficulty).strip().lower()
    aliases = {
        "": "normal",
        "medium": "normal",
        "media": "normal",
        "normal": "normal",
        "facil": "easy",
        "fácil": "easy",
        "easy": "easy",
        "basica": "easy",
        "básica": "easy",
        "avanzado": "hard",
        "avanzada": "hard",
        "dificil": "hard",
        "difícil": "hard",
        "hard": "hard",
    }
    return aliases.get(raw, "normal")


def _difficulty_metadata(level: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    data = dict(metadata or {})
    data["difficulty"] = level
    data["hard"] = level == "hard"
    data["easy"] = level == "easy"
    return data


def _finish_exercise(exercise: dict[str, Any], level: str) -> dict[str, Any]:
    topic = exercise.get("topic")
    if topic in TOPICS:
        exercise["topic_label"] = TOPICS[topic]
    exercise["metadata"] = _difficulty_metadata(level, exercise.get("metadata"))
    return exercise


def _choose_subtype_for_level(rng: random.Random, topic: str, level: str) -> str:
    cats = TOPIC_DIFFICULTY_CATEGORIES[topic]
    return rng.choice(cats[level])


def _all_category_subtypes(topic: str) -> list[str]:
    seen: list[str] = []
    for level in DIFFICULTY_LEVELS:
        for subtype in TOPIC_DIFFICULTY_CATEGORIES[topic][level]:
            if subtype not in seen:
                seen.append(subtype)
    return seen


def _parse_bound(text: str) -> Any:
    token = text.strip()
    if token.lower() in {"oo", "+oo", "inf", "+inf", "infty", "+infty"}:
        return sp.oo
    if token.lower() in {"-oo", "-inf", "-infty"}:
        return -sp.oo
    return _parse_math(token)


def _as_real_set(expr: Any) -> sp.Set:
    if expr is True or expr == sp.S.true:
        return sp.S.Reals
    if expr is False or expr == sp.S.false:
        return sp.EmptySet
    if isinstance(expr, sp.Set):
        return expr
    if isinstance(expr, (list, tuple, set)):
        return sp.FiniteSet(*expr)
    return expr.as_set()


def _parse_real_set_answer(raw: str) -> sp.Set:
    text = str(raw).strip()
    compact = text.lower().replace(" ", "")
    if compact in {"r", "reals", "real", "todoslosnumerosreales", "todoslosreales"}:
        return sp.S.Reals
    if compact in {"", "none", "sin solucion", "sinsolucion", "vacio", "conjuntovacio", "emptyset", "{}", "[]", "∅"}:
        return sp.EmptySet

    normalized = (
        text.replace("∞", "oo")
        .replace("−", "-")
        .replace("∪", "|")
        .replace(" U ", "|")
        .replace(" u ", "|")
        .replace(" o ", "|")
    )

    interval_pattern = re.compile(r"([\(\[])\s*([^,\[\]\(\)]+)\s*,\s*([^,\[\]\(\)]+)\s*([\)\]])")
    intervals = []
    for match in interval_pattern.finditer(normalized):
        left_bracket, left_raw, right_raw, right_bracket = match.groups()
        left = _parse_bound(left_raw)
        right = _parse_bound(right_raw)
        intervals.append(
            sp.Interval(
                left,
                right,
                left_open=left_bracket == "(",
                right_open=right_bracket == ")",
            )
        )
    if intervals:
        return sp.Union(*intervals)

    chain = re.fullmatch(
        r"\s*([^<>=]+)\s*(<=|<)\s*x\s*(<=|<)\s*([^<>=]+)\s*",
        normalized,
    )
    if chain:
        left_raw, left_op, right_op, right_raw = chain.groups()
        return sp.Interval(
            _parse_bound(left_raw),
            _parse_bound(right_raw),
            left_open=left_op == "<",
            right_open=right_op == "<",
        )

    expr = _parse_math(normalized)
    return _as_real_set(expr)


def _sets_equal(left: sp.Set, right: sp.Set) -> bool:
    try:
        result = left.equals(right)
        if result is not None:
            return bool(result)
    except Exception:
        pass
    try:
        return sp.simplify(left.symmetric_difference(right)) == sp.EmptySet
    except Exception:
        return left == right


def _outer_pair_content(text: str) -> str:
    stripped = text.strip()
    pairs = {"(": ")", "[": "]", "{": "}"}
    if len(stripped) < 2 or stripped[0] not in pairs or stripped[-1] != pairs[stripped[0]]:
        return stripped
    depth = 0
    for index, char in enumerate(stripped):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0 and index != len(stripped) - 1:
                return stripped
    return stripped[1:-1].strip()


def _split_top_level(text: str, separators: str = ",;") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char in separators and depth == 0:
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_linear_system_answer(raw: str, variables: list[str]) -> dict[str, Any]:
    text = str(raw).strip().replace("−", "-")
    assignment_pattern = re.compile(
        r"\b([A-Za-z])\s*=\s*(.*?)(?=(?:[,;]\s*)?\b[A-Za-z]\s*=|$)"
    )
    assignments: dict[str, Any] = {}
    for var, value in assignment_pattern.findall(text):
        key = var.lower()
        if key in variables:
            cleaned = value.strip().rstrip(",;")
            if cleaned:
                assignments[key] = _parse_math(cleaned)
    if all(var in assignments for var in variables):
        return assignments

    try:
        parsed = _parse_math(text)
        if isinstance(parsed, (list, tuple, set, sp.FiniteSet)):
            values = list(parsed)
            if len(values) == len(variables):
                return {var: value for var, value in zip(variables, values)}
    except Exception:
        pass

    cleaned = _outer_pair_content(text)
    parts = _split_top_level(cleaned)
    if len(parts) == len(variables):
        return {var: _parse_math(part) for var, part in zip(variables, parts)}
    raise ValueError("Use el formato x=..., y=... o escriba los valores en orden.")


def _linear_system_expected_text(validation: dict[str, Any]) -> str:
    variables = [str(var) for var in validation.get("variables", [])]
    solutions = validation.get("solutions", {})
    pieces = []
    for var in variables:
        value = _parse_math(str(solutions.get(var)))
        pieces.append(f"{var} = {_plain(value)}")
    return ", ".join(pieces)


def validate_answer(exercise: dict[str, Any], user_answer: str) -> dict[str, Any]:
    validation = exercise.get("validation", {})
    validation_type = validation.get("type")
    if validation_type == "linear_system":
        raw = str(user_answer).strip()
        expected_text = _linear_system_expected_text(validation)
        if not raw:
            return {"correct": False, "message": "Respuesta vacia.", "expected": expected_text}
        variables = [str(var).lower() for var in validation.get("variables", [])]
        try:
            user_values = _parse_linear_system_answer(raw, variables)
            expected_values = {
                var: _parse_math(str(validation.get("solutions", {}).get(var)))
                for var in variables
            }
            correct = all(_math_equal(user_values[var], expected_values[var]) for var in variables)
            return {"correct": bool(correct), "expected": expected_text}
        except Exception as exc:
            return {"correct": False, "message": f"No se pudo interpretar la respuesta: {exc}", "expected": expected_text}

    if validation_type != "inequality":
        return _LEGACY_VALIDATE_ANSWER(exercise, user_answer)

    raw = str(user_answer).strip()
    if not raw:
        return {"correct": False, "message": "Respuesta vacia."}

    try:
        user_set = _parse_real_set_answer(raw)
        expected_set = _parse_real_set_answer(validation["solution"])
        return {
            "correct": bool(_sets_equal(user_set, expected_set)),
            "expected": str(validation["solution"]),
        }
    except Exception as exc:
        return {"correct": False, "message": f"No se pudo interpretar la respuesta: {exc}", "expected": str(validation.get("solution", ""))}


validar_respuesta = validate_answer


def _answer_as_input(exercise: dict[str, Any]) -> str:
    answer = exercise.get("answer", {})
    if answer.get("kind") == "list":
        values = answer.get("plain") or []
        return ", ".join(values) if values else "sin solucion"
    return str(answer.get("plain", ""))


def _verify_exercise(exercise: dict[str, Any]) -> bool:
    validation = exercise.get("validation", {})
    validation_type = validation.get("type")
    if not validate_answer(exercise, _answer_as_input(exercise)).get("correct"):
        return False

    try:
        if exercise.get("topic") == "factorizacion" and validation_type in {"factorization", "math_equal"}:
            display_expr = _parse_math(exercise["display"]["plain"])
            expected_expr = _parse_math(validation["answer_expr"])
            return bool(_math_equal(sp.expand(display_expr), sp.expand(expected_expr)))

        if validation_type == "solution_set" and exercise.get("display", {}).get("kind") == "math":
            expected_values = [_parse_math(item) for item in validation.get("solutions", [])]
            display_expr = _parse_math(exercise["display"]["plain"])
            if isinstance(display_expr, sp.Equality):
                solved = _sorted_solutions(sp.solve(display_expr, sp.Symbol(validation.get("variable", "x"), real=True)))
                solved_real = [sol for sol in solved if sp.im(sol).simplify() == 0]
                return _same_solution_set(solved_real, expected_values)

        if validation_type == "linear_system":
            coeffs = [[_parse_math(str(value)) for value in row] for row in validation.get("coefficient_matrix", [])]
            constants = [_parse_math(str(value)) for value in validation.get("constants", [])]
            variables = [str(var) for var in validation.get("variables", [])]
            solution = [_parse_math(str(validation.get("solutions", {}).get(var))) for var in variables]
            matrix = sp.Matrix(coeffs)
            if matrix.rows != len(variables) or matrix.cols != len(variables) or matrix.det() == 0:
                return False
            return list(matrix * sp.Matrix(solution)) == constants

        if validation_type == "inequality":
            display_expr = _parse_math(exercise["display"]["plain"])
            expected_set = _parse_real_set_answer(validation["solution"])
            solved_set = _as_real_set(_solve_inequality(display_expr))
            return _sets_equal(solved_set, expected_set)

        if validation_type == "rationalization":
            expected_expr = _parse_math(validation["answer_expr"], positive_symbols=True)
            return _target_is_rationalized(expected_expr, validation.get("objective", "denominador"))
    except Exception:
        return False

    return True


SYSTEM_VARIABLES_2 = (X, sp.Symbol("y", real=True))
SYSTEM_VARIABLES_3 = (X, sp.Symbol("y", real=True), sp.Symbol("z", real=True))

SYSTEM_TEMPLATES = {
    "easy": [
        "despejada_x", "despejada_y", "suma_resta", "resta_suma",
        "eliminacion_y", "eliminacion_x", "coeficientes_positivos",
        "coeficientes_mixtos_suaves", "doble_uno", "tres_con_uno",
    ],
    "normal": [
        "general", "negativos", "eliminacion_multiplicacion_x",
        "eliminacion_multiplicacion_y", "mixtos", "solucion_grande",
        "signos_cruzados", "dos_negativos", "coeficientes_primos",
        "casi_eliminacion", "sustitucion_oculta", "igualacion_oculta",
        "constantes_grandes", "determinante_medio", "fila_densa",
    ],
    "hard": [
        "2x2_grandes", "2x2_eliminacion_exigente", "2x2_mixtos_negativos",
        "2x2_fracciones", "2x2_signos_cruzados", "2x2_determinante_alto",
        "2x2_casi_proporcional", "2x2_coeficientes_primos",
        "2x2_constantes_altas", "2x2_fila_densa",
        "3x3_pequenos", "3x3_mixtos", "3x3_grandes",
        "3x3_signos_cruzados", "3x3_denso", "3x3_determinante_medio",
        "3x3_constantes_altas", "3x3_fila_negativa",
        "3x3_eliminacion_larga", "3x3_examen",
    ],
}


def _system_variables(dimension: int) -> tuple[sp.Symbol, ...]:
    return SYSTEM_VARIABLES_3 if dimension == 3 else SYSTEM_VARIABLES_2


def _linear_combination(row: list[int], variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.Add(*[sp.Integer(coef) * var for coef, var in zip(row, variables)])


def _system_equations_from_matrix(
    coeffs: list[list[int]],
    variables: tuple[sp.Symbol, ...],
    constants: list[Any],
) -> list[sp.Equality]:
    return [
        sp.Eq(_linear_combination(row, variables), sp.simplify(value), evaluate=False)
        for row, value in zip(coeffs, constants)
    ]


def _system_plain(equations: list[sp.Equality]) -> str:
    return "\n".join(f"{_plain(eq.lhs)} = {_plain(eq.rhs)}" for eq in equations)


def _system_latex(equations: list[sp.Equality]) -> str:
    rows = [f"{sp.latex(eq.lhs)} = {sp.latex(eq.rhs)}" for eq in equations]
    return r"\begin{cases}" + r" \\ ".join(rows) + r"\end{cases}"


def _system_answer_payload(
    variables: tuple[sp.Symbol, ...],
    solution: list[Any],
) -> dict[str, Any]:
    return {
        "label": "Solucion del sistema",
        "kind": "list",
        "plain": [f"{var} = {_plain(value)}" for var, value in zip(variables, solution)],
        "latex": [f"{sp.latex(var)} = {sp.latex(value)}" for var, value in zip(variables, solution)],
    }


def _new_system_exercise(
    level: str,
    template: str,
    coeffs: list[list[int]],
    solution: list[Any],
    statement: str,
    display_equations: list[sp.Equality] | None = None,
) -> dict[str, Any]:
    dimension = len(solution)
    variables = _system_variables(dimension)
    matrix = sp.Matrix(coeffs)
    constants = [sp.simplify(value) for value in list(matrix * sp.Matrix(solution))]
    equations = display_equations or _system_equations_from_matrix(coeffs, variables, constants)
    validation = {
        "type": "linear_system",
        "variables": [str(var) for var in variables],
        "solutions": {str(var): str(value) for var, value in zip(variables, solution)},
        "coefficient_matrix": [[int(value) for value in row] for row in coeffs],
        "constants": [str(value) for value in constants],
    }
    exercise = _new_exercise(
        "ecuaciones",
        "sistemas_ecuaciones",
        statement,
        "text",
        _system_plain(equations),
        "Solucion del sistema",
        validation,
        answer_label="Solucion del sistema",
        metadata={
            "template": template,
            "dimension": dimension,
            "variables": [str(var) for var in variables],
            "exact_solution": {str(var): str(value) for var, value in zip(variables, solution)},
            "coefficient_matrix": validation["coefficient_matrix"],
            "constants": validation["constants"],
            "determinant": str(matrix.det()),
            "generation": "inverse_construction",
        },
    )
    exercise["display"] = {"kind": "math", "plain": _system_plain(equations), "latex": _system_latex(equations)}
    exercise["answer"] = _system_answer_payload(variables, solution)
    return exercise


def _integer_solution(rng: random.Random, dimension: int, low: int, high: int) -> list[sp.Integer]:
    for _ in range(40):
        values = [sp.Integer(rng.randint(low, high)) for _ in range(dimension)]
        if any(value != 0 for value in values):
            return values
    return [sp.Integer(i + 1) for i in range(dimension)]


def _simple_fraction(rng: random.Random) -> sp.Rational:
    for _ in range(40):
        den = rng.choice([2, 3, 4, 5])
        num = rng.randint(-12, 12)
        if num and num % den != 0:
            return sp.Rational(num, den)
    return sp.Rational(3, 2)


def _fraction_solution_2x2(rng: random.Random) -> list[sp.Rational]:
    x_val = _simple_fraction(rng)
    y_val = _simple_fraction(rng)
    while y_val == x_val:
        y_val = _simple_fraction(rng)
    return [x_val, y_val]


def _row_with_density(
    rng: random.Random,
    dimension: int,
    low: int,
    high: int,
    min_nonzero: int,
) -> list[int]:
    for _ in range(80):
        row = [rng.randint(low, high) for _ in range(dimension)]
        if sum(1 for value in row if value != 0) >= min_nonzero:
            return row
    return [1 if i == 0 else rng.randint(1, max(1, high)) for i in range(dimension)]


def _random_system_matrix(
    rng: random.Random,
    dimension: int,
    low: int,
    high: int,
    *,
    min_nonzero: int = 2,
    require_negative: bool = False,
    require_large: bool = False,
    integer_constants_for: list[Any] | None = None,
) -> list[list[int]]:
    for _ in range(240):
        coeffs = [
            _row_with_density(rng, dimension, low, high, min_nonzero)
            for _ in range(dimension)
        ]
        matrix = sp.Matrix(coeffs)
        if matrix.det() == 0:
            continue
        flat = [value for row in coeffs for value in row]
        if require_negative and not any(value < 0 for value in flat):
            continue
        if require_large and max(abs(value) for value in flat) < min(abs(low), abs(high), 7):
            continue
        if integer_constants_for is not None:
            constants = list(matrix * sp.Matrix(integer_constants_for))
            if not all(sp.simplify(value).is_Integer for value in constants):
                continue
        return coeffs
    if dimension == 3:
        return [[2, -1, 1], [1, 3, -2], [4, -2, 5]]
    return [[2, 1], [1, -1]]


def _easy_system_shape(
    rng: random.Random,
    solution: list[Any],
) -> tuple[str, list[list[int]], list[sp.Equality] | None]:
    x, y = SYSTEM_VARIABLES_2
    x_val, y_val = solution
    template = rng.choice(SYSTEM_TEMPLATES["easy"])

    if template == "despejada_x":
        diff = sp.simplify(x_val - y_val)
        coeffs = [[1, -1], [1, 1]]
        equations = [
            sp.Eq(x, y + diff, evaluate=False),
            sp.Eq(x + y, x_val + y_val, evaluate=False),
        ]
    elif template == "despejada_y":
        a = rng.choice([-2, -1, 1, 2])
        diff = sp.simplify(y_val - a * x_val)
        coeffs = [[-a, 1], [1, 1]]
        equations = [
            sp.Eq(y, a * x + diff, evaluate=False),
            sp.Eq(x + y, x_val + y_val, evaluate=False),
        ]
    elif template == "suma_resta":
        coeffs = [[1, 1], [1, -1]]
        equations = None
    elif template == "resta_suma":
        coeffs = [[1, -1], [1, 1]]
        equations = None
    elif template == "eliminacion_y":
        a, b = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        coeffs = [[a, 1], [b, 1]]
        equations = None
    elif template == "eliminacion_x":
        a, b = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        coeffs = [[1, a], [1, b]]
        equations = None
    elif template == "doble_uno":
        coeffs = [[2, 1], [1, -1]]
        equations = None
    elif template == "tres_con_uno":
        coeffs = [[3, 1], [1, 1]]
        equations = None
    elif template == "coeficientes_positivos":
        coeffs = _random_system_matrix(rng, 2, 1, 5, min_nonzero=2)
        equations = None
    else:
        coeffs = _random_system_matrix(rng, 2, -5, 5, min_nonzero=2)
        equations = None

    return template, coeffs, equations


def _system_equations_exercise(rng: random.Random, level: str) -> dict[str, Any]:
    if level == "easy":
        solution = _integer_solution(rng, 2, -10, 10)
        template, coeffs, display_equations = _easy_system_shape(rng, solution)
        statement = "Resuelva el sistema de ecuaciones 2x2."
        return _new_system_exercise(level, template, coeffs, solution, statement, display_equations)

    if level == "normal":
        template = rng.choice(SYSTEM_TEMPLATES["normal"])
        solution = _integer_solution(rng, 2, -15, 15)
        require_negative = template in {"negativos", "signos_cruzados", "dos_negativos", "mixtos"}
        require_large = template in {"solucion_grande", "constantes_grandes", "determinante_medio"}
        coeffs = _random_system_matrix(
            rng,
            2,
            -12,
            12,
            min_nonzero=2,
            require_negative=require_negative,
            require_large=require_large,
        )
        statement = "Resuelva el sistema de ecuaciones 2x2 usando sustitucion, igualacion o eliminacion."
        return _new_system_exercise(level, template, coeffs, solution, statement)

    template = rng.choice(SYSTEM_TEMPLATES["hard"])
    if template.startswith("3x3"):
        solution = _integer_solution(rng, 3, -9, 9)
        require_negative = template in {"3x3_mixtos", "3x3_signos_cruzados", "3x3_fila_negativa", "3x3_examen"}
        require_large = template in {"3x3_grandes", "3x3_constantes_altas", "3x3_eliminacion_larga", "3x3_examen"}
        high = 9 if require_large else 7
        coeffs = _random_system_matrix(
            rng,
            3,
            -high,
            high,
            min_nonzero=2,
            require_negative=require_negative,
            require_large=require_large,
        )
    elif template == "2x2_fracciones":
        solution = _fraction_solution_2x2(rng)
        coeffs = _random_system_matrix(
            rng,
            2,
            -14,
            14,
            min_nonzero=2,
            require_negative=True,
            require_large=True,
            integer_constants_for=solution,
        )
    else:
        solution = _integer_solution(rng, 2, -18, 18)
        require_negative = template in {"2x2_mixtos_negativos", "2x2_signos_cruzados", "2x2_casi_proporcional"}
        require_large = template in {"2x2_grandes", "2x2_eliminacion_exigente", "2x2_determinante_alto", "2x2_constantes_altas"}
        coeffs = _random_system_matrix(
            rng,
            2,
            -14,
            14,
            min_nonzero=2,
            require_negative=require_negative,
            require_large=require_large,
        )

    statement = "Resuelva el sistema de ecuaciones. Elija el metodo algebraico mas conveniente."
    return _new_system_exercise(level, template, coeffs, solution, statement)


def _easy_equation_exercise(rng: random.Random, subtype: str) -> dict[str, Any]:
    x = X
    if subtype == "cuadratica_raices_enteras":
        root = rng.randint(2, 7)
        equation = sp.Eq(x**2 - root**2, 0)
        solutions = [-root, root]
        return _new_exercise(
            "ecuaciones",
            subtype,
            "Resuelva la cuadratica factorizable directa.",
            "math",
            equation,
            solutions,
            {"type": "solution_set", "variable": "x", "solutions": [str(s) for s in solutions]},
            answer_label="Soluciones",
        )

    addend = rng.randint(1, 9)
    result = rng.randint(addend + 1, addend + 12)
    equation = sp.Eq(x + addend, result)
    solution = result - addend
    return _new_exercise(
        "ecuaciones",
        "lineal_basica",
        "Resuelva la ecuacion lineal directa.",
        "math",
        equation,
        [sp.Integer(solution)],
        {"type": "solution_set", "variable": "x", "solutions": [str(solution)]},
        answer_label="Solucion",
    )


def _easy_factorization_exercise(rng: random.Random, subtype: str, direction: str | None = None) -> dict[str, Any]:
    x = X
    direction = direction or "reducir"

    if subtype == "diferencia_cuadrados":
        n = rng.randint(2, 9)
        question = x**2 - n**2
        answer = (x - n) * (x + n)
        statement = "Factorice la diferencia de cuadrados directa."
    elif subtype == "trinomio_cuadrado_perfecto":
        n = rng.randint(2, 8)
        question = x**2 + 2 * n * x + n**2
        answer = (x + n) ** 2
        statement = "Reconozca y factorice el trinomio cuadrado perfecto."
    elif subtype == "trinomio_inspeccion":
        r1, r2 = rng.sample(range(1, 8), 2)
        answer = (x + r1) * (x + r2)
        question = sp.expand(answer)
        statement = "Factorice el trinomio por inspeccion directa."
    elif subtype == "completar_cuadrado":
        h = rng.randint(2, 7)
        question = x**2 + 2 * h * x + h**2
        answer = (x + h) ** 2
        statement = "Complete el cuadrado y escriba el resultado factorizado."
    else:
        factor = rng.randint(2, 9)
        constant = rng.randint(1, 8)
        question = factor * x + factor * constant
        answer = sp.Mul(factor, x + constant, evaluate=False)
        subtype = "factor_comun"
        statement = "Extraiga el factor comun evidente."

    if direction == "expandir" and subtype in EXPANDABLE_SUBTYPES:
        display_expr = answer
        expected_expr = sp.expand(question)
        validation_type = "math_equal"
        statement = "Expanda la expresion usando la identidad visible."
    else:
        display_expr = sp.expand(question)
        expected_expr = answer
        validation_type = "factorization"

    return _new_exercise(
        "factorizacion",
        subtype,
        statement,
        "math",
        display_expr,
        expected_expr,
        {"type": validation_type, "answer_expr": str(expected_expr), "allow_additive": False},
        answer_label="Resultado",
        metadata={"direction": direction},
    )


def _completing_square_case_exercise(rng: random.Random, subtype: str, level: str) -> dict[str, Any]:
    x = X
    n = rng.choice([1, 2] if level == "normal" else [2, 3])
    u = x**n
    s = rng.randint(1, 3 if level == "normal" else 5)

    if subtype == "completar_cuadrado_binomio":
        t = rng.randint(2, 6 if level == "normal" else 10)
        question = sp.expand(s**2 * u**2 - t**2)
        answer = sp.factor((s * u - t) * (s * u + t))
        statement = (
            "Factorice la forma ax^(2n)+c como diferencia de cuadrados. "
            "Use multiplicar/dividir por el conjugado cuando sea necesario."
        )
    else:
        h = _nonzero_int(rng, -4 if level == "normal" else -8, 4 if level == "normal" else 8)
        t = rng.randint(1, 6 if level == "normal" else 10)
        question = sp.expand(s**2 * (u + h) ** 2 - t**2)
        b_over_2a = sp.Rational(2 * s**2 * h, 2 * s**2)
        answer = sp.factor((s * (u + h) - t) * (s * (u + h) + t))
        statement = (
            "Factorice completando cuadrado: sume y reste (b/2a)^2 "
            f"(aqui b/2a = {b_over_2a}) y termine la factorizacion."
        )

    return _new_exercise(
        "factorizacion",
        subtype,
        statement,
        "math",
        question,
        answer,
        {"type": "factorization", "answer_expr": str(answer), "allow_additive": False},
        answer_label="Factorizacion completa",
        metadata={"method": "completar_cuadrado", "direction": "reducir", "power_n": n},
    )


def _easy_rationalization_exercise(rng: random.Random, subtype: str) -> dict[str, Any]:
    n = rng.choice([2, 3, 5, 6, 7, 10])
    coef = rng.randint(1, 5)
    expr = _unevaluated_fraction(coef, sp.sqrt(n))
    answer = sp.simplify(coef * sp.sqrt(n) / n)
    return _new_exercise(
        "racionalizacion",
        "raiz_cuadrada_simple",
        "Racionalice el denominador con una sola raiz cuadrada.",
        "math",
        expr,
        answer,
        {"type": "rationalization", "answer_expr": str(answer), "objective": "denominador", "strict": True},
        answer_label="Resultado racionalizado",
        metadata={"objective": "denominador"},
    )


def _easy_inequality_exercise(rng: random.Random, subtype: str) -> dict[str, Any]:
    x = X
    addend = rng.randint(1, 9)
    limit = rng.randint(addend + 1, addend + 12)
    op = rng.choice([sp.Lt, sp.Le, sp.Gt, sp.Ge])
    rel = op(x + addend, limit)
    answer = _solve_inequality(rel)
    return _new_exercise(
        "inecuaciones",
        "inecuacion_lineal",
        "Resuelva la inecuacion lineal basica.",
        "math",
        rel,
        answer,
        {"type": "inequality", "solution": str(answer)},
        answer_label="Conjunto solucion",
    )


def _easy_absolute_value_exercise(rng: random.Random, subtype: str) -> dict[str, Any]:
    x = X
    if subtype == "valor_absoluto_inecuacion":
        c = rng.randint(2, 8)
        rel = rng.choice([sp.Lt, sp.Le])(sp.Abs(x), c)
        answer = _solve_inequality(rel)
        validation = {"type": "inequality", "solution": str(answer)}
        answer_label = "Conjunto solucion"
        answer_value = answer
        statement = "Resuelva la inecuacion basica de valor absoluto."
    else:
        c = rng.randint(2, 8)
        rel = sp.Eq(sp.Abs(x), c)
        answer_value = [-c, c]
        validation = {"type": "solution_set", "variable": "x", "solutions": [str(-c), str(c)]}
        answer_label = "Soluciones"
        statement = "Resuelva la ecuacion basica de valor absoluto."

    return _new_exercise(
        "valor_absoluto",
        subtype if subtype in ABSOLUTE_VALUE_SUBTYPES else "valor_absoluto_ecuacion",
        statement,
        "math",
        rel,
        answer_value,
        validation,
        answer_label=answer_label,
    )


def generate_absolute_value_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
    difficulty: str | None = None,
) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "valor_absoluto", level)
    if subtype not in ABSOLUTE_VALUE_SUBTYPES:
        raise ValueError(f"Subtipo de valor absoluto desconocido: {subtype}")
    if level == "easy" and subtype in {"valor_absoluto_ecuacion", "valor_absoluto_inecuacion"}:
        return _finish_exercise(_easy_absolute_value_exercise(rng, subtype), level)

    current_inequality_subtypes = globals()["INEQUALITY_SUBTYPES"]
    with _INEQUALITY_SUBTYPE_LOCK:
        try:
            globals()["INEQUALITY_SUBTYPES"] = _ORIGINAL_INEQUALITY_SUBTYPES
            exercise = _LEGACY_GENERATORS["inecuaciones"](rng, subtype, hard=(level == "hard"))
        finally:
            globals()["INEQUALITY_SUBTYPES"] = current_inequality_subtypes
    exercise["topic"] = "valor_absoluto"
    exercise["topic_label"] = TOPICS["valor_absoluto"]
    return _finish_exercise(exercise, level)


def _generate_topic_exercise(
    rng: random.Random,
    topic: str,
    subtype: str,
    level: str,
    direction: str | None = None,
) -> dict[str, Any]:
    if topic == "valor_absoluto":
        return generate_absolute_value_exercise(rng, subtype, difficulty=level)

    if topic == "ecuaciones" and subtype == "sistemas_ecuaciones":
        return _finish_exercise(_system_equations_exercise(rng, level), level)

    if topic == "ecuaciones" and level == "easy" and subtype in TOPIC_DIFFICULTY_CATEGORIES["ecuaciones"]["easy"]:
        return _finish_exercise(_easy_equation_exercise(rng, subtype), level)

    if topic == "factorizacion":
        if subtype in {"completar_cuadrado_trinomio", "completar_cuadrado_binomio"}:
            return _finish_exercise(_completing_square_case_exercise(rng, subtype, level), level)
        if level == "easy" and subtype in TOPIC_DIFFICULTY_CATEGORIES["factorizacion"]["easy"]:
            return _finish_exercise(_easy_factorization_exercise(rng, subtype, direction), level)

    if topic == "racionalizacion" and level == "easy" and subtype in TOPIC_DIFFICULTY_CATEGORIES["racionalizacion"]["easy"]:
        return _finish_exercise(_easy_rationalization_exercise(rng, subtype), level)

    if topic == "inecuaciones" and level == "easy" and subtype in TOPIC_DIFFICULTY_CATEGORIES["inecuaciones"]["easy"]:
        return _finish_exercise(_easy_inequality_exercise(rng, subtype), level)

    if topic == "binomio_newton":
        return generate_binomio_newton_exercise(rng, subtype, difficulty=level)

    if topic == "conversiones":
        exercise = _LEGACY_GENERATORS[topic](rng, subtype, hard=(level == "hard"))
    elif topic == "factorizacion":
        exercise = _LEGACY_GENERATORS[topic](rng, subtype, hard=(level == "hard"), direction=direction)
    else:
        exercise = _LEGACY_GENERATORS[topic](rng, subtype, hard=(level == "hard"))
    return _finish_exercise(exercise, level)


def _generate_verified_exercise(
    rng: random.Random,
    topic: str,
    subtype: str,
    level: str,
    direction: str | None = None,
    attempts: int = 8,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            exercise = _generate_topic_exercise(rng, topic, subtype, level, direction)
            if _verify_exercise(exercise):
                return exercise
        except Exception as exc:
            last_error = exc
    if last_error:
        raise ValueError(f"No se pudo generar un ejercicio valido: {last_error}") from last_error
    raise ValueError("No se pudo generar un ejercicio matematicamente valido tras varios reintentos.")


def generate_conversion_exercise(rng: random.Random, subtype: str | None = None, *, hard: bool = False, difficulty: str | None = None) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "conversiones", level)
    return _generate_verified_exercise(rng, "conversiones", subtype, level)


def generate_equation_exercise(rng: random.Random, subtype: str | None = None, *, hard: bool = False, difficulty: str | None = None) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "ecuaciones", level)
    return _generate_verified_exercise(rng, "ecuaciones", subtype, level)


def generate_factorization_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
    difficulty: str | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "factorizacion", level)
    return _generate_verified_exercise(rng, "factorizacion", subtype, level, direction)


def generate_rationalization_exercise(rng: random.Random, subtype: str | None = None, *, hard: bool = False, difficulty: str | None = None) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "racionalizacion", level)
    return _generate_verified_exercise(rng, "racionalizacion", subtype, level)


def generate_inequality_exercise(rng: random.Random, subtype: str | None = None, *, hard: bool = False, difficulty: str | None = None) -> dict[str, Any]:
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or _choose_subtype_for_level(rng, "inecuaciones", level)
    if subtype in ABSOLUTE_VALUE_SUBTYPES:
        return generate_absolute_value_exercise(rng, subtype, difficulty=level)
    if subtype not in INEQUALITY_SUBTYPES:
        raise ValueError(f"Subtipo de inecuacion desconocido: {subtype}")
    return _generate_verified_exercise(rng, "inecuaciones", subtype, level)


GENERATOR_BY_TOPIC = {
    "conversiones": generate_conversion_exercise,
    "ecuaciones": generate_equation_exercise,
    "factorizacion": generate_factorization_exercise,
    "racionalizacion": generate_rationalization_exercise,
    "inecuaciones": generate_inequality_exercise,
    "valor_absoluto": generate_absolute_value_exercise,
}


def _exercise_fingerprint(exercise: dict[str, Any]) -> str:
    """Huella única de un ejercicio para detectar repeticiones.

    Combina el subtipo con la representación plana de la expresión mostrada,
    lo que permite detectar ejercicios con el mismo contenido matemático
    independientemente de su id UUID.
    """
    display = exercise.get("display", {})
    return f"{exercise.get('subtype', '')}||{display.get('plain', '')}"


def generate_exercise(topic, subtype=None, seed=None, difficulty=None, direction=None, seen_fingerprints=None) -> dict[str, Any]:
    topic = str(topic).strip().lower()
    level = _normalize_difficulty(difficulty)
    if topic == "inecuaciones" and subtype in ABSOLUTE_VALUE_SUBTYPES:
        topic = "valor_absoluto"
    if topic not in TOPICS:
        raise ValueError(f"Tema desconocido: {topic}")
    rng = _make_rng(seed)
    subtype = subtype or _choose_subtype_for_level(rng, topic, level)
    if subtype not in SUBTYPES_BY_TOPIC[topic]:
        raise ValueError(f"Subtipo desconocido para {topic}: {subtype}")

    seen_set = set(seen_fingerprints or [])
    max_attempts = 15 if seen_set else 1
    last_exercise: dict[str, Any] | None = None

    for attempt in range(max_attempts):
        # En el primer intento respetamos la semilla (reproducibilidad).
        # En los siguientes usamos RNG aleatorio para variar el ejercicio.
        attempt_seed = seed if attempt == 0 else None
        attempt_rng = _make_rng(attempt_seed)
        last_exercise = _generate_verified_exercise(attempt_rng, topic, subtype, level, direction)
        if not seen_set or _exercise_fingerprint(last_exercise) not in seen_set:
            return last_exercise

    return last_exercise  # fallback: devolver el último aunque sea duplicado


def _exam_tier_plan(rng: random.Random, level: str, quantity: int) -> list[str]:
    if level == "easy":
        weights = [0.80, 0.20, 0.00]
        ordered = True
    elif level == "hard":
        weights = [0.10, 0.30, 0.60]
        ordered = False
    else:
        weights = [0.35, 0.50, 0.15]
        ordered = True

    tiers = [_weighted_choice(rng, list(DIFFICULTY_LEVELS), weights) for _ in range(quantity)]
    if ordered:
        order = {"easy": 0, "normal": 1, "hard": 2}
        tiers.sort(key=lambda item: order[item])
    else:
        rng.shuffle(tiers)
    return tiers


def _generate_unique_question(
    rng: random.Random,
    topic: str,
    subtype: str,
    level: str,
    seen_fingerprints: set[str],
    max_retries: int = 8,
    direction: str | None = None,
) -> dict[str, Any]:
    """Genera una pregunta intentando evitar duplicados.

    Prueba hasta max_retries veces. Si todas las tentativas producen un
    ejercicio ya visto, devuelve el último generado como fallback para no
    bloquear la generación del examen completo.
    """
    last: dict[str, Any] | None = None
    for _ in range(max_retries):
        candidate = _generate_verified_exercise(rng, topic, subtype, level, direction)
        fp = _exercise_fingerprint(candidate)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            return candidate
        last = candidate
    # Fallback: retornar el último aunque se repita
    if last is not None:
        seen_fingerprints.add(_exercise_fingerprint(last))
        return last
    raise RuntimeError("_generate_unique_question: no se pudo generar ningún candidato.")


def generate_exam(topic, quantity, seed=None, subtypes=None, difficulty=None, title=None):
    topic = str(topic).strip().lower()
    if topic not in TOPICS:
        raise ValueError(f"Tema desconocido: {topic}")

    level = _normalize_difficulty(difficulty)
    quantity = _limit_quantity(quantity)
    rng = _make_rng(seed)
    questions: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()   # deduplicación interna del examen
    _MAX_DEDUP_RETRIES = 8

    if subtypes:
        allowed = [str(item) for item in subtypes if str(item) in SUBTYPES_BY_TOPIC[topic]]
        if not allowed:
            allowed = SUBTYPES_BY_TOPIC[topic]
        for _ in range(quantity):
            subtype = rng.choice(allowed)
            question = _generate_unique_question(rng, topic, subtype, level, seen_fingerprints, _MAX_DEDUP_RETRIES)
            questions.append(question)
    else:
        for tier in _exam_tier_plan(rng, level, quantity):
            subtype = rng.choice(TOPIC_DIFFICULTY_CATEGORIES[topic][tier])
            question = _generate_unique_question(rng, topic, subtype, tier, seen_fingerprints, _MAX_DEDUP_RETRIES)
            questions.append(question)

    exam_title = title or TOPIC_TITLES[topic]
    if title is None:
        exam_title = f"{exam_title} - {DIFFICULTY_LABELS[level]}"

    return {
        "title": exam_title,
        "topic": topic,
        "topic_label": TOPICS[topic],
        "difficulty": level,
        "difficulty_label": DIFFICULTY_LABELS[level],
        "quantity": quantity,
        "questions": questions,
    }


generar_ejercicio = generate_exercise
generar_examen = generate_exam


# =============================================================================
# BINOMIO DE NEWTON
# =============================================================================
# Integración del módulo _Binomio_de_Newton.py al sistema de temas.
#
# Subtipos (tipos de pregunta del archivo original):
#   tipo_1 → coeficiente_posicion_k
#   tipo_2 → termino_posicion_k
#   tipo_3 → polinomio_completo
#   tipo_4 → terminos_rango
#   tipo_5 → posicion_de_potencia
#
# Dificultades mapeadas al sistema easy/normal/hard:
#   easy   → 'facil'   (letras simples, n ≤ 9)
#   normal → 'normal'  (coeficientes mixtos, n ≤ 14)
#   hard   → 'avanzado' (polinomios complejos, n ≤ 20, posible n simbólico)
# =============================================================================

from math import comb as _comb

# Constantes internas del generador de binomio
_BN_LETRAS = [c for c in "abcdfghjklmnpqrstuvwxyz"]
_BN_MAX_COMPLETO = 10
_BN_MAX_EXP = {"easy": 9, "normal": 14, "hard": 20}


# ── Generadores de binomios por dificultad ────────────────────────────────────

def _bn_coef_aleatorio(rng: random.Random, rango: int = 20, permitir_fraccion: bool = False) -> sp.Expr:
    if permitir_fraccion and rng.random() < 0.4:
        num = rng.choice([i for i in range(-10, 11) if i != 0])
        den = rng.randint(1, 10)
        return sp.Rational(num, den)
    v = rng.choice([i for i in range(-rango, rango + 1) if i != 0])
    return sp.Integer(v)


def _bn_polinomio_facil(rng: random.Random) -> sp.Expr:
    letras_pool = rng.sample(_BN_LETRAS, 2)
    x = sp.Symbol(letras_pool[0])
    y = sp.Symbol(letras_pool[1])
    exp_x = rng.choice([1, 1, 1, 2, 3])
    exp_y = rng.choice([1, 1, 1, 2, 3])
    cx = rng.choice([-2, -1, 1, 1, 1, 2])
    cy = rng.choice([-2, -1, 1, 1, 1, 2])
    return cx * x**exp_x + cy * y**exp_y


def _bn_polinomio_normal(rng: random.Random) -> sp.Expr:
    letras_pool = rng.sample(_BN_LETRAS, 2)
    x = sp.Symbol(letras_pool[0])
    y = sp.Symbol(letras_pool[1])
    exp_x = rng.choice([1, 1, 2])
    exp_y = rng.choice([1, 1, 2])

    def coef_normal():
        tipo = rng.choice(["entero", "fraccion", "raiz"])
        if tipo == "entero":
            return _bn_coef_aleatorio(rng, 20)
        elif tipo == "fraccion":
            num = rng.choice([i for i in range(-10, 11) if i != 0])
            den = rng.randint(1, 10)
            return sp.Rational(num, den)
        else:
            base = rng.choice([2, 3, 5, 7])
            signo = rng.choice([-1, 1])
            return signo * sp.sqrt(base)

    return coef_normal() * x**exp_x + coef_normal() * y**exp_y


def _bn_polinomio_avanzado(rng: random.Random, usar_exponente_letra: bool = False):
    letras_pool = rng.sample(_BN_LETRAS, 2)
    x = sp.Symbol(letras_pool[0])
    y = sp.Symbol(letras_pool[1])
    num_terminos = rng.randint(2, 3)

    def coef_avanzado():
        tipo = rng.choice(["entero", "fraccion", "raiz", "entero"])
        if tipo == "entero":
            return _bn_coef_aleatorio(rng, 50)
        elif tipo == "fraccion":
            num = rng.choice([i for i in range(-10, 11) if i != 0])
            den = rng.randint(1, 10)
            return sp.Rational(num, den)
        else:
            base = rng.choice([2, 3, 5, 6, 7, 10])
            signo = rng.choice([-1, 1])
            return signo * sp.sqrt(base)

    terminos = []
    for i in range(num_terminos):
        var = x if i % 2 == 0 else y
        exp = rng.randint(1, 5)
        c = coef_avanzado()
        terminos.append(c * var**exp)

    polinomio = sum(terminos)

    if usar_exponente_letra:
        n_sym = sp.Symbol(rng.choice([l for l in _BN_LETRAS if l not in letras_pool]))
        return polinomio, n_sym

    return polinomio, None


def _bn_seleccionar_dificultad(rng: random.Random, level: str):
    """Devuelve (polinomio, n, n_sym) según el nivel easy/normal/hard."""
    max_exp = _BN_MAX_EXP[level]
    n_sym = None

    if level == "easy":
        polinomio = _bn_polinomio_facil(rng)
        n = rng.randint(2, max_exp)
    elif level == "normal":
        polinomio = _bn_polinomio_normal(rng)
        n = rng.randint(2, max_exp)
    else:  # hard
        usar_letra = rng.random() < 0.3
        polinomio, n_sym = _bn_polinomio_avanzado(rng, usar_exponente_letra=usar_letra)
        n = None if n_sym is not None else rng.randint(2, max_exp)

    return polinomio, n, n_sym


# ── Generadores por subtipo ───────────────────────────────────────────────────

def _bn_display(polinomio: sp.Expr, n, n_sym) -> sp.Expr:
    """Construye la expresión (polinomio)^n para mostrar en el display."""
    exp = n_sym if n is None else sp.Integer(n)
    return sp.Pow(polinomio, exp, evaluate=False)


def _bn_coeficiente_posicion_k(rng: random.Random, polinomio: sp.Expr, n, n_sym, level: str) -> dict[str, Any]:
    """tipo_1: coeficiente binomial en posición k."""
    display_expr = _bn_display(polinomio, n, n_sym)

    if n is None:
        k_sym = sp.Symbol("k")
        expr = sp.binomial(n_sym, k_sym - 1)
        return _new_exercise(
            "binomio_newton", "coeficiente_posicion_k",
            f"¿Cuál es la expresión general del coeficiente en la posición k?",
            "math", display_expr,
            sp.latex(expr),
            {"type": "math_equal", "answer_expr": str(expr)},
            answer_label="Coeficiente general",
            metadata={"level": level, "n_simbolico": True},
        )

    k = rng.randint(1, n + 1)
    coef = _comb(n, k - 1)
    return _new_exercise(
        "binomio_newton", "coeficiente_posicion_k",
        f"¿Cuál es el coeficiente binomial del término en la posición k = {k}?",
        "math", display_expr,
        sp.Integer(coef),
        {"type": "math_equal", "answer_expr": str(coef)},
        answer_label=f"C({n}, {k-1}) = {coef}",
        metadata={"level": level, "n": n, "k": k},
    )


def _bn_termino_posicion_k(rng: random.Random, polinomio: sp.Expr, n, n_sym, level: str) -> dict[str, Any]:
    """tipo_2: término completo en posición k."""
    display_expr = _bn_display(polinomio, n, n_sym)
    partes = sp.Add.make_args(polinomio)
    a_e = partes[0] if len(partes) >= 1 else polinomio
    b_e = partes[1] if len(partes) >= 2 else sp.Integer(1)
    es_binomio = len(partes) == 2  # solo aplicar fórmula C(n,r)*a^(n-r)*b^r si es binomio puro

    if n is None:
        # Con exponente simbólico solo podemos dar la fórmula general binomial;
        # si el polinomio es un trinomio mostramos de todas formas la fórmula con a_e y b_e
        # indicando que es una aproximación (el caso simbólico no ocurre con trinomios en la práctica)
        k_sym = sp.Symbol("k")
        expr_general = (
            sp.binomial(n_sym, k_sym - 1)
            * a_e ** (n_sym - (k_sym - 1))
            * b_e ** (k_sym - 1)
        )
        return _new_exercise(
            "binomio_newton", "termino_posicion_k",
            "¿Cuál es el término general T_k?",
            "math", display_expr,
            sp.latex(expr_general),
            {"type": "math_equal", "answer_expr": str(expr_general)},
            answer_label="Término general T_k",
            metadata={"level": level, "n_simbolico": True},
        )

    k = rng.randint(1, n + 1)
    r = k - 1

    if es_binomio:
        # Fórmula directa C(n,r)*a^(n-r)*b^r
        coef = _comb(n, r)
        termino = sp.expand(sp.Integer(coef) * sp.Pow(a_e, n - r) * sp.Pow(b_e, r))
    else:
        # Polinomio con 3+ términos: obtener el k-ésimo término expandiendo completamente.
        # Los términos se ordenan por grado descendente para que la posición k sea consistente.
        expansion = sp.expand(sp.Pow(polinomio, n))
        todos = sp.Add.make_args(expansion)
        syms = sorted(polinomio.free_symbols, key=str)

        def _grado(t: sp.Expr) -> int:
            if not syms:
                return 0
            try:
                return sp.Poly(t, *syms).degree()
            except Exception:
                try:
                    return int(sum(t.as_powers_dict().get(s, 0) for s in syms))
                except Exception:
                    return 0

        todos_ord = sorted(todos, key=_grado, reverse=True)
        termino = todos_ord[r] if r < len(todos_ord) else sp.Integer(0)

    return _new_exercise(
        "binomio_newton", "termino_posicion_k",
        f"¿Cuál es el término en la posición k = {k}?",
        "math", display_expr,
        termino,
        {"type": "math_equal", "answer_expr": str(termino)},
        answer_label=f"T_{k} = {sp.latex(termino)}",
        metadata={"level": level, "n": n, "k": k},
    )


def _bn_polinomio_completo(rng: random.Random, polinomio: sp.Expr, n, n_sym, level: str) -> dict[str, Any]:
    """tipo_3: desarrollo completo (solo si n ≤ MAX_COMPLETO y numérico)."""
    if n is None or n > _BN_MAX_COMPLETO:
        max_exp = min(_BN_MAX_EXP[level], _BN_MAX_COMPLETO)
        n = rng.randint(2, max_exp)
    display_expr = _bn_display(polinomio, n, n_sym)
    expansion = sp.expand(sp.Pow(polinomio, n))
    return _new_exercise(
        "binomio_newton", "polinomio_completo",
        "Calcula el desarrollo completo.",
        "math", display_expr,
        expansion,
        {"type": "math_equal", "answer_expr": str(expansion)},
        answer_label=f"Desarrollo de ({sp.latex(polinomio)})^{n}",
        metadata={"level": level, "n": n},
    )


def _bn_terminos_rango(rng: random.Random, polinomio: sp.Expr, n, n_sym, level: str) -> dict[str, Any]:
    """tipo_4: términos del a-ésimo al b-ésimo."""
    if n is None:
        n = rng.randint(2, _BN_MAX_EXP[level])
    display_expr = _bn_display(polinomio, n, n_sym)

    total = n + 1
    a = rng.randint(1, total)
    b = rng.randint(a, total)

    expr = sp.expand(sp.Pow(polinomio, n))
    monomios = sp.Add.make_args(expr)
    syms = sorted(polinomio.free_symbols, key=str)

    def _sort_key(t: sp.Expr) -> int:
        """Grado total del monomio; robusto frente a radicales y no-polinomios."""
        if not syms:
            return 0
        try:
            return sp.Poly(t, *syms).degree()
        except Exception:
            # Fallback: suma de exponentes de los símbolos conocidos en el monomio
            try:
                return int(sum(t.as_powers_dict().get(s, 0) for s in syms))
            except Exception:
                return 0

    if syms:
        monomios = sorted(monomios, key=_sort_key, reverse=True)
    terminos = list(monomios)[a - 1: b]
    suma = sp.Add(*terminos) if terminos else sp.Integer(0)

    return _new_exercise(
        "binomio_newton", "terminos_rango",
        f"Calcula la suma de los términos del {a}° al {b}°.",
        "math", display_expr,
        sp.expand(suma),
        {"type": "math_equal", "answer_expr": str(sp.expand(suma))},
        answer_label=f"Suma términos {a}° a {b}°",
        metadata={"level": level, "n": n, "a": a, "b": b},
    )


def _bn_posicion_de_potencia(rng: random.Random, polinomio: sp.Expr, n, n_sym, level: str) -> dict[str, Any]:
    """tipo_5: en qué posición aparece la variable elevada a s."""
    if n is None:
        n = rng.randint(2, _BN_MAX_EXP[level])
    display_expr = _bn_display(polinomio, n, n_sym)

    s = rng.randint(0, n)
    k = n - s + 1
    syms = sorted(polinomio.free_symbols, key=str)
    var_nombre = syms[0] if syms else sp.Symbol("a")

    return _new_exercise(
        "binomio_newton", "posicion_de_potencia",
        f"¿En qué posición k aparece {sp.latex(var_nombre)}^{{{s}}}?",
        "math", display_expr,
        sp.Integer(k),
        {"type": "math_equal", "answer_expr": str(k)},
        answer_label=f"Posición k = {k}",
        metadata={"level": level, "n": n, "s": s, "k": k},
    )


_BN_GENERATORS = {
    "coeficiente_posicion_k": _bn_coeficiente_posicion_k,
    "termino_posicion_k":     _bn_termino_posicion_k,
    "polinomio_completo":     _bn_polinomio_completo,
    "terminos_rango":         _bn_terminos_rango,
    "posicion_de_potencia":   _bn_posicion_de_potencia,
}


def generate_binomio_newton_exercise(
    rng: random.Random,
    subtype: str | None = None,
    *,
    hard: bool = False,
    difficulty: str | None = None,
) -> dict[str, Any]:
    """Genera un ejercicio de Binomio de Newton.

    Parámetros:
        rng: instancia de random.Random.
        subtype: subtipo específico o None para aleatorio.
        hard: alias legacy → difficulty='hard'.
        difficulty: 'easy' | 'normal' | 'hard' (default 'normal').
    """
    level = _normalize_difficulty(difficulty, hard)
    subtype = subtype or rng.choice(BINOMIO_NEWTON_DIFFICULTY_CATEGORIES[level])
    if subtype not in BINOMIO_NEWTON_SUBTYPES:
        raise ValueError(f"Subtipo de binomio de Newton desconocido: {subtype}")

    polinomio, n, n_sym = _bn_seleccionar_dificultad(rng, level)
    fn = _BN_GENERATORS[subtype]
    exercise = fn(rng, polinomio, n, n_sym, level)
    return _finish_exercise(exercise, level)


# Registrar el generador de Binomio de Newton en los dicts globales
# (se hace aquí porque la función debe existir antes de ser referenciada)
GENERATOR_BY_TOPIC["binomio_newton"] = generate_binomio_newton_exercise
SUBTYPES_BY_TOPIC["binomio_newton"] = BINOMIO_NEWTON_SUBTYPES
HARD_SUBTYPES_BY_TOPIC["binomio_newton"] = HARD_BINOMIO_NEWTON_SUBTYPES
DEFAULT_EXAM_SUBTYPES_BY_TOPIC["binomio_newton"] = BINOMIO_NEWTON_SUBTYPES


QUICK_MODES: list[dict[str, str]] = [
    {"label": "Conversiones", "topic": "conversiones", "subtype": ""},
    {"label": "Ecuaciones", "topic": "ecuaciones", "subtype": ""},
    {"label": "Factorizacion", "topic": "factorizacion", "subtype": ""},
    {"label": "Racionalizacion", "topic": "racionalizacion", "subtype": ""},
    {"label": "Inecuaciones", "topic": "inecuaciones", "subtype": ""},
    {"label": "Valor absoluto", "topic": "valor_absoluto", "subtype": ""},
    {"label": "Binomio de Newton", "topic": "binomio_newton", "subtype": ""},
]


def get_options() -> dict[str, Any]:
    """Describe temas, subtipos y accesos rapidos para que el frontend arme sus menus.

    Forma esperada por index.html:
      { topics: {clave: etiqueta}, subtypes: {tema: [subtipos]},
        expandableSubtypes: [subtipos], quickModes: [{label, topic, subtype}] }
    """
    return {
        "topics": dict(TOPICS),
        "subtypes": {topic: list(subs) for topic, subs in SUBTYPES_BY_TOPIC.items()},
        "hardSubtypes": {topic: list(subs) for topic, subs in HARD_SUBTYPES_BY_TOPIC.items()},
        "difficultyCategories": TOPIC_DIFFICULTY_CATEGORIES,
        "difficulties": list(DIFFICULTY_LEVELS),
        "expandableSubtypes": sorted(EXPANDABLE_SUBTYPES),
        "quickModes": QUICK_MODES,
        "maxQuestions": MAX_QUESTIONS,
    }


def smoke_test(seed: int | str = 1234) -> dict[str, Any]:
    """Genera ejemplos de todos los temas en ambos modos y valida sus respuestas."""
    results: dict[str, Any] = {}
    for topic in TOPICS:
        for diff in DIFFICULTY_LEVELS:
            exam = generate_exam(topic, 3, seed=f"{seed}-{topic}-{diff}", difficulty=diff)
            validations = []
            for exercise in exam["questions"]:
                answer = exercise["answer"]
                if answer["kind"] == "list":
                    raw_answer = ", ".join(answer["plain"]) if answer["plain"] else "sin solucion"
                else:
                    raw_answer = answer["plain"]
                result = validate_answer(exercise, raw_answer)
                validations.append(result["correct"])
            results[f"{topic}_{diff}"] = all(validations)
    return results


__all__ = [
    "MAX_QUESTIONS",
    "TOPICS",
    "get_options",
    "QUICK_MODES",
    "SUBTYPES_BY_TOPIC",
    "HARD_SUBTYPES_BY_TOPIC",
    "DIFFICULTY_CATEGORIES",
    "TOPIC_DIFFICULTY_CATEGORIES",
    "DIFFICULTY_LEVELS",
    "ABSOLUTE_VALUE_SUBTYPES",
    "BINOMIO_NEWTON_SUBTYPES",
    "HARD_BINOMIO_NEWTON_SUBTYPES",
    "BINOMIO_NEWTON_DIFFICULTY_CATEGORIES",
    "EXPANDABLE_SUBTYPES",
    "generate_exercise",
    "generate_exam",
    "_exercise_fingerprint",
    "generate_conversion_exercise",
    "generate_equation_exercise",
    "generate_factorization_exercise",
    "generate_rationalization_exercise",
    "generate_inequality_exercise",
    "generate_absolute_value_exercise",
    "generate_binomio_newton_exercise",
    "validate_answer",
    "export_exam_pdf",
    "export_rationalization_exam_pdf",
    "rationalize_expression",
    "smoke_test",
    "generar_ejercicio",
    "generar_examen",
    "validar_respuesta",
    "exportar_examen_pdf",
    "exportar_examen_racionalizacion_pdf",
    "exportar_examen_pdf_bytes",
]


# =============================================================================
# ESQUEMA SUGERIDO PARA CONECTAR CON EL MENU / FRONTEND
# =============================================================================
#
# 1. El frontend muestra TOPICS y, si quieres, SUBTYPES_BY_TOPIC[tema].
#
# 2. Al iniciar un examen normal:
#      exam = generate_exam("racionalizacion", 20)
#
#    Al iniciar un examen de mayor exigencia:
#      exam = generate_exam("racionalizacion", 20, difficulty="hard")
#
# 3. Para pintar cada pregunta:
#      question["statement"]           -> instruccion breve
#      question["display"]["kind"]     -> "math" o "text"
#      question["display"]["latex"]    -> usar MathJax/KaTeX si es math
#      question["display"]["plain"]    -> fallback o texto normal
#
# 4. Al responder:
#      result = validate_answer(question, respuesta_del_estudiante)
#      result["correct"] decide si esta bien.
#
# 5. Para descargar PDF:
#      path = export_exam_pdf(exam, "ruta/examen.pdf", include_answer_key=True)
#
# 6. Rutas API recomendadas:
#      GET  /api/topics
#      POST /api/exams              body: {topic, quantity, subtypes?, seed?, difficulty?}
#      POST /api/answers/check      body: {exercise, answer}
#      POST /api/exams/pdf          body: {exam_id, include_answer_key}
#
# 7. Para el menu actual de consola, cada opcion solo deberia llamar:
#      generar_examen(tema, cantidad)                  # modo normal
#      generar_examen(tema, cantidad, difficulty="hard")  # modo examen
#      validar_respuesta(ejercicio, respuesta)
#      exportar_examen_pdf(examen)
#

# =============================================================================
# SERVIDOR HTTP (stdlib, sin dependencias externas) — CAPA WEB UNIFICADA
# =============================================================================
#
# Expone el motor de arriba (generate_exercise, generate_exam, validate_answer,
# export_exam_pdf_bytes, get_options) como una API HTTP consumida por el
# frontend (index.html). Usa http.server + ThreadingHTTPServer, igual que el
# resto de los backends de la app (modulo0), para no depender de Flask ni de
# nada que haya que instalar aparte de las librerias que el motor ya usa
# (sympy, reportlab, matplotlib).
#
# Contrato de respuesta (coincide con la funcion api() de index.html):
#   Exito:  {"success": true,  "data": {...}}
#   Error:  {"success": false, "error": "mensaje"}
#
# Rutas:
#   GET  /            /health          /api/health   -> estado del servidor
#   GET  /api/options                                -> get_options()
#   POST /api/exercise  {topic, subtype?, difficulty?, direction?, seed?,
#                         seen_fingerprints?}          -> {exercise}
#   POST /api/exam      {topic, quantity, difficulty?, seed?, subtypes?, title?}
#                                                       -> {exam}
#   POST /api/check     {exercise, answer}             -> {result}
#   POST /api/exam/pdf  {exam, include_answer_key?}     -> {download_url, filename}
#   GET  /api/exam/pdf/<token>                          -> el PDF en si (descarga de un solo uso)
#
# El PDF se genera en memoria y se guarda en un almacen temporal (60s, token de
# un solo uso) en vez de mandarlo en base64 dentro del JSON. Esto es lo que
# permite que funcione la descarga nativa en iOS Safari / PWA / WebViews,
# donde URL.createObjectURL + <a download> falla en silencio (patron tomado
# de "prueba_de_aplicacion_con_modulo1.py").
# =============================================================================


class BackendError(Exception):
    """Error controlado con codigo HTTP asociado."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


MAX_REQUEST_BYTES = 2 * 1024 * 1024  # 2 MB — un examen + respuestas nunca pesa esto


# -- Almacen de PDFs en memoria (descarga por token de un solo uso) -------------
# iOS Safari, WebViews y PWAs en modo standalone no soportan bien
# URL.createObjectURL + <a download>. La solucion probada (tomada de
# "prueba_de_aplicacion_con_modulo1.py") es: el POST /api/exam/pdf genera el
# PDF en memoria y devuelve una URL real (GET) para descargarlo; el frontend
# decide si usa window.open (iOS/Safari) o fetch+blob (el resto).
_PDF_STORE: dict[str, tuple[bytes, str, float]] = {}
_PDF_STORE_LOCK = threading.Lock()
PDF_TOKEN_TTL = 60  # segundos hasta que el token expira si nadie lo descarga


def _pdf_store_cleanup() -> None:
    while True:
        time.sleep(30)
        now = time.time()
        with _PDF_STORE_LOCK:
            expired = [tok for tok, (_, _, ts) in _PDF_STORE.items() if now - ts > PDF_TOKEN_TTL]
            for tok in expired:
                del _PDF_STORE[tok]


threading.Thread(target=_pdf_store_cleanup, daemon=True).start()


def _boolp(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "si", "sí", "yes", "on")


def _intp(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def api_get_options(_data: dict[str, Any]) -> dict[str, Any]:
    return _get_options_cached()


@lru_cache(maxsize=1)
def _get_options_cached() -> dict[str, Any]:
    return get_options()


def api_new_exercise(data: dict[str, Any]) -> dict[str, Any]:
    topic = data.get("topic") or "factorizacion"
    exercise = generate_exercise(
        topic,
        subtype=data.get("subtype") or None,
        seed=data.get("seed"),
        difficulty=data.get("difficulty"),
        direction=data.get("direction") or None,
        seen_fingerprints=data.get("seen_fingerprints") or None,
    )
    return {"exercise": exercise}


def api_new_exam(data: dict[str, Any]) -> dict[str, Any]:
    topic = data.get("topic") or "factorizacion"
    quantity = _intp(data.get("quantity", data.get("count", 10)), 10)
    if quantity < 1 or quantity > MAX_QUESTIONS:
        raise BackendError(f"La cantidad de preguntas debe estar entre 1 y {MAX_QUESTIONS}.", 400)
    exam = generate_exam(
        topic,
        quantity,
        seed=data.get("seed"),
        subtypes=data.get("subtypes") or None,
        difficulty=data.get("difficulty"),
        title=data.get("title") or None,
    )
    return {"exam": exam}


def api_check_answer(data: dict[str, Any]) -> dict[str, Any]:
    exercise = data.get("exercise")
    if not isinstance(exercise, dict):
        raise BackendError("Falta el ejercicio a validar.", 400)
    answer = data.get("answer", "")
    result = validate_answer(exercise, str(answer))
    return {"result": result}


def api_exam_pdf(data: dict[str, Any]) -> dict[str, Any]:
    exam = data.get("exam")
    if not isinstance(exam, dict):
        raise BackendError("Falta el examen a exportar.", 400)
    include_answer_key = _boolp(data.get("include_answer_key"), True)
    pdf_bytes = export_exam_pdf_bytes(exam, include_answer_key=include_answer_key)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", str(exam.get("title", "Examen"))).strip("_") or "Examen"
    filename = f"{safe_title}.pdf"
    token = uuid.uuid4().hex
    with _PDF_STORE_LOCK:
        _PDF_STORE[token] = (pdf_bytes, filename, time.time())
    return {"download_url": f"/api/exam/pdf/{token}", "filename": filename}


ROUTES_GET: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/api/options": api_get_options,
}

ROUTES_POST: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/api/exercise": api_new_exercise,
    "/api/exam": api_new_exam,
    "/api/check": api_check_answer,
    "/api/exam/pdf": api_exam_pdf,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "Modulo1Backend/1.0"

    # -- utilidades de respuesta -------------------------------------------------
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ok(self, data: Any) -> None:
        self._send_json({"success": True, "data": data})

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"success": False, "error": message}, status)

    def _send_pdf(self, pdf_bytes: bytes, filename: str) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(pdf_bytes)))
        self.end_headers()
        self.wfile.write(pdf_bytes)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BYTES:
            raise BackendError("Cuerpo de la peticion demasiado grande.", 413)
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    # -- rutas HTTP ---------------------------------------------------------------
    def log_message(self, *_args: Any) -> None:  # silencia el log por defecto
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = re.sub(r"/{2,}", "/", urllib.parse.urlparse(self.path).path).rstrip("/") or "/"
        try:
            if path in ("/", "/health", "/api/health"):
                self._send_ok({"ok": True, "topics": list(TOPICS), "difficulties": list(DIFFICULTY_LEVELS)})
                return
            if path.startswith("/api/exam/pdf/"):
                token = path[len("/api/exam/pdf/"):]
                with _PDF_STORE_LOCK:
                    entry = _PDF_STORE.pop(token, None)  # descarga de un solo uso
                if entry is None:
                    raise BackendError("Token de PDF no encontrado o ya utilizado.", 404)
                pdf_bytes, filename, _created_at = entry
                self._send_pdf(pdf_bytes, filename)
                return
            handler = ROUTES_GET.get(path)
            if handler is None:
                raise BackendError("Ruta no encontrada", 404)
            self._send_ok(handler({}))
        except BackendError as exc:
            self._send_error(str(exc), exc.status)
        except Exception as exc:
            self._send_error(str(exc), 500)

    def do_POST(self) -> None:
        path = re.sub(r"/{2,}", "/", urllib.parse.urlparse(self.path).path).rstrip("/") or "/"
        try:
            handler = ROUTES_POST.get(path)
            if handler is None:
                raise BackendError("Ruta no encontrada", 404)
            data = self._read_json_body()
            self._send_ok(handler(data))
        except BackendError as exc:
            self._send_error(str(exc), exc.status)
        except (ValueError, RuntimeError) as exc:
            # Errores esperables de validacion / parseo de expresiones del usuario.
            self._send_error(str(exc), 400)
        except Exception as exc:
            self._send_error(str(exc), 500)


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) != 0


def _pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 25):
        if _port_free(host, port):
            return port
    raise RuntimeError("No encontre un puerto disponible.")


def run_server(host: str = "0.0.0.0", port: int = 8000, auto_port: bool = False) -> None:
    if auto_port:
        port = _pick_port(host, port)
    server = ThreadingHTTPServer((host, port), Handler)
    print("Modulo 1 - backend unificado")
    print(f"Escuchando en http://{host}:{port}")
    print("Ctrl+C para detener.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backend HTTP unificado para Modulo 1")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    parser.add_argument("--auto-port", action="store_true", help="Si el puerto esta ocupado, busca el siguiente libre (util en local).")
    parser.add_argument("--smoke-test", action="store_true", help="Corre smoke_test() y sale, sin levantar el servidor.")
    args = parser.parse_args(argv)
    if args.smoke_test:
        print(json.dumps(smoke_test(), ensure_ascii=False, indent=2, default=str))
        return
    run_server(args.host, args.port, auto_port=args.auto_port)


if __name__ == "__main__":
    main()
