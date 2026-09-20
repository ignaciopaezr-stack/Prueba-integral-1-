# Para iniciar el servidor:
# 1. python Graficadora_Back.py (usado por el menú principal)
# 2. uvicorn Graficadora_Back:app --reload (para desarrollo local)

import numpy as np
import matplotlib
import sympy as sp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- NUEVO
from pydantic import BaseModel
from typing import List

matplotlib.use('Agg')

from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application, 
    convert_xor
)
from sympy import Symbol, symbols, lambdify
from skimage import measure

app = FastAPI()

# --- NUEVO: PERMITE QUE EL HTML SE CONECTe AL BACKEND ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen (ideal para desarrollo local)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _limpiar_no_finitos(valor):
    """NaN e Infinity son floats válidos en Python, pero NO son JSON válido:
    json.dumps() los escribe tal cual como los literales NaN/Infinity (sin
    comillas), y JSON.parse() en el navegador los rechaza con un error de
    sintaxis. Eso hacía fallar el fetch del frontend con un mensaje genérico
    de "formato inesperado" cada vez que la función no estaba definida en
    algún punto del rango (sqrt de negativos, log de x<=0, asíntotas de
    tan/1/x, etc.). Se reemplazan por None (-> null en JSON), que Plotly ya
    entiende como un hueco en la curva/superficie en vez de romper todo."""
    if isinstance(valor, list):
        return [_limpiar_no_finitos(v) for v in valor]
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


@app.get("/puntos_2d")
def graficar2d_api(polinomio: str, dominio_min: float = -10, dominio_max: float = 10, puntos: int = 500):
    if not polinomio or polinomio.strip() == "":
        return {"status": "error", "data": "El polinomio no puede estar vacío."}
    try:
        # Límites de seguridad: si el rango pedido es inválido o absurdamente
        # grande, se recorta a algo razonable en vez de fallar o colgar el servidor.
        d_min, d_max = float(dominio_min), float(dominio_max)
        if d_min >= d_max:
            d_min, d_max = -10.0, 10.0
        ancho = d_max - d_min
        if ancho > 100000:
            centro = (d_min + d_max) / 2
            d_min, d_max = centro - 50000, centro + 50000

        n_puntos = max(100, min(int(puntos), 2000))
        mis_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        x = Symbol('x')

        if "=" in polinomio:
            # Ecuación en x,y (ej. "y=x", "2*y=4*x+2") en vez de una
            # expresión ya despejada — se despeja y con sympy y se grafica
            # eso. Antes esto ni se intentaba: cualquier "=" se mandaba
            # directo a /superficie_3d como si fuera una superficie 3D
            # (aunque no tuviera z), así que "y=x" terminaba como un plano
            # vertical en vez de la recta 2D que se espera.
            y = Symbol('y')
            lado_izq_str, lado_der_str = polinomio.split("=", 1)
            lado_izq = parse_expr(lado_izq_str, transformations=mis_transformations)
            lado_der = parse_expr(lado_der_str, transformations=mis_transformations)
            ecuacion = lado_izq - lado_der
            simbolos_extra = ecuacion.free_symbols - {x, y}
            if simbolos_extra:
                nombres = ", ".join(sorted(str(s) for s in simbolos_extra))
                return {"status": "error", "data": f"La ecuación usa una variable que no es x o y: {nombres}."}
            if y not in ecuacion.free_symbols:
                return {"status": "error", "data": "Esa ecuación no depende de y — sería una recta vertical, y una recta vertical no es una función y=f(x) (no se puede graficar en 2D acá)."}
            soluciones = sp.solve(sp.Eq(ecuacion, 0), y)
            if not soluciones:
                return {"status": "error", "data": "No se pudo despejar y en esa ecuación."}
            # Si hay más de una solución (ej. y²=x da y=±√x), se grafica
            # solo la primera rama — graficar varias a la vez es una
            # función distinta a esta.
            expresion_limpia = soluciones[0]
        else:
            expresion_limpia = parse_expr(polinomio, transformations=mis_transformations)
            simbolos_extra = expresion_limpia.free_symbols - {x}
            if simbolos_extra:
                nombres = ", ".join(sorted(str(s) for s in simbolos_extra))
                return {"status": "error", "data": f"La expresión usa una variable que no es x: {nombres}."}

        funcion_rapida = lambdify(x, expresion_limpia, "numpy")
        puntos_x = np.linspace(d_min, d_max, n_puntos)
        puntos_y = np.asarray(funcion_rapida(puntos_x), dtype=float)
        # Si la expresión es constante (ej. "5"), lambdify no la evalúa
        # punto por punto y devuelve un solo número en vez de un array de
        # 500 — sin esto, puntos_y.tolist() truena porque un float de
        # Python no tiene .tolist(). Se expande para que quede del mismo
        # tamaño que puntos_x.
        if puntos_y.shape != puntos_x.shape:
            puntos_y = np.broadcast_to(puntos_y, puntos_x.shape).copy()
        return {"status": "success", "x": puntos_x.tolist(), "y": _limpiar_no_finitos(puntos_y.tolist())}
    except Exception as e:
        return {"status": "error", "data": f"Error calculando puntos: {str(e)}"}

@app.get("/puntos_3d")
def graficar3d(polinomio: str, rango: float = 10, resolucion: int = 50):
    if not polinomio or polinomio.strip() == "":
        return {"status": "error", "data": "El polinomio no puede estar vacío."}
    try:
        r = max(0.5, min(float(rango), 50.0))
        n = max(20, min(int(resolucion), 90))
        mis_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        expresion_limpia = parse_expr(polinomio, transformations=mis_transformations)
        x, y = sp.symbols('x y')
        simbolos_extra = expresion_limpia.free_symbols - {x, y}
        if simbolos_extra:
            nombres = ", ".join(sorted(str(s) for s in simbolos_extra))
            return {"status": "error", "data": f"La expresión usa una variable que no es x o y: {nombres}."}
        funcion_rapida = sp.lambdify((x, y), expresion_limpia, "numpy")
        eje_x = np.linspace(-r, r, n)
        eje_y = np.linspace(-r, r, n)
        malla_x, malla_y = np.meshgrid(eje_x, eje_y)
        malla_z = np.asarray(funcion_rapida(malla_x, malla_y), dtype=float)
        # Mismo caso que en /puntos_2d: una expresión constante (o que no
        # depende ni de x ni de y) devuelve un solo número en vez de una
        # grilla 50x50 — se expande para que coincida con malla_x/malla_y.
        if malla_z.shape != malla_x.shape:
            malla_z = np.broadcast_to(malla_z, malla_x.shape).copy()
        return {"status": "success", "x": eje_x.tolist(), "y": eje_y.tolist(), "z": _limpiar_no_finitos(malla_z.tolist())}
    except Exception as e:
        return {"status": "error", "data": f"Error 3D: {str(e)}"}


@app.get("/superficie_3d")
def superficie3d_api(ecuacion: str, rango: float = 8.0, resolucion: int = 60):
    """
    Grafica CUALQUIER superficie 3D (implícita), no solo z = f(x,y):
    esferas, elipsoides, toros, conos, planos, lo que sea que involucre x, y, z.

    Acepta una ecuación con "=" (x**2+y**2+z**2=25) o una expresión suelta
    tipo "x**2+y**2+z**2". Si no hay "=", primero se prueba igualada a 0;
    si eso no da una superficie real (p. ej. una suma de cuadrados, que en
    0 es solo el origen, un punto — no una superficie), se reintenta
    igualada a 1 automáticamente y se avisa en la respuesta. Una ecuación
    con "=" explícito SIEMPRE se respeta tal cual, sin este ajuste.

    Método: se evalúa F(x,y,z) = lado_izq - lado_der en una grilla 3D dentro
    de [-rango, rango]^3, y se extrae la superficie donde F=nivel con el
    algoritmo "marching cubes" (el método estándar para esto — el mismo
    tipo de técnica que usan Desmos 3D, GeoGebra 3D, etc.). Así el eje z
    (y x, y) reflejan la geometría real de la superficie, no un rango fijo.

    No reemplaza ni modifica /puntos_2d ni /puntos_3d — es un endpoint
    nuevo, aparte, para cuando la expresión usa z o trae "=".
    """
    if not ecuacion or ecuacion.strip() == "":
        return {"status": "error", "data": "La ecuación no puede estar vacía."}
    try:
        mis_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        es_ecuacion_explicita = "=" in ecuacion
        if es_ecuacion_explicita:
            lado_izq_str, lado_der_str = ecuacion.split("=", 1)
            lado_izq = parse_expr(lado_izq_str, transformations=mis_transformations)
            lado_der = parse_expr(lado_der_str, transformations=mis_transformations)
            F = lado_izq - lado_der
        else:
            F = parse_expr(ecuacion, transformations=mis_transformations)

        x, y, z = sp.symbols('x y z')
        simbolos_extra = F.free_symbols - {x, y, z}
        if simbolos_extra:
            nombres = ", ".join(sorted(str(s) for s in simbolos_extra))
            return {"status": "error", "data": f"La ecuación usa una variable que no es x, y o z: {nombres}."}
        funcion_rapida = sp.lambdify((x, y, z), F, "numpy")

        # Límites de seguridad para que nadie tire el servidor pidiendo
        # una grilla gigante o un rango absurdo.
        r = max(0.5, min(float(rango), 50.0))
        n = max(20, min(int(resolucion), 90))

        eje = np.linspace(-r, r, n)
        malla_x, malla_y, malla_z = np.meshgrid(eje, eje, eje, indexing="ij")
        volumen = np.asarray(funcion_rapida(malla_x, malla_y, malla_z), dtype=float)

        # Si F no depende de alguna variable (ej. el plano "x=2"), lambdify
        # puede devolver un solo número en vez de un array del tamaño de
        # la grilla — hay que expandirlo para que marching cubes lo acepte.
        if volumen.shape != malla_x.shape:
            volumen = np.broadcast_to(volumen, malla_x.shape).copy()

        if not np.isfinite(volumen).any():
            return {"status": "error", "data": "La superficie no da valores numéricos válidos en este rango."}

        # Si la ecuación tiene una singularidad puntual dentro de la grilla
        # (asíntota, división por cero, etc.), ese punto queda en NaN/±inf
        # y marching_cubes puede fallar o devolver basura — se reemplaza por
        # un centinela bien grande y de signo correcto (claramente "afuera"
        # de cualquier nivel real que se use acá) para que esos puntos
        # sueltos no se confundan con parte de la superficie.
        volumen = np.nan_to_num(volumen, nan=1e10, posinf=1e10, neginf=-1e10)

        nivel = 0.0
        nota = None
        cruza = volumen.min() < nivel < volumen.max()
        if not cruza and nivel == 0.0 and volumen.min() < 1.0 < volumen.max():
            # Apuntar a nivel 0 (ya sea porque no escribiste "=", o porque
            # escribiste "...=0" explícito — son la misma ecuación) puede ser
            # un caso degenerado: muchas sumas de cuadrados (como
            # x**2+y**2+z**2) en 0 son solo el origen, un punto, no una
            # superficie. Se reintenta en 1 en vez de devolver un error.
            nivel = 1.0
            nota = 'Esa ecuación igualada a 0 da solo un punto (no una superficie), así que se graficó igualada a 1. Para elegir el radio/nivel vos mismo, escribí algo como "...=25".'
            cruza = True
        if not cruza:
            return {"status": "error", "data": "La superficie no pasa por el rango de gráfica actual — probá agrandar el rango en Ajustes."}

        paso = eje[1] - eje[0]
        vertices, caras, _, _ = measure.marching_cubes(volumen, level=nivel, spacing=(paso, paso, paso))
        vertices = vertices - r  # recentrar: marching_cubes devuelve coordenadas desde 0

        respuesta = {
            "status": "success",
            "x": vertices[:, 0].tolist(),
            "y": vertices[:, 1].tolist(),
            "z": vertices[:, 2].tolist(),
            "i": caras[:, 0].tolist(),
            "j": caras[:, 1].tolist(),
            "k": caras[:, 2].tolist(),
        }
        if nota:
            respuesta["nota"] = nota
        return respuesta
    except Exception as e:
        return {"status": "error", "data": f"Error en superficie 3D: {str(e)}"}


class PuntosEvaluarRequest(BaseModel):
    ecuaciones: List[str]
    x: List[float]
    y: List[float]
    z: List[float]


@app.post("/evaluar_en_puntos")
def evaluar_en_puntos(payload: PuntosEvaluarRequest):
    """
    Evalúa una o más ecuaciones implícitas G(x,y,z) en una lista de puntos
    ya dados por el cliente (no en una grilla propia, a diferencia de
    /superficie_3d). Se usa para "iluminar" dónde una superficie pasa cerca
    de otra: se le mandan los vértices/grilla de la superficie A y la(s)
    ecuación(es) de la(s) otra(s) superficie(s) B — un valor devuelto
    cercano a 0 en un punto de A significa que ese punto está cerca de B
    (probable intersección).

    Es POST y no GET como los demás endpoints porque acá los puntos pueden
    ser miles — no entran cómodos en query params.

    Acepta el mismo formato flexible por ecuación que /superficie_3d: con
    "=" explícito o una expresión suelta (igualada a 0 implícitamente).
    Cada ecuación de la lista se evalúa por separado: si una falla (variable
    no reconocida, sintaxis inválida), esa entrada devuelve null en la
    respuesta en vez de tirar abajo las demás.
    """
    n = len(payload.x)
    if not (n == len(payload.y) == len(payload.z)):
        return {"status": "error", "data": "x, y, z deben tener la misma longitud."}
    if n == 0:
        return {"status": "error", "data": "No se mandaron puntos para evaluar."}
    if n > 300000:
        return {"status": "error", "data": "Demasiados puntos para evaluar de una vez."}
    if not payload.ecuaciones:
        return {"status": "error", "data": "No se mandó ninguna ecuación."}

    mis_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    x, y, z = sp.symbols('x y z')
    arr_x = np.asarray(payload.x, dtype=float)
    arr_y = np.asarray(payload.y, dtype=float)
    arr_z = np.asarray(payload.z, dtype=float)

    resultados = []
    for ecuacion in payload.ecuaciones:
        try:
            if not ecuacion or ecuacion.strip() == "":
                resultados.append(None)
                continue
            if "=" in ecuacion:
                lado_izq_str, lado_der_str = ecuacion.split("=", 1)
                lado_izq = parse_expr(lado_izq_str, transformations=mis_transformations)
                lado_der = parse_expr(lado_der_str, transformations=mis_transformations)
                F = lado_izq - lado_der
            else:
                F = parse_expr(ecuacion, transformations=mis_transformations)

            if F.free_symbols - {x, y, z}:
                resultados.append(None)
                continue

            funcion_rapida = sp.lambdify((x, y, z), F, "numpy")
            valores = np.asarray(funcion_rapida(arr_x, arr_y, arr_z), dtype=float)
            if valores.shape != arr_x.shape:
                valores = np.broadcast_to(valores, arr_x.shape).copy()
            resultados.append(_limpiar_no_finitos(valores.tolist()))
        except Exception:
            resultados.append(None)

    return {"status": "success", "valores": resultados}

class InterseccionRequest(BaseModel):
    ecuaciones: List[str]
    rango: float = 8.0
    resolucion: int = 60


@app.post("/interseccion_3d")
def interseccion_3d(payload: InterseccionRequest):
    """
    Calcula la superficie límite del VOLUMEN que dos o más superficies
    implícitas cerradas tienen en común (la intersección booleana de sus
    "adentros"), con la misma técnica que /superficie_3d (marching cubes)
    pero sobre F_combinada = max(F1_normalizada, F2_normalizada, ...): el
    máximo de varias funciones implícitas da exactamente la región donde
    TODAS son negativas a la vez — la construcción estándar para
    intersección booleana con superficies implícitas (así como min() daría
    la unión).

    Cada ecuación puede estar escrita con cualquier convención de signo
    (ej. "x**2+y**2+z**2=25" y "25-x**2-y**2-z**2=0" son la misma esfera,
    con signos opuestos) — para normalizarlas todas a "negativo = adentro"
    antes de combinarlas, se mira qué fracción de la grilla muestreada da
    negativo para cada una: una figura cerrada normal ocupa una fracción
    chica del volumen total, así que la mayoría de los puntos deberían caer
    afuera (positivo); si es al revés (mayoría negativo), se invierte esa
    ecuación en particular. No depende de dónde esté centrada la figura —
    no es infalible para formas que ocupan una fracción grande de la grilla
    muestreada, o formas muy irregulares (ej. un toro delgado, donde
    "adentro" es una fracción chica del volumen igual que "afuera").

    Requiere al menos 2 ecuaciones utilizables. Si alguna no es válida
    (variable rara, sintaxis inválida), se descarta esa sola en vez de
    fallar todo — salvo que queden menos de 2.
    """
    if not payload.ecuaciones or len(payload.ecuaciones) < 2:
        return {"status": "error", "data": "Hacen falta al menos 2 ecuaciones para calcular una intersección."}
    try:
        mis_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        x, y, z = sp.symbols('x y z')

        r = max(0.5, min(float(payload.rango), 50.0))
        n = max(20, min(int(payload.resolucion), 90))
        eje = np.linspace(-r, r, n)
        malla_x, malla_y, malla_z = np.meshgrid(eje, eje, eje, indexing="ij")

        volumenes = []
        for ecuacion in payload.ecuaciones:
            try:
                if not ecuacion or ecuacion.strip() == "":
                    continue
                if "=" in ecuacion:
                    lado_izq_str, lado_der_str = ecuacion.split("=", 1)
                    lado_izq = parse_expr(lado_izq_str, transformations=mis_transformations)
                    lado_der = parse_expr(lado_der_str, transformations=mis_transformations)
                    F = lado_izq - lado_der
                else:
                    F = parse_expr(ecuacion, transformations=mis_transformations)
                if F.free_symbols - {x, y, z}:
                    continue

                funcion_rapida = sp.lambdify((x, y, z), F, "numpy")

                vol = np.asarray(funcion_rapida(malla_x, malla_y, malla_z), dtype=float)
                if vol.shape != malla_x.shape:
                    vol = np.broadcast_to(vol, malla_x.shape).copy()
                vol = np.nan_to_num(vol, nan=1e10, posinf=1e10, neginf=-1e10)

                # Normalizar signo: una figura cerrada normal (esfera,
                # elipsoide, etc.) ocupa una fracción chica del volumen
                # muestreado — así que la mayoría de los puntos de la
                # grilla caen AFUERA de ella, sea donde sea que esté
                # centrada. Si más de la mitad de la grilla da negativo acá,
                # es que la convención de esta ecuación en particular tiene
                # "negativo" del lado de afuera (al revés de lo que se
                # necesita para combinar), así que se invierte. Evaluar en
                # el origen en vez de esto fallaba para cualquier figura que
                # no estuviera centrada ahí — con dos figuras separadas del
                # origen, ambas terminaban "adentro" por error.
                fraccion_negativa = float(np.mean(vol < 0))
                if fraccion_negativa > 0.5:
                    vol = -vol

                volumenes.append(vol)
            except Exception:
                continue

        if len(volumenes) < 2:
            return {"status": "error", "data": "No se pudo interpretar al menos 2 de las ecuaciones para calcular la intersección."}

        combinado = volumenes[0]
        for v in volumenes[1:]:
            combinado = np.maximum(combinado, v)

        nivel = 0.0
        if not (combinado.min() < nivel < combinado.max()):
            # No se cruzan dentro del rango actual (o una contiene a la
            # otra por completo) — no es un error, simplemente no hay nada
            # que dibujar todavía.
            return {"status": "success", "vacio": True, "x": [], "y": [], "z": [], "i": [], "j": [], "k": []}

        paso = eje[1] - eje[0]
        vertices, caras, _, _ = measure.marching_cubes(combinado, level=nivel, spacing=(paso, paso, paso))
        vertices = vertices - r

        return {
            "status": "success", "vacio": False,
            "x": vertices[:, 0].tolist(), "y": vertices[:, 1].tolist(), "z": vertices[:, 2].tolist(),
            "i": caras[:, 0].tolist(), "j": caras[:, 1].tolist(), "k": caras[:, 2].tolist(),
        }
    except Exception as e:
        return {"status": "error", "data": f"Error calculando la intersección: {str(e)}"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)