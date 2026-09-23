"""
Este backend usa Flask (no FastAPI), así que se activa así, no con
uvicorn — uvicorn es para apps ASGI y Flask es WSGI:

    python Calc_fac_back.py

Eso ya deja el servidor corriendo en el puerto 5000 (ver el bloque
if __name__ == "__main__" al final del archivo).
"""

import os
from dotenv import load_dotenv
load_dotenv()

import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application, convert_xor)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Configuración de CORS segura: Solo permite peticiones desde tu GitHub Pages y entornos de prueba locales.
# Si tu repositorio se llama de otra manera, ajusta la URL en la lista de origins.
CORS(app, origins=["https://ignaciopaezr-stack.github.io", "http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:5000"])

# Se elimina la variable API_KEY_SECRETA ya que la seguridad ahora la maneja CORS

def factorizar_api(polinomio):
    if not polinomio or polinomio.strip() == "":
        return {"status":"error", "data":"El polinomio no puede estar vacío."}
    
    try:
        mis_transformaciones = standard_transformations + (implicit_multiplication_application, convert_xor)
        expresion_limpia = parse_expr(polinomio, transformations=mis_transformaciones)
        resultado = sp.factor(expresion_limpia, gaussian=True)
        return {"status":"success", "data": str(resultado)}
        
    except SyntaxError:
        return {"status":"error", "data":"La expresión no es correcta. Revise los paréntesis y operadores."}
    except ValueError:
        return {"status":"error", "data":"La expresión no es correcta."}
    except ZeroDivisionError:
        return {"status":"error", "data":"Error matemático: División por cero detectada."}
    except Exception as e:
        return {"status":"error", "data":"Error matemático: " + str(e)}


@app.route('/factorizar', methods=['POST'])
def api_endpoints():
    # Se elimina la validación que solicitaba el "x-api-key"
    
    datos = request.get_json()
    polinomio = datos.get("polinomio", "") if datos else ""

    respuesta = factorizar_api(polinomio)

    return jsonify(respuesta)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
