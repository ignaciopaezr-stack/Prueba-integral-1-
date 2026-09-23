
"""
Este backend usa Flask (no FastAPI), así que se activa así, no con
uvicorn — uvicorn es para apps ASGI y Flask es WSGI:

    python Calc_fac_back.py

Eso ya deja el servidor corriendo en el puerto 5000 (ver el bloque
if __name__ == "__main__" al final del archivo).
"""

#Bibliotecas
import os
from dotenv import load_dotenv
load_dotenv()

import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations,implicit_multiplication_application,convert_xor)
#Estas son herramientas de traducción como para hacer que 2x sea 2*x, o que 2^3 sea 2**3, etc.

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  #Esto es para permitir que el front-end pueda hacer peticiones a esta API desde otro dominio diferente al front-end.

# Definimos una API Key estática para proteger nuestro backend
API_KEY_SECRETA = os.getenv("API_SECRET_KEY", "teorema-api-secure-key-123")

###############################################################################################################################################


#Función para factorizar


def factorizar_api(polinomio):
    
    
    if not polinomio or polinomio.strip() == "":
        return {"status":"error", "data":"El polinomio no puede estar vacío."}
    
    try:
    
        #Establecemos las reglas de traducción
        
        mis_transformaciones= standard_transformations + (implicit_multiplication_application, convert_xor)
        
        expresion_limpia=parse_expr(polinomio,transformations=mis_transformaciones)
        
        resultado=sp.factor(expresion_limpia, gaussian=True)
        
        return {"status":"success", "data":str(resultado)}

        
        
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
    client_api_key = request.headers.get("x-api-key")

    if not client_api_key or client_api_key != API_KEY_SECRETA:
        return (
            jsonify({
                "status": "error",
                "data": "Acceso denegado: API Key inválida o faltante.",
            }),
            401,
        )

    # 2. Procesamiento de datos de entrada
    datos = request.get_json()
    polinomio = datos.get("polinomio", "") if datos else ""

    # 3. Llamar a tu lógica de Python
    respuesta = factorizar_api(polinomio)

    return jsonify(respuesta)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
        
##########################################################################################################################################################
