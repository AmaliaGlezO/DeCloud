import math
import random
import heapq

# ---------------------------------------------------------
# 1. GENERADORES DE VARIABLES ALEATORIAS (TRANSFORMADA INVERSA)
# ---------------------------------------------------------
def uniform_rv(a, b):
    """Genera U(a, b) usando la transformada inversa sobre U(0,1)."""
    u = random.random()
    return a + (b - a) * u

def exp_rv(mean_time):
    """Genera Exp(lambda) con media 'mean_time' mediante transformada inversa."""
    u = random.random()
    while u == 0:  # Evitar log(0)
        u = random.random()
    return -mean_time * math.log(u)

# ---------------------------------------------------------
# 2. CLASES ESTRUCTURALES
# ---------------------------------------------------------
class Evento:
    def __init__(self, tiempo, tipo, cliente=None):
        self.tiempo = tiempo
        self.tipo = tipo  # 'LLEGADA' o 'FIN_PREPARACION'
        self.cliente = cliente

    def __lt__(self, other):
        return self.tiempo < other.tiempo

class Cliente:
    def __init__(self, id_cliente, tiempo_llegada, tipo_producto):
        self.id = id_cliente
        self.tiempo_llegada = tiempo_llegada
        self.tipo_producto = tipo_producto  # 'sandwich' o 'sushi'

# ---------------------------------------------------------
# 3. NÚCLEO DE LA SIMULACIÓN
# ---------------------------------------------------------
def simular_jornada(usar_empleado_extra=False, media_arribo=3.0):
    TIEMPO_TOTAL = 660.0  # 10:00 AM a 9:00 PM (minutos)
    PEAK_1 = (90.0, 210.0)   # 11:30 AM a 1:30 PM
    PEAK_2 = (420.0, 540.0)  # 5:00 PM a 7:00 PM

    def es_hora_pico(t):
        return (PEAK_1[0] <= t <= PEAK_1[1]) or (PEAK_2[0] <= t <= PEAK_2[1])

    reloj = 0.0
    fel = []  # Lista de Eventos Futuros (Priority Queue)
    cola = []
    
    empleados_ocupados = 0
    total_clientes = 0
    clientes_quejas = 0

    # Programar primer arribo
    id_counter = 1
    t_primer_arribo = exp_rv(media_arribo)
    if t_primer_arribo <= TIEMPO_TOTAL:
        prod = 'sandwich' if uniform_rv(0, 1) < 0.5 else 'sushi'
        heapq.heappush(fel, Evento(t_primer_arribo, 'LLEGADA', Cliente(id_counter, t_primer_arribo, prod)))

    while fel:
        ev = heapq.heappop(fel)
        reloj = ev.tiempo

        # Capacidad actual del personal
        max_empleados = 3 if (usar_empleado_extra and es_hora_pico(reloj)) else 2

        if ev.tipo == 'LLEGADA':
            if reloj <= TIEMPO_TOTAL:
                total_clientes += 1
                
                # Programar siguiente arribo
                id_counter += 1
                sig_tiempo = reloj + exp_rv(media_arribo)
                if sig_tiempo <= TIEMPO_TOTAL:
                    p_next = 'sandwich' if uniform_rv(0, 1) < 0.5 else 'sushi'
                    heapq.heappush(fel, Evento(sig_tiempo, 'LLEGADA', Cliente(id_counter, sig_tiempo, p_next)))

                # Atención directa o encolado
                if empleados_ocupados < max_empleados:
                    empleados_ocupados += 1
                    t_prep = uniform_rv(3, 5) if ev.cliente.tipo_producto == 'sandwich' else uniform_rv(5, 8)
                    heapq.heappush(fel, Evento(reloj + t_prep, 'FIN_PREPARACION', ev.cliente))
                else:
                    cola.append(ev.cliente)

        elif ev.tipo == 'FIN_PREPARACION':
            empleados_ocupados -= 1

            if cola and empleados_ocupados < max_empleados:
                cliente_siguiente = cola.pop(0)
                empleados_ocupados += 1
                
                tiempo_espera = reloj - cliente_siguiente.tiempo_llegada
                if tiempo_espera > 5.0:
                    clientes_quejas += 1

                t_prep = uniform_rv(3, 5) if cliente_siguiente.tipo_producto == 'sandwich' else uniform_rv(5, 8)
                heapq.heappush(fel, Evento(reloj + t_prep, 'FIN_PREPARACION', cliente_siguiente))

    porcentaje_quejas = (clientes_quejas / total_clientes * 100) if total_clientes > 0 else 0
    return porcentaje_quejas

# ---------------------------------------------------------
# 4. EXPERIMENTACIÓN (CORRIDAS MONTE CARLO)
# ---------------------------------------------------------
def ejecutar_experimentos(num_replicas=500):
    print(f"Ejecutando {num_replicas} simulaciones...")
    
    quejas_2_emp = [simular_jornada(usar_empleado_extra=False) for _ in range(num_replicas)]
    quejas_3_emp = [simular_jornada(usar_empleado_extra=True) for _ in range(num_replicas)]

    prom_2 = sum(quejas_2_emp) / num_replicas
    prom_3 = sum(quejas_3_emp) / num_replicas

    print("\n--- RESULTADOS DE LA SIMULACIÓN ---")
    print(f"Porcentaje promedio de quejas (2 Empleados): {prom_2:.2f}%")
    print(f"Porcentaje promedio de quejas (3 Empleados en Hora Pico): {prom_3:.2f}%")
    print(f"Reducción lograda: {prom_2 - prom_3:.2f}%")

if __name__ == "__main__":
    ejecutar_experimentos()