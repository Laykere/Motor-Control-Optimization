# motor_simulator.py
import numpy as np
from scipy.integrate import solve_ivp

class MotorBLDC:
    """Representa a dinâmica matemática de um Motor BLDC Trifásico."""
    
    def __init__(self, R, L, M, J, B, Ke, P):
        # Parâmetros Físicos
        self.R = R       # Resistência de fase
        self.L = L       # Auto-indutância de fase
        self.M = M       # Indutância mútua
        self.Leff = L - M # Indutância efetiva para o sistema reduzido
        self.J = J       # Inércia
        self.B = B       # Fricção viscosa
        self.Ke = Ke     # Constante de Back-EMF (Volt-seg/rad)
        self.P = P       # Pares de polos
        
        # Estado: [ia, ib, omega, theta]
        self.estado = np.zeros(4)
        
        # Variáveis de Entrada (atualizadas pelo Inversor)
        self.Va, self.Vb, self.Vc = 0.0, 0.0, 0.0
        self.TL = 0.0 # Torque de Carga

    def aplicar_tensoes(self, Va, Vb, Vc):
        """Atualiza as tensões de fase aplicadas pelo Inversor."""
        self.Va, self.Vb, self.Vc = Va, Vb, Vc

    def _trapezoidal_func(self, theta_m):
        """
        Calcula a função f(theta) que define a forma de onda trapezoidal normalizada (f(theta)).
        """
        # 1. Converter Posição Mecânica para Posição Elétrica
        theta_e = self.P * theta_m

        # 2. Normalizar o ângulo entre 0 e 2*pi
        theta_norm = np.fmod(theta_e, 2 * np.pi)
        if theta_norm < 0: theta_norm += 2 * np.pi
        
        # Constante para a inclinação da rampa (de 0 a 1 em pi/6 radianos)
        # 1 / (pi/6) = 6/pi
        INCLINACAO = 6 / np.pi 

        # --- Lógica do Mapeamento Trapezoidal (6 Seções) ---

        if 0 <= theta_norm < np.pi/6: 
            # Rampa de subida 1 (0 a 1): 0 a 30 graus
            return INCLINACAO * theta_norm
        
        elif np.pi/6 <= theta_norm < 5*np.pi/6: 
            # Platô de 120 graus: 30 a 150 graus
            return 1.0
        
        elif 5*np.pi/6 <= theta_norm < 7*np.pi/6: 
            # Rampa de descida (1 a -1): 150 a 210 graus
            # Valor inicial (em 5*pi/6) é 1.0. A rampa tem amplitude 2.
            return 1.0 - INCLINACAO * (theta_norm - 5*np.pi/6)
        
        elif 7*np.pi/6 <= theta_norm < 11*np.pi/6: 
            # Platô de 120 graus: 210 a 330 graus
            return -1.0
        
        else: # 11*np.pi/6 <= theta_norm < 2*np.pi
            # Rampa de subida 2 (-1 a 0): 330 a 360 graus
            # Valor inicial (em 11*pi/6) é -1.0. 
            # A rampa deve subir para 0.
            return -1.0 + INCLINACAO * (theta_norm - 11*np.pi/6)

        # Nota: A primeira rampa precisa ir de 0 a 1, e não de 0 a 0.5.
        # O modelo que vai de 0 a 1.0 em 30 graus é o mais comum para BLDCs.

    def _sinusoidal_func(self, theta_e):
        """NOVO: Forma de onda senoidal normalizada."""
        return np.sin(theta_e)

    def set_bemf_shape(self, shape='trapezoidal'):
        """Define a forma da BEMF (trapezoidal ou sinusoidal)."""
        if shape in ['trapezoidal', 'sinusoidal']:
            self.bemf_shape = shape
            print(f"Forma da BEMF do motor definida para: {shape}")

    def calcular_back_emf(self, theta_m, omega_m):

        # Converte o offset elétrico (120°) para mecânico
        offset_120_mec = (2 * np.pi / 3) / self.P
        offset_240_mec = (4 * np.pi / 3) / self.P

        if self.bemf_shape == 'sinusoidal':
            # Para FOC, BEMF é senoidal
            theta_e_a = self.P * theta_m
            theta_e_b = self.P * (theta_m - offset_120_mec)
            theta_e_c = self.P * (theta_m - offset_240_mec)
            
            e_a = self.Ke * omega_m * self._sinusoidal_func(theta_e_a)
            e_b = self.Ke * omega_m * self._sinusoidal_func(theta_e_b)
            e_c = self.Ke * omega_m * self._sinusoidal_func(theta_e_c)
        else:
            # Para 6-Step, BEMF é trapezoidal (usando a lógica corrigida)
            e_a = self.Ke * omega_m * self._trapezoidal_func(theta_m)
            e_b = self.Ke * omega_m * self._trapezoidal_func(theta_m - offset_120_mec)
            e_c = self.Ke * omega_m * self._trapezoidal_func(theta_m - offset_240_mec)
            
        return e_a, e_b, e_c

    def derivadas(self, t, estado):
        """Função de derivadas: [dia/dt, dib/dt, domega/dt, dtheta/dt]"""
        ia, ib, omega, theta = estado
        
        R, J, B, Leff, TL = self.R, self.J, self.B, self.Leff, self.TL
        Va, Vb, Vc = self.Va, self.Vb, self.Vc
        
        # --- 1. Variáveis Dependentes e BEMF ---
        ic = -(ia + ib)
        ea, eb, ec = self.calcular_back_emf(theta, omega)

        # --- 2. CÁLCULO DO TORQUE (ÚNICA VEZ) ---
        # A fórmula P_conv / omega funciona para ambas as formas de onda
        P_conv = (ia * ea) + (ib * eb) + (ic * ec)
        
        if abs(omega) < 1e-3:
            # Se omega é zero (ou perto), usa a fórmula baseada na forma de onda (mais estável na partida)
            e_a_norm, e_b_norm, e_c_norm = self.calcular_back_emf(theta, 1.0) 
            
            Te = self.Ke * self.P * (ia * e_a_norm + ib * e_b_norm + ic * e_c_norm)
        else:
            Te = P_conv / omega 

        # --- 3. DERIVADAS MECÂNICAS (d(omega)/dt e d(theta)/dt) ---
        domega_dt = (1/J) * (Te - B * omega - TL)
        dtheta_dt = omega

        # --- 4. DERIVADAS ELÉTRICAS (d(ia)/dt e d(ib)/dt) ---
        # Numeradores (Tensões de Linha menos Quedas)
        Num_ia = (Va - Vc) - R * (ia - ic) - (ea - ec)
        Num_ib = (Vb - Vc) - R * (ib - ic) - (eb - ec)
        
        # Matriz Inversa
        det_L = Leff * Leff * 0.75
        
        dia_dt = (1/det_L) * (Leff * Num_ia - (Leff/2) * Num_ib)
        dib_dt = (1/det_L) * (-(Leff/2) * Num_ia + Leff * Num_ib)
        
        # --- 5. RETORNO FINAL ---
        return [dia_dt, dib_dt, domega_dt, dtheta_dt]

    def avancar_tempo(self, dt):
        """Integra o sistema por um passo de tempo dt usando RK45."""
        sol = solve_ivp(
            self.derivadas, 
            [0, dt], 
            self.estado, 
            method='RK45', 
            t_eval=[dt]
        )
        
        if sol.success:
            self.estado = sol.y[:, -1]
        else:
            print(f"Erro na integração: {sol.message}")
            return False
        return True