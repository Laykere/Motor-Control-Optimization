# controlador.py
import numpy as np

#---------------------------------------------------------------------------------------------------------------------
# 1. Constants & Macros
#---------------------------------------------------------------------------------------------------------------------
SQRT_2_3 = np.sqrt(2.0/3.0)
SQRT_3 = np.sqrt(3.0)
INV_SQRT_3 = 1.0 / SQRT_3

#---------------------------------------------------------------------------------------------------------------------
# 2. PID Controller Implementation
#---------------------------------------------------------------------------------------------------------------------
class ControladorPID:
    """
    @brief      Implementa o controlador PID (MODIFICADO para anti-windup externo).
    """

    def __init__(self, Kp, Ki, Kd, V_MIN, V_MAX): 
        self.Kp = Kp  
        self.Ki = Ki
        self.Kd = Kd
        
        self.V_MIN = V_MIN
        self.V_MAX = V_MAX
        
        self.integrador = 0.0
        self.erro_anterior = 0.0

    #-----------------------------------------------------------------------------------------------------------------
    # @brief    Resets the internal state of the PID (Integrator and Previous Error).
    #-----------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.integrador = 0.0
        self.erro_anterior = 0.0
        
    #-----------------------------------------------------------------------------------------------------------------
    # @brief    Calculates the raw PID command before saturation logic.
    #           Rate: Called every simulation step (dt).
    # @param    referencia: The target setpoint.
    # @param    valor_atual: The measured value.
    # @param    dt: Delta time.
    # @return   Tuple (Raw Command, Error)
    #-----------------------------------------------------------------------------------------------------------------
    def calcular_comando_bruto(self, referencia, valor_atual, dt):
        
        # Calculate error
        erro = referencia - valor_atual
        
        # 1. Proportional and Derivative Terms
        termo_p = self.Kp * erro
        derivada = (erro - self.erro_anterior) / dt if dt > 0 else 0
        termo_d = self.Kd * derivada
        
        # 2. Raw Command Calculation (P + I + D)
        # Note: Integrator is applied from the previous step's accumulation
        comando_v_bruto = termo_p + self.Ki * self.integrador + termo_d
        
        # 3. State Update for next iteration
        self.erro_anterior = erro
        
        return comando_v_bruto, erro

    #-----------------------------------------------------------------------------------------------------------------
    # @brief    Updates the integrator state using Back-Calculation Anti-Windup.
    #           This ensures the integrator only accumulates when the system is not saturated.
    #-----------------------------------------------------------------------------------------------------------------
    def atualizar_integrador(self, comando_bruto, comando_real_aplicado, erro, dt):
        
        # 1. Saturation Check (Command effectively sent to the Inverter)
        # Ensure the applied command is within PID limits
        comando_saturado = np.clip(comando_real_aplicado, self.V_MIN, self.V_MAX)
        
        # 2. Anti-Windup Logic (Back-Calculation)
        # Difference between desired (raw) and applied (saturated)
        diferenca_saturacao = comando_bruto - comando_saturado
        
        Kb = 0.0
        if self.Ki > 1e-6:
            Kb = 1.0 / self.Ki 

        # 3. Integrator Update
        # Accumulate error AND subtract the anti-windup correction
        self.integrador += (erro * dt) - (Kb * diferenca_saturacao * dt)

#---------------------------------------------------------------------------------------------------------------------
# 3. FOC Controller Implementation (Vector Control)
#---------------------------------------------------------------------------------------------------------------------
class ControladorFOC:
    """
    @brief      Implementa a cascata de controle FOC completa.
                (Malha de velocidade externa -> Malhas de corrente d/q internas)
    """
    def __init__(self, Kp_vel, Ki_vel, Kd_vel, 
                 Kp_d, Ki_d, Kp_q, Ki_q, 
                 V_BUS_MAX, P):
        
        self.P = P # Pares de polos
        self.V_BUS_MAX = V_BUS_MAX
        
        IQ_MAX = 20.0 
        V_FASE_MAX = V_BUS_MAX / 2.0
        
        # Instantiate the 3 PIDs
        self.pid_velocidade = ControladorPID(Kp_vel, Ki_vel, Kd_vel, -IQ_MAX, IQ_MAX)
        self.pid_d = ControladorPID(Kp_d, Ki_d, 0.0, -V_FASE_MAX, V_FASE_MAX)
        self.pid_q = ControladorPID(Kp_q, Ki_q, 0.0, -V_FASE_MAX, V_FASE_MAX)

    #-----------------------------------------------------------------------------------------------------------------
    # 3.1 Static Math Helpers (Transforms)
    #-----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def clarke_transform(ia, ib, ic):
        """Transformada de Clarke (abc -> alpha, beta) - Amplitude Invariante."""
        ialpha = SQRT_2_3 * (ia - 0.5 * ib - 0.5 * ic)
        ibeta = SQRT_2_3 * (0.5 * SQRT_3 * ib - 0.5 * SQRT_3 * ic)
        return (ialpha, ibeta)

    @staticmethod
    def park_transform(ialpha, ibeta, theta_e):
        """Transformada de Park (alpha, beta -> d, q)."""
        cos_t = np.cos(theta_e)
        sin_t = np.sin(theta_e)
        id = ialpha * cos_t + ibeta * sin_t
        iq = -ialpha * sin_t + ibeta * cos_t
        return (id, iq)

    @staticmethod
    def inverse_park_transform(vd, vq, theta_e):
        """Transformada Inversa de Park (d, q -> alpha, beta)."""
        cos_t = np.cos(theta_e)
        sin_t = np.sin(theta_e)
        valpha = vd * cos_t - vq * sin_t
        vbeta = vd * sin_t + vq * cos_t
        return (valpha, vbeta)

    @staticmethod
    def inverse_clarke_saturado(valpha, vbeta, v_bus_max):
        """Transformada Inversa de Clarke (alpha, beta -> a, b, c) com Saturação."""
        v_lim_fase = v_bus_max / 2.0
        
        v_mag = np.sqrt(valpha**2 + vbeta**2)
        if v_mag > v_lim_fase:
            valpha *= v_lim_fase / v_mag
            vbeta *= v_lim_fase / v_mag

        # Inverse Clarke (Amplitude Invariant)
        va = SQRT_2_3 * valpha
        vb = SQRT_2_3 * (-0.5 * valpha + (SQRT_3 / 2.0) * vbeta)
        vc = SQRT_2_3 * (-0.5 * valpha - (SQRT_3 / 2.0) * vbeta)
        
        # Final Saturation
        va = np.clip(va, -v_lim_fase, v_lim_fase)
        vb = np.clip(vb, -v_lim_fase, v_lim_fase)
        vc = np.clip(vc, -v_lim_fase, v_lim_fase)

        return (va, vb, vc)

    def reset(self):
        """Reseta todos os PIDs internos."""
        self.pid_velocidade.reset()
        self.pid_d.reset()
        self.pid_q.reset()

    def atualizar_ganhos_velocidade(self, Kp, Ki, Kd):
        """Atualiza os ganhos da malha de velocidade (vindos da GUI)."""
        self.pid_velocidade.Kp = Kp
        self.pid_velocidade.Ki = Ki
        self.pid_velocidade.Kd = Kd

    #-----------------------------------------------------------------------------------------------------------------
    # @brief    Executes the complete FOC control step.
    #           Pipeline: Velocity Loop -> Clarke/Park -> Current Loops -> Inv Park/Clarke
    # @param    w_ref: Target velocity
    # @param    w_atual: Actual velocity
    # @param    ia, ib: Phase currents
    # @param    theta_m: Mechanical angle
    #-----------------------------------------------------------------------------------------------------------------
    def calcular_tensao(self, w_ref, w_atual, ia, ib, theta_m, dt):
        
        # 1. Electrical Angle Calculation
        theta_e = np.fmod(self.P * theta_m, 2 * np.pi)
        
        # 2. Velocity Loop (External)
        # Calculate raw PID output for velocity
        comando_bruto_iq, erro_iq = self.pid_velocidade.calcular_comando_bruto(w_ref, w_atual, dt)
        
        # Saturate output to generate Iq Reference
        iq_ref_saturado = np.clip(comando_bruto_iq, self.pid_velocidade.V_MIN, self.pid_velocidade.V_MAX)
        
        # Update Velocity Integrator (Anti-Windup)
        self.pid_velocidade.atualizar_integrador(comando_bruto_iq, iq_ref_saturado, erro_iq, dt)
        
        # Set Current References (Id = 0 for max torque per amp)
        iq_ref = -iq_ref_saturado 
        id_ref = 0.0
        
        # 3. Measurement and Transformation (abc -> dq)
        # Calculate missing phase current
        ic = -ia - ib
        
        # Clarke Transform (abc -> alpha, beta)
        ialpha, ibeta = self.clarke_transform(ia, ib, ic)
        
        # Park Transform (alpha, beta -> d, q)
        id_atual, iq_atual = self.park_transform(ialpha, ibeta, theta_e)
        
        # 4. Current Loops (Internal) -> Generate Voltage Reference (Vd, Vq)
        
        # D-Axis Loop
        comando_bruto_vd, erro_vd = self.pid_d.calcular_comando_bruto(id_ref, id_atual, dt)
        vd_ref = np.clip(comando_bruto_vd, self.pid_d.V_MIN, self.pid_d.V_MAX)
        self.pid_d.atualizar_integrador(comando_bruto_vd, vd_ref, erro_vd, dt)

        # Q-Axis Loop
        comando_bruto_vq, erro_vq = self.pid_q.calcular_comando_bruto(iq_ref, iq_atual, dt)
        vq_ref = np.clip(comando_bruto_vq, self.pid_q.V_MIN, self.pid_q.V_MAX)
        self.pid_q.atualizar_integrador(comando_bruto_vq, vq_ref, erro_vq, dt)

        # 5. Inverse Transformations (dq -> abc)
        # Inverse Park (d, q -> alpha, beta)
        valpha, vbeta = self.inverse_park_transform(vd_ref, vq_ref, theta_e)
        
        # Inverse Clarke (alpha, beta -> a, b, c) & Final Saturation
        va, vb, vc = self.inverse_clarke_saturado(valpha, vbeta, self.V_BUS_MAX)

        return (va, vb, vc)
    
#---------------------------------------------------------------------------------------------------------------------
# 4. 6-Step Control Classes
#---------------------------------------------------------------------------------------------------------------------
class ComutadorTrapezoidal:
    """Decodifica a posição do rotor para comandos de acionamento (1, -1, 0)."""
    
    def __init__(self, pares_polos):
        self.P = pares_polos

    def obter_comandos_de_fase(self, theta_m):
        """Retorna [Fase A, Fase B, Fase C] (1:High, -1:Low, 0:Off)."""
        
        theta_e = self.P * theta_m
        theta_norm = np.fmod(theta_e, 2 * np.pi)
        
        if theta_norm < 0: theta_norm += 2 * np.pi
            
        setor = int(theta_norm * 6 / (2 * np.pi))
        
        # Commutation Table (Standard 6-Step)
        tabela = {
            0: [1, -1, 0],  # Setor 0 (0-60°): A-B
            1: [1, 0, -1],  # Setor 1 (60-120°): A-C
            2: [0, 1, -1],  # Setor 2 (120-180°): B-C
            3: [-1, 1, 0],  # Setor 3 (180-240°): B-A
            4: [-1, 0, 1],  # Setor 4 (240-300°): C-A
            5: [0, -1, 1]   # Setor 5 (300-360°): C-B
        }
        return tabela.get(setor, [0, 0, 0])

class Inversor:
    """Simula a ponte H trifásica que aplica a tensão ao motor (Modo 6-Step)."""

    def __init__(self, V_bus_max):
        self.V_BUS_MAX = V_bus_max
    
    def aplicar_tensao_nas_fases(self, comando_v, comandos_fase):
        """Calcula as tensões Va, Vb, Vc para o modo 6-Step."""
        
        # Clip input command
        V_aplicada = np.clip(comando_v, 0, self.V_BUS_MAX)
        Va, Vb, Vc = 0.0, 0.0, 0.0
        
        # Apply Line Voltage to energized phases
        for i, cmd in enumerate(comandos_fase):
            if cmd == 1:
                if i == 0: Va = V_aplicada / 2
                elif i == 1: Vb = V_aplicada / 2
                else: Vc = V_aplicada / 2
            elif cmd == -1:
                if i == 0: Va = -V_aplicada / 2
                elif i == 1: Vb = -V_aplicada / 2
                else: Vc = -V_aplicada / 2
            
        return (Va, Vb, Vc)