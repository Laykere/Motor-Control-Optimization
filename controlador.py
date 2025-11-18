# controlador.py
import numpy as np


SQRT_2_3 = np.sqrt(2.0/3.0)
SQRT_3 = np.sqrt(3.0)
INV_SQRT_3 = 1.0 / SQRT_3

class ControladorPID:
    """Implementa o controlador PID (MODIFICADO para anti-windup externo)."""

    def __init__(self, Kp, Ki, Kd, V_MIN, V_MAX): 
        self.Kp = Kp  
        self.Ki = Ki
        self.Kd = Kd
        
        self.V_MIN = V_MIN
        self.V_MAX = V_MAX
        
        self.integrador = 0.0
        self.erro_anterior = 0.0

    def reset(self):
        """Zera o estado do PID."""
        self.integrador = 0.0
        self.erro_anterior = 0.0
        
    def calcular_comando_bruto(self, referencia, valor_atual, dt):
        """
        Calcula o comando PID *antes* da saturação.
        Retorna (comando_bruto, erro)
        """
        erro = referencia - valor_atual
        
        # 1. Termos P e D
        termo_p = self.Kp * erro
        derivada = (erro - self.erro_anterior) / dt if dt > 0 else 0
        termo_d = self.Kd * derivada
        
        # 2. Comando Bruto (P + I + D)
        comando_v_bruto = termo_p + self.Ki * self.integrador + termo_d
        
        # 3. Atualização para a próxima iteração
        self.erro_anterior = erro
        
        return comando_v_bruto, erro

    def atualizar_integrador(self, comando_bruto, comando_real_aplicado, erro, dt):
        """
        Atualiza o integrador (Anti-Windup) usando o comando que
        foi *realmente* aplicado (após rampas ou clipes externos).
        """
        
        # 1. Saturação (Comando que REALMENTE foi para o Inversor)
        # Garantimos que o comando_real_aplicado esteja dentro dos limites do PID
        comando_saturado = np.clip(comando_real_aplicado, self.V_MIN, self.V_MAX)
        
        # 2. LÓGICA ANTI-WINDUP (Back-Calculation)
        # Diferença entre o que o PID quis (bruto) e o que foi aplicado (saturado)
        diferenca_saturacao = comando_bruto - comando_saturado
        
        Kb = 0.0
        if self.Ki > 1e-6:
            Kb = 1.0 / self.Ki 

        # 3. Atualização do Integrador
        # (Acumula o erro E subtrai a correção anti-windup)
        self.integrador += (erro * dt) - (Kb * diferenca_saturacao * dt)

# --- CLASSE FOC (AGORA INCLUI TRANSFORMAÇÕES) ---
class ControladorFOC:
    """
    Implementa a cascata de controle FOC completa.
    (Malha de velocidade externa -> Malhas de corrente d/q internas)
    """
    def __init__(self, Kp_vel, Ki_vel, Kd_vel, 
                 Kp_d, Ki_d, Kp_q, Ki_q, 
                 V_BUS_MAX, P):
        
        self.P = P # Pares de polos
        self.V_BUS_MAX = V_BUS_MAX
        
        IQ_MAX = 20.0 
        V_FASE_MAX = V_BUS_MAX / 2.0
        
        # Instancia os 3 PIDs (AGORA 'ControladorPID' está definido)
        self.pid_velocidade = ControladorPID(Kp_vel, Ki_vel, Kd_vel, -IQ_MAX, IQ_MAX)
        self.pid_d = ControladorPID(Kp_d, Ki_d, 0.0, -V_FASE_MAX, V_FASE_MAX)
        self.pid_q = ControladorPID(Kp_q, Ki_q, 0.0, -V_FASE_MAX, V_FASE_MAX)

    # --- TRANSFORMAÇÕES (AGORA SÃO MÉTODOS ESTÁTICOS) ---
    
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

        # Inversa de Clarke (Amplitude Invariante)
        va = SQRT_2_3 * valpha
        vb = SQRT_2_3 * (-0.5 * valpha + (SQRT_3 / 2.0) * vbeta)
        vc = SQRT_2_3 * (-0.5 * valpha - (SQRT_3 / 2.0) * vbeta)
        
        # Saturação final (embora a saturação do vetor já ajude)
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

    def calcular_tensao(self, w_ref, w_atual, ia, ib, theta_m, dt):
        """Executa um passo de controle FOC completo."""
        
        # 1. Ângulo Elétrico
        theta_e = np.fmod(self.P * theta_m, 2 * np.pi)
        
        # 2. Malha de Velocidade (Externa)
        comando_bruto_iq, erro_iq = self.pid_velocidade.calcular_comando_bruto(w_ref, w_atual, dt)
        
        # O "comando real" aqui é a saída saturada, que vira a referência Iq
        iq_ref_saturado = np.clip(comando_bruto_iq, self.pid_velocidade.V_MIN, self.pid_velocidade.V_MAX)
        
        # Atualiza o anti-windup da velocidade
        self.pid_velocidade.atualizar_integrador(comando_bruto_iq, iq_ref_saturado, erro_iq, dt)
        
        # Inverte o sinal para a referência de corrente
        iq_ref = -iq_ref_saturado 
        id_ref = 0.0
        
        # 3. Medição e Transformação (abc -> dq)
        ic = -ia - ib
        ialpha, ibeta = self.clarke_transform(ia, ib, ic)
        id_atual, iq_atual = self.park_transform(ialpha, ibeta, theta_e)
        
        # 4. Malhas de Corrente (Internas) -> Geram Referência de Tensão (Vd, Vq)
        
        # Malha D
        comando_bruto_vd, erro_vd = self.pid_d.calcular_comando_bruto(id_ref, id_atual, dt)
        vd_ref = np.clip(comando_bruto_vd, self.pid_d.V_MIN, self.pid_d.V_MAX)
        self.pid_d.atualizar_integrador(comando_bruto_vd, vd_ref, erro_vd, dt)

        # Malha Q
        comando_bruto_vq, erro_vq = self.pid_q.calcular_comando_bruto(iq_ref, iq_atual, dt)
        vq_ref = np.clip(comando_bruto_vq, self.pid_q.V_MIN, self.pid_q.V_MAX)
        self.pid_q.atualizar_integrador(comando_bruto_vq, vq_ref, erro_vq, dt)

        # 5. Transformação Inversa (dq -> abc)
        valpha, vbeta = self.inverse_park_transform(vd_ref, vq_ref, theta_e)
        
        # 6. SVPWM (Simplificado) e Saturação
        va, vb, vc = self.inverse_clarke_saturado(valpha, vbeta, self.V_BUS_MAX)

        return (va, vb, vc)
    
# --- CLASSES DE CONTROLE 6-STEP ---
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
        
        # Tabela de Comutação A-B-C Padrão
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
        
        V_aplicada = np.clip(comando_v, 0, self.V_BUS_MAX)
        Va, Vb, Vc = 0.0, 0.0, 0.0
        
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
