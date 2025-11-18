# gui_app.py (Versão PySide6 + PyQtGraph)

import sys
import time
import numpy as np

# 1. Importações do Qt (PySide6)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QFrame, QScrollArea, QStackedWidget, QCheckBox,
    QMessageBox
)
from PyQt6.QtCore import QTimer, Qt

# 2. Importação do PyQtGraph
import pyqtgraph as pg

# 3. Importações da sua lógica de backend (NENHUMA MUDANÇA AQUI)
from motor_simulator import MotorBLDC
from controlador import ControladorPID, ComutadorTrapezoidal, Inversor, ControladorFOC

# 4. Classe Simulador (IDÊNTICA À ANTERIOR, copiada para cá)
# Nenhuma mudança necessária nesta classe.
class Simulador:
    """Orquestra o motor, controlador e armazena os dados históricos."""
    
    def __init__(self, dt):

        self.dt = dt
        self.tempo = 0.0
        self.referencia_omega = 50.0
        self.control_mode = '6-Step' 
        
        # 1. Instanciar Componentes
        R, L, M, J, B, Ke, P = 1.0, 0.02, 0.006, 0.002, 0.0001, 0.2, 2
        self.V_BUS_MAX = 80.0 
        self.P = P
        
        self.motor = MotorBLDC(R, L, M, J, B, Ke, P)
        self.motor.estado[3] = 0.001 
        self.motor.set_bemf_shape('trapezoidal')
        
        self.voltage_ramp_rate = 200.0 
        self.comando_v_anterior = 0.0  
        
        # Ganhos separados para cada modo
        self.pid_ganhos_6step = {'Kp': 1.5, 'Ki': 0.1, 'Kd': 0.01}
        self.pid_ganhos_foc = {'Kp': 6.0, 'Ki': 10.0, 'Kd': 0.01}
        
        # --- Componentes do Modo 6-Step ---
        self.pid_vel_6step = ControladorPID(
            self.pid_ganhos_6step['Kp'], self.pid_ganhos_6step['Ki'], self.pid_ganhos_6step['Kd'], 
            0.0, self.V_BUS_MAX
        )
        self.comutador = ComutadorTrapezoidal(P)
        self.inversor_6step = Inversor(self.V_BUS_MAX)

        # --- Componentes do Modo FOC ---
        Kp_d, Ki_d = 0.8, 0.2
        Kp_q, Ki_q = 0.8, 0.2
        
        self.controlador_foc = ControladorFOC(
            self.pid_ganhos_foc['Kp'], self.pid_ganhos_foc['Ki'], self.pid_ganhos_foc['Kd'], 
            Kp_d, Ki_d, Kp_q, Ki_q, 
            self.V_BUS_MAX, self.P
        )

        # 2. Histórico de Dados
        # Limita o histórico para não consumir memória infinita
        self.max_hist_points = 10000 
        self.hist_tempo = []
        self.hist_omega = []
        self.hist_referencia = []
        self.hist_Va = []
        self.hist_Vb = []
        self.hist_Vc = []
        self.hist_Ia = []
        self.hist_Ib = []
        self.hist_Ic = []
        
    def rodar_passo(self):
        """Executa um passo de simulação (lógica de modo dual)."""
        
        ia, ib, omega, theta = self.motor.estado
        
        if self.control_mode == '6-Step':
            # 1. Lógica 6-Step (com Rampa e Anti-Windup Corrigido)
            
            # 1.A. Calcula a tensão que o PID 'deseja' (bruto) e o erro
            comando_v_pid_bruto, erro_pid = self.pid_vel_6step.calcular_comando_bruto(self.referencia_omega, omega, self.dt)
            
            # 1.B. Calcula o comando que o PID *teria* enviado se estivesse sozinho
            # (Limitado apenas pelo V_BUS)
            comando_v_saturado_pid = np.clip(comando_v_pid_bruto, 
                                             self.pid_vel_6step.V_MIN, 
                                             self.pid_vel_6step.V_MAX)
            
            # 1.C. ATUALIZA O INTEGRADOR (Anti-Windup)
            # O anti-windup só se importa com a saturação do V_BUS.
            # Ele não deve saber sobre a rampa.
            self.pid_vel_6step.atualizar_integrador(comando_v_pid_bruto, comando_v_saturado_pid, erro_pid, self.dt)
            
            # 1.D. Agora, aplica a RAMPA à saída JÁ SATURADA do PID
            max_v_step = self.voltage_ramp_rate * self.dt 
            comando_v_real = np.clip(comando_v_saturado_pid, 
                                     self.comando_v_anterior - max_v_step, 
                                     self.comando_v_anterior + max_v_step)
            
            # 1.E. Salva a saída da rampa para o próximo ciclo
            self.comando_v_anterior = comando_v_real
            
            # 1.F. Envia o comando real (pós-rampa) para o inversor
            comandos_fase = self.comutador.obter_comandos_de_fase(theta)
            Va, Vb, Vc = self.inversor_6step.aplicar_tensao_nas_fases(comando_v_real, comandos_fase)
        
        else: # self.control_mode == 'FOC'
            # 1. Lógica FOC (que agora lida com seu próprio anti-windup interno)
            Va, Vb, Vc = self.controlador_foc.calcular_tensao(
                self.referencia_omega, omega, ia, ib, theta, self.dt
            )
            # Reseta a rampa do 6-Step
            self.comando_v_anterior = 0.0 

        # 3. Avanço do Motor
        self.motor.aplicar_tensoes(Va, Vb, Vc)
        if not self.motor.avancar_tempo(self.dt):
            return False 

        # 4. Salvar Histórico
        self.hist_tempo.append(self.tempo)
        self.hist_omega.append(omega)
        self.hist_referencia.append(self.referencia_omega)
        self.hist_Va.append(Va)
        self.hist_Vb.append(Vb)
        self.hist_Vc.append(Vc)
        
        ic = -ia - ib 
        self.hist_Ia.append(ia)
        self.hist_Ib.append(ib)
        self.hist_Ic.append(ic)
        
        self.tempo += self.dt
        return True

class AplicacaoGUI(QWidget):
    """Cria a janela principal e o loop de visualização usando Qt."""
    
    def __init__(self, simulador):
        super().__init__()
        self.simulador = simulador
        self.rodando = False
        
        # Configurações de simulação
        self.batch_size = 100 
        self.is_realtime_locked = False
        self.start_real_time = 0.0 
        self.start_sim_time = 0.0  
        
        # Configura o QTimer (o substituto do 'self.after' do Tkinter)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop_simulacao)
        
        self.init_ui()
        
    def init_ui(self):
        """Configura todos os widgets e layouts."""
        
        # Layout principal (Horizontal: Controles | Gráficos)
        main_layout = QHBoxLayout(self)
        
        # --- Lado Esquerdo: Controles ---
        self.criar_painel_controles()
        
        # --- Lado Direito: Gráficos ---
        self.criar_painel_graficos()
        
        # Adiciona os painéis ao layout principal
        main_layout.addWidget(self.scroll_area, 1) # Proporção 1
        main_layout.addWidget(self.plot_widget_container, 3) # Proporção 3 (mais largo)
        
        # Configurações da Janela
        self.setWindowTitle("Simulador BLDC (6-Step e FOC) - Versão Qt")
        self.setGeometry(100, 100, 1200, 800) # (x, y, largura, altura)

    def criar_painel_controles(self):
        """Cria a área de rolagem e todos os widgets de controle."""
        
        # QScrollArea substitui o Canvas+Scrollbar do Tkinter
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(260) 
        
        # Widget e Layout internos para a área de rolagem
        scroll_content = QWidget()
        frame_controle = QVBoxLayout(scroll_content) # QVBoxLayout é vertical
        frame_controle.setAlignment(Qt.AlignmentFlag.AlignTop) # Alinha no topo
        
        self.scroll_area.setWidget(scroll_content)

        # --- Botão Iniciar/Parar ---
        self.btn_toggle = QPushButton("INICIAR SIMULAÇÃO", self)
        self.btn_toggle.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.btn_toggle.clicked.connect(self.toggle_simulacao)
        frame_controle.addWidget(self.btn_toggle)
        
        # --- Checkbox Tempo Real ---
        self.chk_realtime = QCheckBox("Limitar a 1x Tempo Real", self)
        self.chk_realtime.toggled.connect(self.on_realtime_toggled)
        frame_controle.addWidget(self.chk_realtime)
        
        self.status_label = QLabel("Status: N/A", self)
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        frame_controle.addWidget(self.status_label)
        
        # --- Referência ---
        frame_controle.addWidget(self.criar_linha_separadora())
        frame_controle.addWidget(QLabel("Ref. Velocidade (rad/s):"))
        self.ref_entry = QLineEdit(str(self.simulador.referencia_omega), self)
        self.btn_apply_ref = QPushButton("Aplicar Referência", self)
        self.btn_apply_ref.clicked.connect(self.aplicar_referencia)
        frame_controle.addWidget(self.ref_entry)
        frame_controle.addWidget(self.btn_apply_ref)

        # --- Modo de Controle ---
        frame_controle.addWidget(self.criar_linha_separadora())
        frame_controle.addWidget(QLabel("--- Modo de Controle ---"))
        self.control_mode_combo = QComboBox(self)
        self.control_mode_combo.addItems(['6-Step', 'FOC'])
        frame_controle.addWidget(self.control_mode_combo)

        # --- Ganhos PID (com QStackedWidget) ---
        frame_controle.addWidget(QLabel("--- Ganhos PID (Velocidade) ---"))
        
        # QStackedWidget é a forma correta no Qt de alternar painéis
        self.pid_stack = QStackedWidget(self)
        
        # Ganhos 6-Step (Página 0)
        self.frame_ganhos_6step = QWidget()
        layout_6step = QVBoxLayout(self.frame_ganhos_6step)
        self.kp_entry_6step = QLineEdit(str(self.simulador.pid_ganhos_6step['Kp']), self)
        self.ki_entry_6step = QLineEdit(str(self.simulador.pid_ganhos_6step['Ki']), self)
        self.kd_entry_6step = QLineEdit(str(self.simulador.pid_ganhos_6step['Kd']), self)
        layout_6step.addWidget(QLabel("Kp (6-Step):"))
        layout_6step.addWidget(self.kp_entry_6step)
        layout_6step.addWidget(QLabel("Ki (6-Step):"))
        layout_6step.addWidget(self.ki_entry_6step)
        layout_6step.addWidget(QLabel("Kd (6-Step):"))
        layout_6step.addWidget(self.kd_entry_6step)
        
        # Ganhos FOC (Página 1)
        self.frame_ganhos_foc = QWidget()
        layout_foc = QVBoxLayout(self.frame_ganhos_foc)
        self.kp_entry_foc = QLineEdit(str(self.simulador.pid_ganhos_foc['Kp']), self)
        self.ki_entry_foc = QLineEdit(str(self.simulador.pid_ganhos_foc['Ki']), self)
        self.kd_entry_foc = QLineEdit(str(self.simulador.pid_ganhos_foc['Kd']), self)
        layout_foc.addWidget(QLabel("Kp (FOC):"))
        layout_foc.addWidget(self.kp_entry_foc)
        layout_foc.addWidget(QLabel("Ki (FOC):"))
        layout_foc.addWidget(self.ki_entry_foc)
        layout_foc.addWidget(QLabel("Kd (FOC):"))
        layout_foc.addWidget(self.kd_entry_foc)
        
        self.pid_stack.addWidget(self.frame_ganhos_6step) # Índice 0
        self.pid_stack.addWidget(self.frame_ganhos_foc)  # Índice 1
        frame_controle.addWidget(self.pid_stack)
        
        # Conecta o ComboBox ao StackedWidget
        self.control_mode_combo.currentIndexChanged.connect(self.pid_stack.setCurrentIndex)
        
        # --- Controles de Simulação ---
        frame_controle.addWidget(self.criar_linha_separadora())
        frame_controle.addWidget(QLabel("--- Controles de Simulação ---"))
        self.dt_entry = QLineEdit(str(self.simulador.dt), self)
        self.batch_entry = QLineEdit(str(self.batch_size), self)
        self.ramp_entry = QLineEdit(str(self.simulador.voltage_ramp_rate), self)
        frame_controle.addWidget(QLabel("DT (s):"))
        frame_controle.addWidget(self.dt_entry)
        frame_controle.addWidget(QLabel("Passos/Frame:"))
        frame_controle.addWidget(self.batch_entry)
        frame_controle.addWidget(QLabel("Rampa de Tensão (V/s):"))
        frame_controle.addWidget(self.ramp_entry)

        # --- Janela de Gráfico Rolante ---
        self.chk_rolling_window = QCheckBox("Janela de Gráfico Rolante", self)
        self.chk_rolling_window.setChecked(False) # Começa desligado
        frame_controle.addWidget(self.chk_rolling_window)
        
        frame_controle.addWidget(QLabel("Tamanho da Janela (s):"))
        self.window_entry = QLineEdit("5.0", self) # Valor padrão de 5.0s
        frame_controle.addWidget(self.window_entry)        

        # --- Simulação em Batch ---
        frame_controle.addWidget(self.criar_linha_separadora())
        frame_controle.addWidget(QLabel("--- Simulação em Batch ---"))
        self.duration_entry = QLineEdit("1.0", self)
        self.btn_batch_run = QPushButton("Rodar Simulação (Batch)", self)
        self.btn_batch_run.setStyleSheet("background-color: orange;")
        self.btn_batch_run.clicked.connect(self.run_batch_simulation)
        frame_controle.addWidget(QLabel("Duração Sim. (s):"))
        frame_controle.addWidget(self.duration_entry)
        frame_controle.addWidget(self.btn_batch_run)

        # --- Botão Aplicar/Reiniciar ---
        self.btn_apply = QPushButton("Aplicar Ganhos e Reiniciar", self)
        self.btn_apply.setStyleSheet("background-color: lightblue;")
        self.btn_apply.clicked.connect(self.aplicar_ganhos_e_reiniciar)
        frame_controle.addWidget(self.btn_apply)

    def criar_painel_graficos(self):
        """Cria os 3 gráficos usando PyQtGraph."""
        
        # Define o fundo padrão do PyQtGraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_widget_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget_container)
        
        # Gráfico de Velocidade
        self.plot_velocidade = pg.PlotWidget(title="Velocidade")
        self.plot_velocidade.addLegend()
        self.plot_velocidade.setLabel('left', 'Velocidade (rad/s)')
        self.plot_velocidade.showGrid(x=True, y=True)
        self.linha_omega = self.plot_velocidade.plot(pen=pg.mkPen('b', width=2), name='ω Real (rad/s)')
        self.linha_ref = self.plot_velocidade.plot(pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2), name='ω Ref (rad/s)')
        plot_layout.addWidget(self.plot_velocidade)

        # Gráfico de Tensão
        self.plot_tensao = pg.PlotWidget(title="Tensão de Fase")
        self.plot_tensao.addLegend()
        self.plot_tensao.setLabel('left', 'Tensão (V)')
        self.plot_tensao.showGrid(x=True, y=True)
        self.linha_Va = self.plot_tensao.plot(pen=pg.mkPen('b', width=1), name='Va (V)')
        self.linha_Vb = self.plot_tensao.plot(pen=pg.mkPen('g', width=1), name='Vb (V)')
        self.linha_Vc = self.plot_tensao.plot(pen=pg.mkPen('r', width=1), name='Vc (V)')
        v_lim = self.simulador.V_BUS_MAX / 2 * 1.1
        self.plot_tensao.setYRange(-v_lim, v_lim)
        plot_layout.addWidget(self.plot_tensao)
        
        # Gráfico de Corrente
        self.plot_corrente = pg.PlotWidget(title="Corrente de Fase")
        self.plot_corrente.addLegend()
        self.plot_corrente.setLabel('left', 'Corrente (A)')
        self.plot_corrente.setLabel('bottom', 'Tempo (s)')
        self.plot_corrente.showGrid(x=True, y=True)
        self.linha_Ia = self.plot_corrente.plot(pen=pg.mkPen('b', width=1), name='Ia (A)')
        self.linha_Ib = self.plot_corrente.plot(pen=pg.mkPen('g', width=1), name='Ib (A)')
        self.linha_Ic = self.plot_corrente.plot(pen=pg.mkPen('r', width=1), name='Ic (A)')
        self.plot_corrente.setYRange(-25, 25) # Limite fixo de corrente
        plot_layout.addWidget(self.plot_corrente)

        # Linka os eixos X
        self.plot_tensao.setXLink(self.plot_velocidade)
        self.plot_corrente.setXLink(self.plot_velocidade)
        
    def criar_linha_separadora(self):
        """Helper para criar uma linha HFrame."""
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
   
    def aplicar_referencia(self):
        try:
            nova_ref = float(self.ref_entry.text())
            self.simulador.referencia_omega = nova_ref
            print(f"Referência de velocidade atualizada para {nova_ref} rad/s")
        except ValueError:
            QMessageBox.warning(self, "Erro", "Referência inválida. Insira um número.")
            
    def on_realtime_toggled(self, checked):
        self.is_realtime_locked = checked

    def aplicar_ganhos_e_reiniciar(self):
        """Lê os widgets, atualiza os ganhos, MODO, e reinicia a simulação."""
        try:
            # 1. Lê os ganhos de AMBOS os painéis
            new_kp_6step = float(self.kp_entry_6step.text())
            new_ki_6step = float(self.ki_entry_6step.text())
            new_kd_6step = float(self.kd_entry_6step.text())
            
            new_kp_foc = float(self.kp_entry_foc.text())
            new_ki_foc = float(self.ki_entry_foc.text())
            new_kd_foc = float(self.kd_entry_foc.text())
            
            # 2. Lê os valores de simulação
            new_dt = float(self.dt_entry.text())
            new_batch_size = int(self.batch_entry.text())
            new_ramp_rate = float(self.ramp_entry.text())
            
            # 3. Atualiza os valores
            self.simulador.dt = new_dt
            self.batch_size = new_batch_size
            self.simulador.voltage_ramp_rate = new_ramp_rate
            
            # 4. Lê o modo de controle
            new_mode = self.control_mode_combo.currentText()

            # 5. Atualiza os ganhos nos controladores
            self.simulador.pid_vel_6step.Kp = new_kp_6step
            self.simulador.pid_vel_6step.Ki = new_ki_6step
            self.simulador.pid_vel_6step.Kd = new_kd_6step
            
            self.simulador.controlador_foc.atualizar_ganhos_velocidade(
                new_kp_foc, new_ki_foc, new_kd_foc
            )
            
            # 6. Zera o estado
            self.simulador.pid_vel_6step.reset()
            self.simulador.controlador_foc.reset()
            self.simulador.comando_v_anterior = 0.0 
            
            self.simulador.control_mode = new_mode
            if new_mode == 'FOC':
                self.simulador.motor.set_bemf_shape('sinusoidal')
            else: # '6-Step'
                self.simulador.motor.set_bemf_shape('trapezoidal')
                
            self.simulador.motor.estado = np.zeros(4)
            self.simulador.motor.estado[3] = 0.001 
            self.simulador.tempo = 0.0
            
            # 7. Limpa o histórico para o gráfico
            self.simulador.hist_tempo = []
            self.simulador.hist_omega = []
            self.simulador.hist_referencia = []
            self.simulador.hist_Va = []
            self.simulador.hist_Vb = []
            self.simulador.hist_Vc = []
            self.simulador.hist_Ia = []
            self.simulador.hist_Ib = []
            self.simulador.hist_Ic = []
            
            print(f"Modo: {new_mode}. Ganhos 6-Step: {new_kp_6step}/{new_ki_6step}/{new_kd_6step}. Ganhos FOC: {new_kp_foc}/{new_ki_foc}/{new_kd_foc}")
            print(f"Simulador atualizado: DT={new_dt}s, Passos/Frame={new_batch_size}, Rampa={new_ramp_rate} V/s")
            
            # --- SUBSTITUA A LINHA 'self.atualizar_grafico()' POR ESTA: ---
            # Esta chamada força a limpeza visual dos gráficos
            self.update_plots(force_clear=True)
            # --- FIM DA ALTERAÇÃO ---
            
        except ValueError:
            QMessageBox.warning(self, "Erro", "Valores inválidos. Insira números válidos para Ganhos, DT e Passos.")

    def toggle_simulacao(self):
        self.rodando = not self.rodando
        if self.rodando:
            self.btn_toggle.setText("PARAR SIMULAÇÃO")
            self.btn_toggle.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.btn_batch_run.setEnabled(False) # Desabilita o botão de batch

            self.start_real_time = time.time()
            self.start_sim_time = self.simulador.tempo
            
            # Inicia o QTimer. O delay é gerenciado dentro do loop.
            self.timer.start(1) 
        else:
            self.timer.stop()
            self.btn_toggle.setText("INICIAR SIMULAÇÃO")
            self.btn_toggle.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            self.btn_batch_run.setEnabled(True) # Reabilita o botão de batch
            self.status_label.setText("Parado.")
            self.status_label.setStyleSheet("color: gray;")
            self.update_plots()

    def loop_simulacao(self):
        """Este método é chamado pelo QTimer."""
        if not self.rodando:
            return

        # 1. Roda o "batch" de simulação
        try:
            for _ in range(self.batch_size):
                if not self.simulador.rodar_passo():
                    self.toggle_simulacao() # Para em caso de erro
                    return
        except Exception as e:
            QMessageBox.critical(self, "Erro de Simulação", f"Erro na física: {e}")
            self.toggle_simulacao()
            return
        
        # 2. Atualiza o gráfico (visual)
        self.update_plots()

        # 3. Lógica de "Delay" e "Status"
        delay_ms = 1 # Padrão: 1ms (Roda o mais rápido possível)
        status_text = "Rápido (Tempo Real Desligado)"
        status_color = "gray"

        if self.is_realtime_locked:
            sim_time_elapsed = self.simulador.tempo - self.start_sim_time
            real_time_elapsed = time.time() - self.start_real_time
            time_diff_s = sim_time_elapsed - real_time_elapsed
            
            if time_diff_s > 0.01:
                delay_ms = int(time_diff_s * 1000)
                status_text = f"Sincronizado (Aguardando {delay_ms}ms)"
                status_color = "green"
            elif time_diff_s < -0.1:
                delay_ms = 1 
                status_text = f"Atrasado (Carga Alta: {time_diff_s:.2f}s)"
                status_color = "red"
            else:
                delay_ms = 1 
                status_text = "Sincronizado"
                status_color = "green"
        
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-style: italic;")
        
        # Agenda o próximo loop
        self.timer.start(max(1, delay_ms))

    def run_batch_simulation(self):
        """Roda a simulação inteira de uma vez (modo 'batch')."""
        
        if self.rodando:
            QMessageBox.warning(self, "Erro", "Pare a simulação interativa antes de rodar em batch.")
            return

        try:
            duracao = float(self.duration_entry.text())
            if duracao <= 0: raise ValueError("Duração deve ser positiva")
        except ValueError:
            QMessageBox.warning(self, "Erro", "Duração inválida. Insira um número positivo.")
            return

        try:
            # 2. Limpa e prepara a simulação
            self.aplicar_ganhos_e_reiniciar()
            
            # 3. Atualiza a GUI para mostrar "Rodando..."
            self.status_label.setText(f"Rodando batch até {duracao}s...")
            self.status_label.setStyleSheet("color: blue; font-style: italic;")
            QApplication.processEvents() # Força a atualização da GUI

            # 4. Roda a simulação (Isso irá congelar a GUI)
            print("Iniciando simulação em batch...")
            start_batch_time = time.time()
            
            while self.simulador.tempo < duracao:
                if not self.simulador.rodar_passo():
                    raise Exception("Erro no solver do motor.")
            
            end_batch_time = time.time()
            print(f"Batch concluído em {end_batch_time - start_batch_time:.4f}s reais.")

            # 5. Atualiza o gráfico com os resultados finais
            self.update_plots()
            self.status_label.setText("Batch concluído.")
            self.status_label.setStyleSheet("color: green; font-style: italic;")

        except Exception as e:
            error_msg = f"Erro no batch: {e}"
            print(error_msg)
            QMessageBox.critical(self, "Erro de Simulação", error_msg)
            self.status_label.setText("Erro no batch.")
            self.status_label.setStyleSheet("color: red; font-style: italic;")

# gui_app.py

    def update_plots(self, force_clear=False):
        """Atualiza os gráficos do PyQtGraph com os novos dados."""
        
        # Se forçado a limpar (ex: ao reiniciar), apaga todos os dados do gráfico
        if force_clear:
            self.linha_omega.setData(x=[], y=[])
            self.linha_ref.setData(x=[], y=[])
            self.linha_Va.setData(x=[], y=[])
            self.linha_Vb.setData(x=[], y=[])
            self.linha_Vc.setData(x=[], y=[])
            self.linha_Ia.setData(x=[], y=[])
            self.linha_Ib.setData(x=[], y=[])
            self.linha_Ic.setData(x=[], y=[])
            self.plot_velocidade.setXRange(0, 0.1, padding=0)
            return

        tempo = self.simulador.hist_tempo
        if not tempo: # Não faz nada se o histórico estiver vazio
            return
            
        # --- Lógica de Fatiamento de Dados (Slicing) ---
        
        # Por padrão, usa todos os dados
        tempo_plot = tempo
        omega_plot = self.simulador.hist_omega
        ref_plot = self.simulador.hist_referencia
        va_plot = self.simulador.hist_Va
        vb_plot = self.simulador.hist_Vb
        vc_plot = self.simulador.hist_Vc
        ia_plot = self.simulador.hist_Ia
        ib_plot = self.simulador.hist_Ib
        ic_plot = self.simulador.hist_Ic
        
        current_time = tempo[-1]
        t_start = 0.0

        # Verifica se "Janela Rolante" está LIGADA e se a simulação está RODANDO
        if self.chk_rolling_window.isChecked() and self.rodando:
            try:
                window_size = float(self.window_entry.text())
                if window_size <= 0: window_size = 5.0
            except ValueError:
                window_size = 5.0
            
            t_start = max(0.0, current_time - window_size)
            
            # Encontra o índice inicial para fatiar os dados
            # (np.searchsorted é muito rápido para encontrar o índice em uma lista ordenada)
            start_index = np.searchsorted(np.array(tempo), t_start)
            
            # Fatia (Slice) todos os dados para plotar apenas a janela
            tempo_plot = tempo[start_index:]
            omega_plot = self.simulador.hist_omega[start_index:]
            ref_plot = self.simulador.hist_referencia[start_index:]
            va_plot = self.simulador.hist_Va[start_index:]
            vb_plot = self.simulador.hist_Vb[start_index:]
            vc_plot = self.simulador.hist_Vc[start_index:]
            ia_plot = self.simulador.hist_Ia[start_index:]
            ib_plot = self.simulador.hist_Ib[start_index:]
            ic_plot = self.simulador.hist_Ic[start_index:]
        else:
            # Se pausado ou se a janela rolante estiver desligada,
            # garante que o X Range mostre tudo de 0 até o fim
            t_start = 0.0
            
        # --- Atualiza os dados das linhas (com os dados fatiados ou completos) ---
        self.linha_omega.setData(x=tempo_plot, y=omega_plot)
        self.linha_ref.setData(x=tempo_plot, y=ref_plot)
        
        self.linha_Va.setData(x=tempo_plot, y=va_plot)
        self.linha_Vb.setData(x=tempo_plot, y=vb_plot)
        self.linha_Vc.setData(x=tempo_plot, y=vc_plot)
        
        self.linha_Ia.setData(x=tempo_plot, y=ia_plot)
        self.linha_Ib.setData(x=tempo_plot, y=ib_plot)
        self.linha_Ic.setData(x=tempo_plot, y=ic_plot)
        
        # --- Atualiza os limites (Range) ---
        self.plot_velocidade.setXRange(t_start, current_time, padding=0.01)

        # Lógica de atualização do Eixo Y (Idêntico a antes)
        y_data = self.simulador.hist_omega + self.simulador.hist_referencia
        if y_data:
             min_y = min(y_data) * 1.1 - 5 
             max_y = max(y_data) * 1.1 + 5
             if min_y == max_y or abs(min_y - max_y) < 1: 
                 min_y -= 10
                 max_y += 10
             self.plot_velocidade.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
             self.plot_velocidade.setYRange(min_y, max_y, padding=0)


# --- Bloco Principal de Execução ---
if __name__ == '__main__':
    DT = 0.0001 # 0.1 ms (Um DT menor é melhor para a estabilidade do FOC)
    
    # Configurações do PyQtGraph (Opcional, mas melhora a suavidade)
    pg.setConfigOptions(antialias=True)
    
    # Cria a Aplicação Qt
    app = QApplication(sys.argv)
    
    # Instancia o simulador e a GUI
    sim = Simulador(dt=DT)
    window = AplicacaoGUI(sim)
    window.show()
    
    # Executa o loop principal do Qt
    sys.exit(app.exec())