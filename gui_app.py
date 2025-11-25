# gui_app.py
import sys
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QFrame, QScrollArea, QStackedWidget, QCheckBox,
    QMessageBox, QGroupBox, QFormLayout, QSlider, QProgressBar
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
from simulator import Simulator

class GuiApplication(QWidget):
    
    def __init__(self, simulator_instance):
        super().__init__()
        self.sim = simulator_instance
        self.is_running = False
        
        # OPTIMIZATION: Reduced batch size for better responsiveness
        self.batch_size = 20 
        
        self.is_realtime_locked = False
        self.start_real_time = 0.0 
        self.start_sim_time = 0.0  
        self.timer = QTimer(self)
        self.timer.setSingleShot(True) # Ensure timer only fires once per call
        self.timer.timeout.connect(self.simulation_loop)
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        self.create_control_panel()
        self.create_plot_panel()
        main_layout.addWidget(self.scroll_area, 1)
        main_layout.addWidget(self.plot_container, 3)
        self.setWindowTitle("White Goods BLDC Simulator - v2.6 (Stop Fix)")
        self.setGeometry(100, 100, 1280, 800)

    def create_control_panel(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(320) 
        
        scroll_content = QWidget()
        self.layout_ctrl = QVBoxLayout(scroll_content)
        self.layout_ctrl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(scroll_content)

        # 1. MAIN CONTROL
        self.btn_toggle = QPushButton("START SIMULATION", self)
        self.btn_toggle.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.btn_toggle.clicked.connect(self.toggle_simulation)
        self.layout_ctrl.addWidget(self.btn_toggle)
        
        self.chk_realtime = QCheckBox("Lock to Real-Time", self)
        self.chk_realtime.toggled.connect(self.on_realtime_toggled)
        self.layout_ctrl.addWidget(self.chk_realtime)
        
        self.lbl_status = QLabel("Status: Idle", self)
        self.lbl_status.setStyleSheet("color: gray; font-style: italic; margin-bottom: 10px;")
        self.layout_ctrl.addWidget(self.lbl_status)

        # 2. MOTOR DB
        group_motor = QGroupBox("Appliance Motor (config_motors.json)")
        l_motor = QVBoxLayout()
        self.combo_motor = QComboBox()
        self.combo_motor.addItems(list(self.sim.motor_db.keys()))
        self.combo_motor.currentTextChanged.connect(self.on_motor_changed)
        l_motor.addWidget(self.combo_motor)
        group_motor.setLayout(l_motor)
        self.layout_ctrl.addWidget(group_motor)

        # 3. REFERENCE
        group_ref = QGroupBox("Reference Speed")
        l_ref = QVBoxLayout()
        self.entry_ref = QLineEdit(str(self.sim.target_speed))
        self.btn_ref = QPushButton("Set Speed (rad/s)", self)
        self.btn_ref.clicked.connect(self.apply_reference)
        l_ref.addWidget(self.entry_ref)
        l_ref.addWidget(self.btn_ref)
        group_ref.setLayout(l_ref)
        self.layout_ctrl.addWidget(group_ref)

        # 4. LOAD PROFILE
        group_load = QGroupBox("Load Profile (config_loads.json)")
        l_load = QVBoxLayout()
        
        l_load.addWidget(QLabel("Load Characteristic:"))
        self.combo_load = QComboBox()
        self.combo_load.addItems(list(self.sim.load_db.keys()))
        self.combo_load.currentTextChanged.connect(self.on_load_profile_changed)
        l_load.addWidget(self.combo_load)
        
        self.lbl_load_val = QLabel("Scale / Torque: 0.00")
        l_load.addWidget(self.lbl_load_val)
        
        self.slider_load = QSlider(Qt.Orientation.Horizontal)
        self.slider_load.setMinimum(0)
        self.slider_load.setMaximum(1000) 
        self.slider_load.setValue(0)
        self.slider_load.valueChanged.connect(self.on_load_value_changed)
        l_load.addWidget(self.slider_load)
        
        group_load.setLayout(l_load)
        self.layout_ctrl.addWidget(group_load)

        # 5. TUNING
        group_pid = QGroupBox("Inverter Tuning")
        l_pid = QVBoxLayout()
        l_pid.addWidget(QLabel("Algorithm:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(['6-Step (Trapezoidal)', 'FOC (Sinusoidal)'])
        l_pid.addWidget(self.combo_mode)
        
        self.stack_pid = QStackedWidget()
        
        self.page_6step = QWidget()
        form_6step = QFormLayout(self.page_6step)
        self.in_kp_6step = QLineEdit(str(self.sim.pid_gains_6step['Kp']))
        self.in_ki_6step = QLineEdit(str(self.sim.pid_gains_6step['Ki']))
        self.in_kd_6step = QLineEdit(str(self.sim.pid_gains_6step['Kd']))
        form_6step.addRow("Kp:", self.in_kp_6step)
        form_6step.addRow("Ki:", self.in_ki_6step)
        form_6step.addRow("Kd:", self.in_kd_6step)
        
        self.page_foc = QWidget()
        form_foc = QFormLayout(self.page_foc)
        self.in_kp_foc = QLineEdit(str(self.sim.pid_gains_foc['Kp']))
        self.in_ki_foc = QLineEdit(str(self.sim.pid_gains_foc['Ki']))
        self.in_kd_foc = QLineEdit(str(self.sim.pid_gains_foc['Kd']))
        form_foc.addRow("Kp:", self.in_kp_foc)
        form_foc.addRow("Ki:", self.in_ki_foc)
        form_foc.addRow("Kd:", self.in_kd_foc)
        
        self.stack_pid.addWidget(self.page_6step)
        self.stack_pid.addWidget(self.page_foc)
        l_pid.addWidget(self.stack_pid)
        self.combo_mode.currentIndexChanged.connect(self.stack_pid.setCurrentIndex)
        group_pid.setLayout(l_pid)
        self.layout_ctrl.addWidget(group_pid)

        # 6. SIM OPTIONS
        group_sim = QGroupBox("Simulation Physics")
        form_sim = QFormLayout()
        self.in_dt = QLineEdit(str(self.sim.dt))
        self.in_batch = QLineEdit(str(self.batch_size))
        self.in_ramp = QLineEdit(str(self.sim.voltage_ramp_rate))
        form_sim.addRow("DT (s):", self.in_dt)
        form_sim.addRow("Steps/Frame:", self.in_batch)
        form_sim.addRow("V Ramp (V/s):", self.in_ramp)
        group_sim.setLayout(form_sim)
        self.layout_ctrl.addWidget(group_sim)

        # 7. TOOLS
        group_vis = QGroupBox("Tools")
        l_vis = QVBoxLayout()
        
        self.chk_rolling = QCheckBox("Rolling Chart", self)
        l_vis.addWidget(self.chk_rolling)
        
        form_tools = QFormLayout()
        self.in_window = QLineEdit("5.0")
        form_tools.addRow("Window (s):", self.in_window)
        self.in_duration = QLineEdit("1.0")
        form_tools.addRow("Batch Time (s):", self.in_duration)
        l_vis.addLayout(form_tools)
        
        # Loading Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #007bff; }")
        l_vis.addWidget(self.progress_bar)
        
        self.btn_batch = QPushButton("Run Batch", self)
        self.btn_batch.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold;")
        self.btn_batch.clicked.connect(self.run_batch)
        l_vis.addWidget(self.btn_batch)
        
        group_vis.setLayout(l_vis)
        self.layout_ctrl.addWidget(group_vis)

        # APPLY
        self.btn_apply = QPushButton("APPLY SETTINGS & RESET", self)
        self.btn_apply.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; padding: 8px; margin-top: 5px;")
        self.btn_apply.clicked.connect(self.apply_settings_restart)
        self.layout_ctrl.addWidget(self.btn_apply)

    def create_plot_panel(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_container = QWidget()
        layout = QVBoxLayout(self.plot_container)
        
        self.plot_speed = pg.PlotWidget(title="Rotor Speed (ω)")
        self.plot_speed.addLegend()
        self.plot_speed.showGrid(x=True, y=True)
        self.curve_speed = self.plot_speed.plot(pen=pg.mkPen('#007bff', width=2), name='Real')
        self.curve_ref = self.plot_speed.plot(pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2), name='Ref')
        layout.addWidget(self.plot_speed)

        self.plot_voltage = pg.PlotWidget(title="Phase Voltages (UVW)")
        self.plot_voltage.showGrid(x=True, y=True)
        self.curve_Va = self.plot_voltage.plot(pen=pg.mkPen('r', width=1))
        self.curve_Vb = self.plot_voltage.plot(pen=pg.mkPen('g', width=1))
        self.curve_Vc = self.plot_voltage.plot(pen=pg.mkPen('b', width=1))
        layout.addWidget(self.plot_voltage)
        
        self.plot_current = pg.PlotWidget(title="Phase Currents (Iabc)")
        self.plot_current.showGrid(x=True, y=True)
        self.curve_Ia = self.plot_current.plot(pen=pg.mkPen('r', width=1))
        self.curve_Ib = self.plot_current.plot(pen=pg.mkPen('g', width=1))
        self.curve_Ic = self.plot_current.plot(pen=pg.mkPen('b', width=1))
        self.plot_current.setYRange(-10, 10)
        layout.addWidget(self.plot_current)
        
        self.plot_voltage.setXLink(self.plot_speed)
        self.plot_current.setXLink(self.plot_speed)

    # --- HANDLERS ---
    def on_motor_changed(self, motor_name):
        self.sim.load_motor_from_db(motor_name)
        self.lbl_status.setText(f"Loaded: {motor_name}")
        self.update_plots(force_clear=True)

    def on_load_profile_changed(self, profile_name):
        self.sim.set_load_profile(profile_name)
        load_type = self.sim.active_load_profile.get("type", "none")
        if load_type == 'none':
            self.slider_load.setEnabled(False)
            self.lbl_load_val.setText("Scale / Torque: 0.00 (Free Spin)")
        else:
            self.slider_load.setEnabled(True)
            val = self.slider_load.value() / 100.0
            self.lbl_load_val.setText(f"Scale / Torque: {val:.2f}")

    def on_load_value_changed(self, val):
        torque = val / 100.0 
        self.sim.load_torque_mag = torque
        self.lbl_load_val.setText(f"Scale / Torque: {torque:.2f}")

    def apply_reference(self):
        try:
            val = float(self.entry_ref.text())
            self.sim.target_speed = val
        except ValueError: pass

    def on_realtime_toggled(self, checked):
        self.is_realtime_locked = checked

    def apply_settings_restart(self):
        try:
            kp_6 = float(self.in_kp_6step.text())
            ki_6 = float(self.in_ki_6step.text())
            kd_6 = float(self.in_kd_6step.text())
            kp_f = float(self.in_kp_foc.text())
            ki_f = float(self.in_ki_foc.text())
            kd_f = float(self.in_kd_foc.text())
            
            dt = float(self.in_dt.text())
            batch = int(self.in_batch.text())
            
            # Limit batch size to prevent freezing
            #CHECK THE INTERFERENCE WITH BATCH PROCESSING AND WINDOW RESIZING (MIGHT BE CAUSING ISSUES) ########################################################################################################################################################################################################################################################################
            # if batch > 200: 
            #     batch = 200
            #     self.in_batch.setText("200")
            
            ramp = float(self.in_ramp.text())
            
            self.sim.dt = dt
            self.batch_size = batch
            self.sim.voltage_ramp_rate = ramp
            
            self.sim.pid_vel_6step.Kp = kp_6
            self.sim.pid_vel_6step.Ki = ki_6
            self.sim.pid_vel_6step.Kd = kd_6
            
            self.sim.controller_foc.atualizar_ganhos_velocidade(kp_f, ki_f, kd_f)
            
            self.sim.pid_vel_6step.reset()
            self.sim.controller_foc.reset()
            self.sim.prev_voltage_cmd = 0.0
            
            mode_txt = self.combo_mode.currentText()
            if '6-Step' in mode_txt:
                self.sim.control_mode = '6-Step'
                self.sim.motor.set_bemf_shape('trapezoidal')
            else:
                self.sim.control_mode = 'FOC'
                self.sim.motor.set_bemf_shape('sinusoidal')
                
            self.sim.motor.estado = np.zeros(4)
            self.sim.motor.estado[3] = 0.001
            self.sim.time = 0.0
            self.sim.reset_history()
            self.update_plots(force_clear=True)
            
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid numerical input.")

    def toggle_simulation(self):
        self.is_running = not self.is_running
        
        if self.is_running:
            self.btn_toggle.setText("STOP")
            self.btn_toggle.setStyleSheet("background-color: #dc3545; color: white;")
            self.start_real_time = time.time()
            self.start_sim_time = self.sim.time
            self.timer.start(1)
        else:
            # STOP LOGIC
            self.timer.stop() # Kill timer
            self.btn_toggle.setText("START SIMULATION")
            self.btn_toggle.setStyleSheet("background-color: #28a745; color: white;")
            self.lbl_status.setText("Stopped")

    def simulation_loop(self):
        # 1. Immediate exit check
        if not self.is_running: 
            return

        try:
            for i in range(self.batch_size):
                if not self.sim.run_step():
                    self.toggle_simulation()
                    return
                
                # 2. Check for stop signal INSIDE the loop
                # This makes the Stop button responsive during calculation
                if i % 5 == 0:
                    QApplication.processEvents()
                    if not self.is_running:
                        return

        except Exception as e:
            print(f"Sim Error: {e}")
            self.toggle_simulation()
            return
            
        self.update_plots()
        
        # 3. Delay Logic
        delay_ms = 1
        status = "Fast (Uncapped)"
        if self.is_realtime_locked:
            sim_elapsed = self.sim.time - self.start_sim_time
            real_elapsed = time.time() - self.start_real_time
            diff = sim_elapsed - real_elapsed
            if diff > 0.01:
                delay_ms = int(diff * 1000)
                status = "Synced"
            elif diff < -0.1:
                status = "Lagging"
        
        self.lbl_status.setText(f"Status: {status}")
        
        # 4. CONDITIONAL RESTART
        # Only restart the timer if the user hasn't clicked STOP during this loop
        if self.is_running:
            self.timer.start(max(1, delay_ms))

    def run_batch(self):
        if self.is_running: return
        
        try:
            duration_s = float(self.in_duration.text())
            if duration_s <= 0: raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Batch Duration")
            return

        # --- FIX: DYNAMIC MEMORY RESIZING ---
        # Calculate how many points are needed for this specific duration
        # Add 2000 points margin to be safe
        needed_points = int(duration_s / self.sim.dt) + 2000
        
        # Override the simulator's limit BEFORE resetting
        self.sim.max_hist_points = needed_points
        print(f"Batch Memory Resized to: {needed_points} points")

        self.apply_settings_restart()
        
        # --- UI SETUP ---
        self.progress_bar.setValue(0)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setText("Calculating...")
        QApplication.processEvents()
        
        total_steps = int(duration_s / self.sim.dt)
        steps_per_update = max(1, total_steps // 100)
        current_step = 0
        
        try:
            while self.sim.time < duration_s:
                if not self.sim.run_step(): break
                
                current_step += 1
                if current_step % steps_per_update == 0:
                    percent = int((current_step / total_steps) * 100)
                    self.progress_bar.setValue(percent)
                    QApplication.processEvents()
        except Exception as e:
            print(f"Batch Error: {e}")
        
        self.progress_bar.setValue(100)
        self.update_plots()
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText("Run Batch")

    def update_plots(self, force_clear=False):
        if force_clear:
            self.curve_speed.setData([], []); self.curve_ref.setData([], [])
            self.curve_Va.setData([], []); self.curve_Vb.setData([], []); self.curve_Vc.setData([], [])
            self.curve_Ia.setData([], []); self.curve_Ib.setData([], []); self.curve_Ic.setData([], [])
            return

        if len(self.sim.hist_time) == 0: return

        # 1. Converte tempo para array numpy (Otimização)
        t = np.array(self.sim.hist_time)
        
        # 2. Define janela de visualização (Padrão: tudo)
        t_start = 0.0
        idx = 0
        
        # Só calcula janela se a opção estiver marcada E a simulação estiver rodando interativamente
        if self.chk_rolling.isChecked() and self.is_running:
            try:
                win_size = float(self.in_window.text())
            except ValueError:
                win_size = 5.0
            t_start = max(0.0, t[-1] - win_size)
            idx = np.searchsorted(t, t_start)

        # 3. Define a função auxiliar (AGORA FORA DO IF PARA EVITAR O ERRO)
        def get_data(deque_obj):
            arr = np.array(deque_obj)
            return arr[idx:]

        # 4. Atualiza curvas usando a função auxiliar
        # Dados fatiados (Slicing)
        view_speed = get_data(self.sim.hist_speed)
        view_ref = get_data(self.sim.hist_ref)

        self.curve_speed.setData(t[idx:], view_speed)
        self.curve_ref.setData(t[idx:], view_ref)
        self.curve_Va.setData(t[idx:], get_data(self.sim.hist_Va))
        self.curve_Vb.setData(t[idx:], get_data(self.sim.hist_Vb))
        self.curve_Vc.setData(t[idx:], get_data(self.sim.hist_Vc))
        self.curve_Ia.setData(t[idx:], get_data(self.sim.hist_Ia))
        self.curve_Ib.setData(t[idx:], get_data(self.sim.hist_Ib))
        self.curve_Ic.setData(t[idx:], get_data(self.sim.hist_Ic))
        
        # 5. Trava Eixo Y (Velocidade) em 0
        if len(view_speed) > 0:
            current_max = max(np.max(view_speed), np.max(view_ref))
            top_limit = max(current_max * 1.1, 10.0)
            self.plot_speed.setYRange(0, top_limit, padding=0)
        
        self.plot_speed.setXRange(t_start, t[-1], padding=0.02)
# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    DT = 0.00005 
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    sim = Simulator(dt=DT)
    window = GuiApplication(sim)
    window.show()
    sys.exit(app.exec())