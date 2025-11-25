# simulator.py
import numpy as np
import json
import os
from collections import deque  # <--- IMPORT THIS

# Project Imports
from motor_simulator import MotorBLDC
from controlador import ControladorPID, ComutadorTrapezoidal, Inversor, ControladorFOC

# --- DEFAULT BACKUPS (In case files are missing) ---
DEFAULT_MOTORS = {
    "Generic Motor": {"R": 1.0, "L": 0.01, "M": 0.003, "J": 0.01, "B": 0.001, "Ke": 0.1, "P": 4}
}

DEFAULT_LOADS = {
    "No Load": {"type": "none"}
}

class Simulator:
    """
    @brief      Orchestrates simulation using separated JSON configurations.
    """
    
    def __init__(self, dt):
        self.dt = dt
        self.time = 0.0
        self.target_speed = 100.0
        self.control_mode = '6-Step' 
        
        self.voltage_ramp_rate = 300.0 
        self.prev_voltage_cmd = 0.0  
        
        # Default Gains
        self.pid_gains_6step = {'Kp': 2.5, 'Ki': 0.5, 'Kd': 0.01}
        self.pid_gains_foc = {'Kp': 8.0, 'Ki': 15.0, 'Kd': 0.0}
        self.pid_vel_6step = ControladorPID(2.5, 0.5, 0.01, 0.0, 310.0) 

        # --- LOAD CONFIGURATIONS ---
        # Now loading from two separate files
        self.motor_db = self.load_json_file('config_motors.json', DEFAULT_MOTORS)
        self.load_db = self.load_json_file('config_loads.json', DEFAULT_LOADS)
        
        # Initialize with First Available Options
        first_motor = list(self.motor_db.keys())[0]
        self.load_motor_from_db(first_motor)
        
        first_load = list(self.load_db.keys())[0]
        self.set_load_profile(first_load)
        
        self.load_torque_mag = 0.0 

        # History Buffers
        self.max_hist_points = 10000 
        self.reset_history()

    def load_json_file(self, filename, default_data):
        """
        @brief Generic JSON loader. Returns default_data if file fails.
        """
        if not os.path.exists(filename):
            print(f"Warning: '{filename}' not found. Using built-in defaults.")
            return default_data
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading '{filename}': {e}. Using built-in defaults.")
            return default_data

    def load_motor_from_db(self, preset_name):
        """Loads physical parameters for the motor."""
        if preset_name not in self.motor_db: return

        p = self.motor_db[preset_name]
        self.V_BUS_MAX = 310.0 
        
        # Instantiate Physics
        self.motor = MotorBLDC(p['R'], p['L'], p['M'], p['J'], p['B'], p['Ke'], p['P'])
        self.P = p['P']
        
        # Instantiate Controllers
        self.switcher = ComutadorTrapezoidal(self.P)
        self.inverter_6step = Inversor(self.V_BUS_MAX)
        self.motor.estado[3] = 0.001 
        
        # Instantiate FOC
        Kp_d, Ki_d = 2.0, 50.0
        Kp_q, Ki_q = 2.0, 50.0
        self.controller_foc = ControladorFOC(
            self.pid_gains_foc['Kp'], self.pid_gains_foc['Ki'], self.pid_gains_foc['Kd'], 
            Kp_d, Ki_d, Kp_q, Ki_q, self.V_BUS_MAX, self.P
        )
        
        # Set BEMF shape based on current mode
        if self.control_mode == 'FOC':
            self.motor.set_bemf_shape('sinusoidal')
        else:
            self.motor.set_bemf_shape('trapezoidal')
            
        print(f"Motor Loaded: {preset_name}")

    def set_load_profile(self, profile_name):
        """Sets the active load parameters from the Load DB."""
        if profile_name in self.load_db:
            self.active_load_profile = self.load_db[profile_name]
            print(f"Load Profile Set: {profile_name} ({self.active_load_profile['type']})")
        else:
            self.active_load_profile = {"type": "none"}

    def reset_history(self):
# Use deque with fixed maxlen. It automatically discards old items efficiently.
        self.hist_time = deque(maxlen=self.max_hist_points)
        self.hist_speed = deque(maxlen=self.max_hist_points)
        self.hist_ref = deque(maxlen=self.max_hist_points)
        
        self.hist_Va = deque(maxlen=self.max_hist_points)
        self.hist_Vb = deque(maxlen=self.max_hist_points)
        self.hist_Vc = deque(maxlen=self.max_hist_points)
        
        self.hist_Ia = deque(maxlen=self.max_hist_points)
        self.hist_Ib = deque(maxlen=self.max_hist_points)
        self.hist_Ic = deque(maxlen=self.max_hist_points)

    def calculate_load_torque(self, theta, omega):
        """
        @brief  Calculates dynamic load using parameters from config_loads.json.
        """
        load_type = self.active_load_profile.get("type", "none")
        
        # Global overrides
        if load_type == 'none': return 0.0
        if self.load_torque_mag <= 0.001: return 0.0
            
        # --- LOAD LOGIC ---
        if load_type == 'static':
            return self.load_torque_mag * np.sign(omega) if abs(omega) > 1.0 else 0.0
            
        elif load_type == 'compressor':
            base = self.active_load_profile.get("pulse_base", 0.6)
            amp = self.active_load_profile.get("pulse_amp", 0.4)
            return self.load_torque_mag * (base + amp * np.sin(theta))
            
        elif load_type == 'fan':
            coeff = self.active_load_profile.get("drag_coeff", 0.0001)
            sign = np.sign(omega) if abs(omega) > 0.1 else 0
            return self.load_torque_mag * (omega * omega) * coeff * sign

        elif load_type == 'crusher':
            if abs(omega) > 5.0:
                coeff = self.active_load_profile.get("drag_coeff", 0.00005)
                drag = (omega * omega) * coeff
                
                impact = 0.0
                chance = self.active_load_profile.get("impact_chance", 0.15)
                if np.random.random() > (1.0 - chance): 
                    scale_min = self.active_load_profile.get("impact_scale_min", 0.5)
                    scale_max = self.active_load_profile.get("impact_scale_max", 1.5)
                    impact = self.load_torque_mag * np.random.uniform(scale_min, scale_max)
                
                return (drag + impact) * np.sign(omega)
            return 0.0

        return 0.0

    def run_step(self):
        ia, ib, omega, theta = self.motor.estado
        self.motor.TL = self.calculate_load_torque(theta, omega)
        
        if self.control_mode == '6-Step':
            raw, err = self.pid_vel_6step.calcular_comando_bruto(self.target_speed, omega, self.dt)
            sat = np.clip(raw, self.pid_vel_6step.V_MIN, self.pid_vel_6step.V_MAX)
            self.pid_vel_6step.atualizar_integrador(raw, sat, err, self.dt)
            if abs(err) < 2.0:
                real_v = sat
                self.prev_voltage_cmd = real_v
            else:
                step = self.voltage_ramp_rate * self.dt 
                real_v = np.clip(sat, self.prev_voltage_cmd - step, self.prev_voltage_cmd + step)
                self.prev_voltage_cmd = real_v
            
            cmds = self.switcher.obter_comandos_de_fase(theta)
            Va, Vb, Vc = self.inverter_6step.aplicar_tensao_nas_fases(real_v, cmds)
        else: 
            Va, Vb, Vc = self.controller_foc.calcular_tensao(self.target_speed, omega, ia, ib, theta, self.dt)
            self.prev_voltage_cmd = 0.0 

        self.motor.aplicar_tensoes(Va, Vb, Vc)
        if not self.motor.avancar_tempo(self.dt): return False 

        self.hist_time.append(self.time)
        self.hist_speed.append(omega)
        self.hist_ref.append(self.target_speed)
        
        self.hist_Va.append(Va)
        self.hist_Vb.append(Vb)
        self.hist_Vc.append(Vc)
        
        ic = -ia - ib 
        self.hist_Ia.append(ia)
        self.hist_Ib.append(ib)
        self.hist_Ic.append(ic)
        
        self.time += self.dt
        
        if len(self.hist_time) > self.max_hist_points:
            self.hist_time.pop(0); self.hist_speed.pop(0); self.hist_ref.pop(0)
            self.hist_Va.pop(0); self.hist_Vb.pop(0); self.hist_Vc.pop(0)
            self.hist_Ia.pop(0); self.hist_Ib.pop(0); self.hist_Ic.pop(0)

        return True

        """
        Runs a headless simulation (no history saving) for the specific 
        profile: 0-1s (0rpm), 1-3s (50rpm), 3-5s (150rpm).
        Returns: Total Error Score (Mean Squared Error).
        """
        # 1. Reset State
        self.motor.estado = np.zeros(4)
        self.motor.estado[3] = 0.001
        self.time = 0.0
        self.prev_voltage_cmd = 0.0
        
        # 2. Apply Test Gains
        if self.control_mode == '6-Step':
            self.pid_vel_6step.Kp = kp
            self.pid_vel_6step.Ki = ki
            self.pid_vel_6step.Kd = kd
            self.pid_vel_6step.reset()
        else:
            # For FOC, we update the Velocity PID inside the controller
            self.controller_foc.atualizar_ganhos_velocidade(kp, ki, kd)
            self.controller_foc.reset()

        total_error_sq = 0.0
        steps = int(duration / self.dt)
        
        # 3. Fast Loop
        for _ in range(steps):
            # --- PROFILE GENERATOR ---
            if self.time < 1.0:
                ref = 0.0
            elif self.time < 3.0:
                ref = 50.0
            else:
                ref = 150.0
            
            # --- PHYSICS STEP (Inline logic for speed) ---
            ia, ib, omega, theta = self.motor.estado
            self.motor.TL = self.calculate_load_torque(theta, omega) # Keep current load settings
            
            # Controller Step
            if self.control_mode == '6-Step':
                raw, err = self.pid_vel_6step.calcular_comando_bruto(ref, omega, self.dt)
                sat = np.clip(raw, self.pid_vel_6step.V_MIN, self.pid_vel_6step.V_MAX)
                self.pid_vel_6step.atualizar_integrador(raw, sat, err, self.dt)
                
                # Ramp Logic
                if abs(err) < 2.0: real_v = sat
                else:
                    max_step = self.voltage_ramp_rate * self.dt
                    real_v = np.clip(sat, self.prev_voltage_cmd - max_step, self.prev_voltage_cmd + max_step)
                self.prev_voltage_cmd = real_v
                
                cmds = self.switcher.obter_comandos_de_fase(theta)
                Va, Vb, Vc = self.inverter_6step.aplicar_tensao_nas_fases(real_v, cmds)
            else:
                Va, Vb, Vc = self.controller_foc.calcular_tensao(ref, omega, ia, ib, theta, self.dt)

            self.motor.aplicar_tensoes(Va, Vb, Vc)
            if not self.motor.avancar_tempo(self.dt):
                return 9999999.0 # Penalize crash
            
            self.time += self.dt
            
            # --- COST FUNCTION CALCULATION ---
            # We penalize error, but we wait until t=1.2s to start counting
            # to allow for some initial settlement if needed.
            # Or we calculate everything to ensure fast rise time.
            error = ref - omega
            total_error_sq += (error * error)

        return total_error_sq