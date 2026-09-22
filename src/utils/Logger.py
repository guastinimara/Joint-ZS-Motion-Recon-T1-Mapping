"""Text log file for a reconstruction run, plus optimizer callbacks that write to it."""
from datetime import datetime
import os

from data.loading_data import get_prefix


class Logger:
    """Writes a log file ``{save_folder}/{prefix}/log_{prefix}_{today}_{HHMMSS}.txt`` with a header
    listing all parameters, and collects CG / L-BFGS / Adam iteration messages."""
    def __init__(self, 
            patient_name, 
            today,
            params_reconstruction, 
            params_motion_net, 
            run_settings,
            save_folder):
        """
        Args:
            patient_name: full file name of the subject (the prefix is used for folder and file name).
            today: date string used in the file name.
            params_reconstruction, params_motion_net, run_settings: config dicts written in the header.
            save_folder: root folder for the logs.
        """
        self.log_file = self.log_init(patient_name, 
                                      today, 
                                      params_reconstruction, 
                                      params_motion_net, 
                                      run_settings,
                                      save_folder)
        
        self.lbfgs_messages = []
        self.cg_messages = []
        self.adam_messages = []

    
    def log(self, message, print_log=False):
        """Append a timestamped message to the log file."""
        self.log_to_file(self.log_file, message)
        if print_log:
            print(message)

    def log_init(self,
             patient_name, 
             today,
             params_reconstruction, 
             params_motion_net, 
             run_settings,
             save_folder
             ):
        """Create the log file with the parameter header and return its path."""
    
        pat =  get_prefix(patient_name)
        log_filename = f"log_{pat}_{today}_{datetime.now().strftime('%H%M%S')}.txt"
        log_filepath = os.path.join(save_folder, pat, log_filename)
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        
        # Write header information
        with open(log_filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RECONSTRUCTION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Patient: {pat}\n")
            f.write(f"Full filename: {patient_name}\n\n")
            f.write("RECONSTRUCTION PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for key, value in params_reconstruction.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-" * 40 + "\n\n")
            f.write("MOTION PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for key, value in params_motion_net.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-" * 40 + "\n\n")
            f.write("RUN SETTINGS:\n")
            f.write("-" * 40 + "\n")
            for key, value in run_settings.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
        return log_filepath
    
    # Clear the collected optimizer messages
    def empty_lbfgs_messages(self):
        self.lbfgs_messages = []
        
    def empty_cg_messages(self):
        self.cg_messages = []
        
    def empty_adam_messages(self):
        self.adam_messages = []

    # Optimizer callbacks (pass to mrpro optimizers): format, store and log the iteration message
    def callback_lbfgs_logged(self, optimizer_status):
        message = self.callback_lbfgs(optimizer_status)
        self.lbfgs_messages.append(message)
        self.log(message)
        
    def callback_adam_logged(self, optimizer_status):
        message = self.callback_adam(optimizer_status)
        self.adam_messages.append(message)
        self.log(message)

    def callback_cg_logged(self, optimizer_status):
        message = self.callback_cg(optimizer_status)
        self.cg_messages.append(message)
        self.log(message)

    @staticmethod
    def log_to_file(log_file, message):
        """Helper function to write to log file with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    # Format and print one iteration message
    @staticmethod
    def callback_lbfgs(optimizer_status):
        message = f"  LBFGS Iter: {optimizer_status['iteration_number']}, Residual l2: {optimizer_status['residual'].abs().square().sum()}"
        print(message)
        return message
    
    @staticmethod
    def callback_adam(optimizer_status):
        message = f"  ADAM Iter: {optimizer_status['iteration_number']}"
        print(message)
        return message

    @staticmethod
    def callback_cg(optimizer_status):
        message = f"  CG Iter: {optimizer_status['iteration_number']}, Residual l2: {optimizer_status['residual'][0].abs().square().sum()}"
        print(message)
        return message
