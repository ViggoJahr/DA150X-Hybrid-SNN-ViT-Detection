import json
import sys
import os

def create_log_from_json(target_dir):
    json_path = os.path.join(target_dir, "multiclass-adamw.json")
    log_path = os.path.join(target_dir, "training.log")

    if not os.path.exists(json_path):
        print(f"Hittade ingen JSON-fil i {target_dir}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    total_epochs = len(data["train_loss"])
    
    # Kolla vad lr-listan heter i JSON-filen (skiljer sig mellan baseline och ViT)
    if "lr_schedule" in data:
        lr_key = "lr_schedule"
    elif "lr" in data and isinstance(data["lr"], list):
        lr_key = "lr"
    else:
        lr_key = None # Fallback om listan inte finns
    
    # Öppna (eller skapa) training.log för att skriva till den
    with open(log_path, 'w') as log_file:
        header = f"Återskapad träningslogg från JSON (Modell: {data.get('model', 'SNN Baseline')})\n"
        header += "-" * 95 + "\n"
        header += f"{'Epoch':<6} | {'Train':<8} | {'Val':<8} | {'Person':<7} | {'Car':<7} | {'Bus':<7} | {'Truck':<7} | {'LR'}\n"
        header += "-" * 95 + "\n"
        
        print(header, end="")
        log_file.write(header)

        for i in range(total_epochs):
            t_loss = data["train_loss"][i][0]
            v_loss = data["validation_loss"][i][0]
            v_person = data["validation_loss"][i][1]
            v_car = data["validation_loss"][i][2]
            v_bus = data["validation_loss"][i][3]
            v_truck = data["validation_loss"][i][4]
            
            # Hantera learning rate formateringen smartare
            if lr_key:
                lr_val = data[lr_key][i]
                if isinstance(lr_val, list):
                    if len(lr_val) >= 2:
                        lr_str = f"[{lr_val[0]:.2e}, {lr_val[1]:.2e}]" # Phase 2 (två LRs)
                    else:
                        lr_str = f"{lr_val[0]:.2e}" # Phase 1 (lista med ett värde)
                else:
                    lr_str = f"{lr_val:.2e}" # Gammal baseline (ensam float)
            else:
                lr_str = "N/A"

            row = f"{i+1:<6} | {t_loss:<8.3f} | {v_loss:<8.3f} | {v_person:<7.3f} | {v_car:<7.3f} | {v_bus:<7.3f} | {v_truck:<7.3f} | {lr_str}\n"
            
            # Skriv till både terminal och fil
            print(row, end="")
            log_file.write(row)

    print("-" * 95)
    print(f"\nKlart! Loggen är nu sparad till: {log_path}")

# Hämta mapp från argument i terminalen, annars använd standard
if len(sys.argv) > 1:
    target_directory = sys.argv[1]
else:
    print("Ange vilken mapp du vill återskapa loggen för.")
    sys.exit(1)

create_log_from_json(target_directory)
