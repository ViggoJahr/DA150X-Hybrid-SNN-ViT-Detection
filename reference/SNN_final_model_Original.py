import argparse
import datetime
import os
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loading import get_data
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import json

from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
sequence_length, overlap, batch_size = 60, 25, 24
num_inputs = 256 * 256
num_outputs = 64 * 64  # 4096
# tau_list = [120, 140, 160, 180, 200] #For trying different taus
nr_par_last_layer_list = [500]
w_decay = 1e-4  # Changed from 1e-4 on v1
lr = 1e-4
print_image = (
    False  # Set true to save current image for each loop, used to see progress during training
)
loss_function = nn.MSELoss()  # nn.HuberLoss(delta=1.345)
tau_mem = 180


def save_data(
    model, output_dir, train_loss, validation_loss, val_accuracy, tau, epoch, lr_list, save_model
):
    file_name = os.path.join(output_dir, f"multiclass-adamw")

    if save_model:
        torch.save(model.state_dict(), f"{file_name}-{epoch}-{validation_loss[-1][0]}.pth")

    data = {
        "Epoch": epoch,
        "Tau": tau,
        "w_decay": w_decay,
        "lr": lr,
        "layer_size": nr_par_last_layer_list,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "lr": lr_list,
        "validation_accuracy": val_accuracy,
    }

    with open(f"{file_name}.json", "w") as f:
        json.dump(data, f)

    if save_model:
        print(
            f"Model has been saved to \x1b[1m{file_name}-{epoch}-{validation_loss[-1][0]}.pth\x1b[22m"
        )

    print(f"Training stats saved to \x1b[1m{file_name}.json\x1b[22m")


def save_current_result(output_frame, target_frame, frames, step, classId):
    maximum_value = torch.max(output_frame[0][1]).item()
    fig, axes = plt.subplots(1, 3, figsize=(14, 10))
    axes[0].imshow(frames[0][step].cpu().detach().numpy(), cmap="gray")
    axes[1].imshow(target_frame[0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=0.03)
    axes[2].imshow(output_frame[0].cpu().detach().numpy(), cmap="gray", vmax=maximum_value)

    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    # fig.tight_layout()

    plt.savefig(f"test-class{classId}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def pretty_time(seconds):
    if not seconds:
        return f"0s"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    measures = (
        (hours, "h"),
        (minutes, "m"),
        (seconds, "s"),
    )
    return " ".join([f"{count}{unit}" for (count, unit) in measures if count])


for layer_nr in nr_par_last_layer_list:

    class SNN(nn.Module):
        def __init__(self):
            super(SNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=0)
            self.bn1 = nn.BatchNorm2d(8)
            self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=0)
            self.bn2 = nn.BatchNorm2d(8)
            self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(8)
            self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.lif4 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            self.lif5 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            self.lif6 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            self.lif7 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.fcperson1 = nn.Linear(3200, layer_nr)
            self.fcperson2 = nn.Linear(layer_nr, layer_nr)
            self.lifperson = LILinearCell(layer_nr, 4096)

            self.fccar1 = nn.Linear(3200, layer_nr)
            self.fccar2 = nn.Linear(layer_nr, layer_nr)
            self.lifcar = LILinearCell(layer_nr, 4096)

            self.fcbus1 = nn.Linear(3200, layer_nr)
            self.fcbus2 = nn.Linear(layer_nr, layer_nr)
            self.lifbus = LILinearCell(layer_nr, 4096)

            self.fctruck1 = nn.Linear(3200, layer_nr)
            self.fctruck2 = nn.Linear(layer_nr, layer_nr)
            self.liftruck = LILinearCell(layer_nr, 4096)

            self.maxpool = nn.MaxPool2d(2, 2)
            # self.avgpool = nn.AvgPool2d(2, 2)
            self.dropout1 = nn.Dropout(p=0.4)
            self.dropout2 = nn.Dropout(p=0.5)

        def forward(self, x, mem_states):
            batch_size, C, W, H = x.shape
            x = (x != 0).float()  # Ensure binary data for this case

            (
                mem1,
                mem2,
                mem3,
                mem5_1,
                mem5_2,
                mem6_1,
                mem6_2,
                mem7_1,
                mem7_2,
                mem8_1,
                mem8_2,
            ) = mem_states

            v1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(v1, mem1)

            v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1))))
            # v2 = self.dropout(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(v2, mem2)

            v3 = self.dropout1(self.bn3(self.conv3(spk2)))
            # v3 = self.dropout1(self.maxpool(self.bn3(self.conv3(spk2))))
            spk3, mem3 = self.lif3(v3, mem3)

            spk3_flat = spk3.view(batch_size, -1)

            v5 = self.dropout2(self.fcperson1(spk3_flat))
            spk5_1, mem5_1 = self.lif4(v5, mem5_1)
            v5 = self.dropout2(self.fcperson2(spk5_1))
            spk5_2, mem5_2 = self.lifperson(v5, mem5_2)

            v6 = self.dropout2(self.fccar1(spk3_flat))
            spk6_1, mem6_1 = self.lif5(v6, mem6_1)
            v6 = self.dropout2(self.fccar2(spk6_1))
            spk6_2, mem6_2 = self.lifcar(v6, mem6_2)

            v7 = self.dropout2(self.fcbus1(spk3_flat))
            spk7_1, mem7_1 = self.lif6(v7, mem7_1)
            v7 = self.dropout2(self.fcbus2(spk7_1))
            spk7_2, mem7_2 = self.lifbus(v7, mem7_2)

            v8 = self.dropout2(self.fctruck1(spk3_flat))
            spk8_1, mem8_1 = self.lif7(v8, mem8_1)
            v8 = self.dropout2(self.fctruck2(spk8_1))
            spk8_2, mem8_2 = self.liftruck(v8, mem8_2)

            return (
                spk5_2,
                spk6_2,
                spk7_2,
                spk8_2,
                (
                    mem1,
                    mem2,
                    mem3,
                    mem5_1,
                    mem5_2,
                    mem6_1,
                    mem6_2,
                    mem7_1,
                    mem7_2,
                    mem8_1,
                    mem8_2,
                ),
            )

    def loss_fn(output_frame, target_frame, step, class_id):
        mse_loss = 0
        mse_loss += loss_function(
            output_frame, target_frame * 1000
        )  # Multiplication due to the numbers being too small, should be fixed when creating the data

        # if print_image:
        #     save_current_result(output_frame, target_frame, frames, step, class_id)

        return mse_loss

        # if print_image:
        #     save_current_result(output_frame, target_frame, frames, step)
    data_dirs = [
        "week_32-box_3",
        "week_33-box_2",
        "week_34-box_1",
        "week_35-box_2",
        "week_36-box_3" ]

    # [
    #     "w33-1-08-14",
    #     "w35-1-09-04",
    # ]

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    def start_training(training_data_dir, output_dir, num_epochs):

        cur_time = datetime.datetime.now()
        output_dir = os.path.join(
            output_dir, f"{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNN().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay, eps=1e-8)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=0, factor=0.5)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Print data of the model
        print(f"Total trainable parameters: {trainable_params}")
        trainable_weights = sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad and "weight" in name
        )

        print(f"Total trainable weights: {trainable_weights}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_type = "weights" if "weight" in name else "biases"
                print(f"Layer: {name} | Type: {param_type} | Number of Parameters: {param.numel()}")

        print("===================================================")
        frame_size = (64, 64)

        train_loss_list = []
        val_loss_list = []
        lr_list = []
        val_acc_list = []

        best_val = 999999
        scaler = GradScaler()

        data_files = []

        for curr_dir in data_dirs:

            data_files.extend(
                [
                    os.path.join(os.path.join(training_data_dir, curr_dir), f)
                    for f in os.listdir(os.path.join(training_data_dir, curr_dir))
                    if f.endswith(".pt")
                ]
            )

        random.shuffle(data_files)
        for epoch in range(num_epochs):

            total_train_loss = 0
            person_train_loss = 0
            car_train_loss = 0
            buss_train_loss = 0
            truck_train_loss = 0
            num_train_batches = 0

            model.train()
            start = time.time()

            for i, data_paths in enumerate(chunker(data_files, 4)):
                data = get_data(data_paths)

                if data is not None:
                    train_data, val_data = data

                    # {data_path.removeprefix(training_data_dir)}
                    print(
                        f"\r\x1b[2KEpoch {epoch + 1} | File Chunk \x1b[1m {i}/{len(data_files)//4}\x1b[22m Training. | train loss: \x1b[1m{total_train_loss/(num_train_batches+0.00000001):.3f}\x1b[22m person \x1b[1m{person_train_loss/(num_train_batches+0.00000001):.3f}\x1b[22m | car \x1b[1m{car_train_loss/(num_train_batches+0.00000001):.3f}\x1b[22m | buss \x1b[1m{buss_train_loss/(num_train_batches+0.00000001):.3f}\x1b[22m | truck \x1b[1m{truck_train_loss/(num_train_batches+0.00000001):.3f}\x1b[22m | \x1b[1m{pretty_time(time.time()-start)}\x1b[22m",
                        end="",
                    )
                    train_loader = train_data

                    for i, (frames, targets) in enumerate(train_loader):
                        mem_states = (
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )

                        optimizer.zero_grad()

                        frames, targets = frames.to(device), targets.to(device)

                        loss = 0
                        person_loss = 0
                        car_loss = 0
                        buss_loss = 0
                        truck_loss = 0

                        for step in range(sequence_length):
                            input_frame = frames[:, step].unsqueeze(1)
                            output1, output2, output3, output4, mem_states = model(
                                input_frame, mem_states
                            )
                            # print(output.shape)

                            if step >= overlap:
                                final_output1 = output1.view(batch_size, 64, 64)
                                final_output2 = output2.view(batch_size, 64, 64)
                                final_output3 = output3.view(batch_size, 64, 64)
                                final_output4 = output4.view(batch_size, 64, 64)

                                # print(targets.shape)

                                person_loss += (
                                    0.7
                                    * 2
                                    * loss_fn(final_output1, targets[:, 0, step], step, 0)
                                    / (sequence_length - overlap)
                                )
                                car_loss += (
                                    0.5
                                    * 2
                                    * loss_fn(final_output2, targets[:, 1, step], step, 2)
                                    / (sequence_length - overlap)
                                )
                                buss_loss += (
                                    4
                                    * 6
                                    * loss_fn(final_output3, targets[:, 2, step], step, 5)
                                    / (sequence_length - overlap)
                                )
                                truck_loss += (
                                    4.2
                                    * 6
                                    * loss_fn(final_output4, targets[:, 3, step], step, 7)
                                    / (sequence_length - overlap)
                                )  # only train on the last 50 frames

                        loss = person_loss + car_loss + buss_loss + truck_loss

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)

                        optimizer.step()

                        total_train_loss += loss.item()
                        person_train_loss += person_loss.item()
                        car_train_loss += car_loss.item()
                        buss_train_loss += buss_loss.item()
                        truck_train_loss += truck_loss.item()

                        num_train_batches += 1

            model.eval()
            total_val_loss = 0
            person_val_loss = 0
            car_val_loss = 0
            buss_val_loss = 0
            truck_val_loss = 0
            val_acc = 0
            num_test_batches = 0
            with torch.no_grad():
                for i, data_paths in enumerate(chunker(data_files, 4)):
                    data = get_data(data_paths)

                    if data is not None:
                        train_data, val_data = data

                        # {data_path.removeprefix(training_data_dir)}
                        print(
                            f"\r\x1b[2KEpoch {epoch + 1} | File Chunk \x1b[1m {i}/{len(data_files)//4}\x1b[22m Validation. | val loss: \x1b[1m{total_val_loss/(num_test_batches+0.00000001):.3f}\x1b[22m person \x1b[1m{person_val_loss/(num_test_batches+0.00000001):.3f}\x1b[22m | car \x1b[1m{car_val_loss/(num_test_batches+0.00000001):.3f}\x1b[22m | buss \x1b[1m{buss_val_loss/(num_test_batches+0.00000001):.3f}\x1b[22m | truck \x1b[1m{truck_val_loss/(num_test_batches+0.00000001):.3f}\x1b[22m | \x1b[1m{pretty_time(time.time()-start)}\x1b[22m",
                            end="",
                        )
                        validation_loader = val_data

                        for i, (frames, targets) in enumerate(validation_loader):
                            mem_states = (
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                            )

                            frames, targets = frames.to(device), targets.to(device)
                            loss = 0
                            person_loss = 0
                            car_loss = 0
                            buss_loss = 0
                            truck_loss = 0

                            # batch_acc = torch.zeros((batch_size, 64, 64), device=device)
                            for step in range(sequence_length):
                                input_frame = frames[:, step].unsqueeze(1)
                                output1, output2, output3, output4, mem_states = model(
                                    input_frame, mem_states
                                )

                                final_output1 = output1.view(batch_size, 64, 64)
                                final_output2 = output2.view(batch_size, 64, 64)
                                final_output3 = output3.view(batch_size, 64, 64)
                                final_output4 = output4.view(batch_size, 64, 64)

                                person_loss += (
                                    0.7
                                    * 2
                                    * loss_fn(final_output1, targets[:, 0, step], step, 0)
                                    / (sequence_length - overlap)
                                )
                                car_loss += (
                                    0.5
                                    * 2
                                    * loss_fn(final_output2, targets[:, 1, step], step, 2)
                                    / (sequence_length - overlap)
                                )
                                buss_loss += (
                                    4
                                    * 6
                                    * loss_fn(final_output3, targets[:, 2, step], step, 5)
                                    / (sequence_length - overlap)
                                )
                                truck_loss += (
                                    4.2
                                    * 6
                                    * loss_fn(final_output4, targets[:, 3, step], step, 7)
                                    / (sequence_length - overlap)
                                )

                            loss = (
                                person_loss + car_loss + buss_loss + truck_loss
                            )  # only train on the last 50 frames

                            # batch_acc += (
                            #     torch.abs(final_output1 - targets[:, 0, step])
                            #     + torch.abs(final_output2 - targets[:, 1, step])
                            #     + torch.abs(final_output3 - targets[:, 2, step])
                            #     + torch.abs(final_output4 - targets[:, 3, step])
                            # )

                            # val_acc += torch.mean(batch_acc / len(targets)).item()

                            total_val_loss += loss.item()
                            person_val_loss += person_loss.item()
                            car_val_loss += car_loss.item()
                            buss_val_loss += buss_loss.item()
                            truck_val_loss += truck_loss.item()

                            num_test_batches += 1

            scheduler.step(total_val_loss / num_test_batches)
            del data
            gc.collect()

            epoch_time = time.time() - start
            if epoch == 0:
                print(
                    f"\x1b[0G\x1b[2KEpoch {epoch+1} | train loss: \x1b[1m{total_train_loss/num_train_batches:.3f}\x1b[22m | val loss \x1b[1m{total_val_loss/num_test_batches:.3f}\x1b[22m | person \x1b[1m{person_val_loss/num_test_batches:.3f}\x1b[22m | car \x1b[1m{car_val_loss/num_test_batches:.3f}\x1b[22m | buss \x1b[1m{buss_val_loss/num_test_batches:.3f}\x1b[22m | truck \x1b[1m{truck_val_loss/num_test_batches:.3f}\x1b[22m | finished in \x1b[1m{pretty_time(epoch_time)}\x1b[22m",
                    end="\n",
                )
            else:
                print(
                    f"\x1b[2K\x1b[0GEpoch {epoch+1} | train loss: \x1b[1m{total_train_loss/num_train_batches:.3f}\x1b[22m | val loss \x1b[1m{total_val_loss/num_test_batches:.3f}\x1b[22m | person \x1b[1m{person_val_loss/num_test_batches:.3f}\x1b[22m | car \x1b[1m{car_val_loss/num_test_batches:.3f}\x1b[22m | buss \x1b[1m{buss_val_loss/num_test_batches:.3f}\x1b[22m | truck \x1b[1m{truck_val_loss/num_test_batches:.3f}\x1b[22m | finished in \x1b[1m{pretty_time(epoch_time)}\x1b[22m",
                    end="\n",
                )

            train_loss_list.append(
                [
                    np.round(total_train_loss / num_train_batches, 4),
                    np.round(person_train_loss / num_train_batches, 4),
                    np.round(car_train_loss / num_train_batches, 4),
                    np.round(buss_train_loss / num_train_batches, 4),
                    np.round(truck_train_loss / num_train_batches, 4),
                ]
            )
            val_loss_list.append(
                [
                    np.round(total_val_loss / num_test_batches, 4),
                    np.round(person_val_loss / num_test_batches, 4),
                    np.round(car_val_loss / num_test_batches, 4),
                    np.round(buss_val_loss / num_test_batches, 4),
                    np.round(truck_val_loss / num_test_batches, 4),
                ]
            )
            lr_list.append(
                scheduler.get_last_lr(),
            )
            val_acc_list.append(np.round(val_acc / num_test_batches, 4))

            if total_val_loss / num_test_batches < best_val:
                best_val = total_val_loss / num_test_batches
                # file_name = os.path.join(
                #     output_dir,
                #     f"multiclass-adamw-{epoch}-{np.round(total_val_loss / num_test_batches, 4)}",
                # )

                save_data(
                    model,
                    output_dir,
                    train_loss_list,
                    val_loss_list,
                    val_acc_list,
                    tau_mem,
                    epoch,
                    lr_list,
                    True,
                )
                # torch.save(model.state_dict(), f"{file_name}.pth")
            else:
                save_data(
                    model,
                    output_dir,
                    train_loss_list,
                    val_loss_list,
                    val_acc_list,
                    tau_mem,
                    epoch,
                    lr_list,
                    False,
                )

        save_data(
            model,
            output_dir,
            train_loss_list,
            val_loss_list,
            val_acc_list,
            tau_mem,
            epoch,
            lr_list,
            True,
        )


parser = argparse.ArgumentParser(
    description="Trainer for traffic monitoring SNN model",
    usage="%(prog)s <path/to/training_data/> <path/to/output_dir/> [options]",
)

parser.add_argument("input_dir", help="The path to the directory containing the training data")
parser.add_argument(
    "output_dir", help="The path to the directory where the model checkpoints should be saved"
)
parser.add_argument(
    "--epoch",
    default=1,
    type=int,
    help="The number of epochs to train the model for (default: %(default)s epoch)",
)

parser.add_argument(
    "--gpu",
    type=int,
    default=2,
    help="The CUDA device ID to use (default: %(default)s)",
)

args = parser.parse_args()

# SET CUDA DEVICE HERE
torch.cuda.set_device(args.gpu)

start_training(args.input_dir, args.output_dir, args.epoch)
