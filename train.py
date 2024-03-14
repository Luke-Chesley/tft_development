from model import build_tft, build_time_series_ds
from config import get_config
from predict import make_predictions
import os
import json

def get_file(dir_name):
    path = f"checkpoints/{dir_name}"
    try:
        files = next(os.walk(path))[2]  # Get list of files in the directory
        if files:  # Check if the list is not empty
            return os.path.join(path, files[0])  # Return the first file with full path
        else:
            return "No files found in the directory."
    except StopIteration:
        return "Directory does not exist."


def train_model(config):
    train_dataloader, val_dataloader, _ = build_time_series_ds(config)

    trainer, tft = build_tft(config)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # include this to resume training, comment out to train with new params
        # figure out way to link config and different checkpoints?
        # ckpt_path="tft/checkpoints/6/epoch=4-val_loss=787.53.ckpt",
    )

    del trainer, tft, train_dataloader, val_dataloader

subject = "max_encoder_len"

param_dict = {
        1: 96,
        2: 168,
        3: 240,
        4: 312,
        5: 384,
        6: 456,
        7: 528,
        8: 600,
        9: 744,
        10: 888,
    }


if __name__ == "__main__":
    results_dict = {}
    for i in range(1, 11):

        config = get_config(i)

        config[subject] = param_dict[i]

        print(f"training model: {i}")
        train_model(config)

        tft_file = get_file(i)
        test_start = "2024-01-01 00:00:00"

        ((_, _), (mae, sum_mae, _), _, _, tft, raw_preds) = make_predictions(
            config,
            test_start,
            tft_file,
            3,
        )


        interpretation = tft.interpret_output(raw_preds.output, reduction="sum")

        run_results_dict = {
            "interpretation": interpretation,
            "mae": mae,
            "sum_mae": sum_mae,
            'config':config
        }

        results_dict[i] = run_results_dict

    with open('results.json','w') as f:
        json.dump(results_dict,f)