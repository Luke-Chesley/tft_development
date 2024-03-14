from config import get_config
from train import train_model
from predict import make_predictions
import os


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

def run_experiment():
    subject = 'max_encoder_len'

    param_dict = {1:96,
                2:168,
                3:240,
                4:312,
                5:384,
                6:456,
                7:528,
                8:600,
                9:744,
                10:888
                }


    results_dict = {}

    for folder in range(1,11):
        
        config = get_config(folder)

        run_var = param_dict[folder]

        # adjust config for each
        config[subject] = run_var


        print(config[subject])

            
        # train here
        train_model(config)

        print(f'done training model: {folder}')

        tft_file = get_file(folder)
        test_start = "2024-01-01 00:00:00"

        ((_, _), (mae, sum_mae, _), _, _, tft, raw_preds) = (
            make_predictions(
                config,
                test_start,
                tft_file,
                3,
            )
        )

        # evaluate here?

        interpretation = tft.interpret_output(raw_preds.output, reduction="sum")

        run_results_dict = {'interpretation':interpretation,
                        'mae':mae,
                        'sum_mae':sum_mae}

        results_dict[folder] = run_results_dict

    return results_dict


