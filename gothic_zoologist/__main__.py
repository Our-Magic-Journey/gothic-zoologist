from model import CNN
import train
import dataset
import readline
import shlex
import atexit
import os
from jax import numpy as jnp

def title():
    print(
"""
   ____       _   _     _                 
  / ___| ___ | |_| |__ (_) ___            
 | |  _ / _ \| __| '_ \| |/ __|           
 | |_| | (_) | |_| | | | | (__            
  \____|\___/ \__|_| |_|_|\___| _     _   
 |__  /___   ___ | | ___   __ _(_)___| |_ 
   / // _ \ / _ \| |/ _ \ / _` | / __| __|
  / /| (_) | (_) | | (_) | (_| | \__ \ |_ 
 /____\___/ \___/|_|\___/ \__, |_|___/\__|
                          |___/           
-----------------------------------------------------------
""")


def print_help():
    print(
"""Available commands:
    help                  - Show this help message.
    exit                  - Exit the program.
    load                  - Load a trained model checkpoint.
    train                 - Train a new model.
    verify_training       - Verify the training using the test dataset.
    identify <image_path> - Identify the content of an image.
                            <image_path> can be a file or a directory.
""")


def main():
    print("loading checkpoint")
    checkpoint = train.load_checkpoint()
    params = None
    model: CNN = None
    categories = []

    if checkpoint is None:
        print("checkpoint not found, pleas train model.")
    else:
        categories, params = train.load_checkpoint()
        model = CNN(outputs=len(categories))


    history_file = '.command_history'

    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, history_file)

    title()
    print_help()

    while True:
        cmd, *args = shlex.split(input('> '))
        os.system('cls' if os.name=='nt' else 'clear')
        title()
        print(f"> {cmd} {' '.join(args)}")

        if cmd == 'exit':
            break

        elif cmd == 'help':
            print_help()

        elif cmd == 'load':
            checkpoint = train.load_checkpoint()

            if checkpoint is None:
                print("model not found, please train model.")
                continue

            categories, params = checkpoint
            model = CNN(outputs=len(categories))


        elif cmd =='train':
            train.train_model()
            categories, params = train.load_checkpoint()
            model = CNN(outputs=len(categories))


        elif cmd =='verify_training':
            if model is None:
                print("model not found, please train model.")
                continue

            test_data, _, _ = dataset.load_gothic_dataset()
            train.verify_training(model, params, test_data)


        elif cmd =='identify':
            if model is None:
                print("model not found, please train model.")
                continue

            if len(args) != 1:
                print("Usage: identify <image path>")
                continue
            try:
                image = dataset.load_image(args[0])
                result = model.apply(params, image)
                result = jnp.exp(result)

                for i, r in enumerate(result):
                    print(f'{categories[i]}: {r:.2f}%')


            except FileNotFoundError:
                print(f"File: {args[0]} not found.")
                continue
            except IsADirectoryError:
                print(f"Directory: {args[0]}: ")
                for file in sorted(os.listdir(args[0])):
                        print('         ' + file)
                continue

        else:
            print('Unknown command: {}'.format(cmd))


if __name__ == '__main__':
    main()
