# fips-wood-id-model-luke-test
Takes Python model used to identify wood and converts it to a form useable in a mobile app

## Setup

1. Install [Python 3](https://www.python.org/downloads/) (if you don't have it already)
2. Clone this repo: `git clone git@github.com:jedld/fips-wood-id-model.git`
3. Install the required packages by running `pip install -r requirements.txt` from inside the project directory


### How to generate a model for mobile phones

You can use the `generate_mobile_model.py` script to convert a PyTorch model to a format suitable for mobile applications. The script allows you to specify various parameters such as the model file, class labels, sample input, model name, and description via command-line options.

#### Command-line Options

- `-m`, `--model`: Path to the model weights file (default: `wts.pth`)
- `-c`, `--class_labels`: Path to the class labels file (default: `class_labels.txt`)
- `-s`, `--sample_input`: Path to the sample input image file (default: `../imgdb/000.png`)
- `-n`, `--name`: Name of the model (default: `resnet-18-31-class`)
- `-d`, `--description`: Description of the model (default: `31 class resnet50 Wood Model V5`)
- `-t`, `--type`: Type of the model architecture (default: `resnet18`)
- `-o`, `--output`: Output file name for the generated model archive (default: `model.zip`)
- `-p`, `--path`: Path to the directory containing the input files (default: `.`)

#### Examples

1. Generate a mobile model with default settings:
    ```sh
    python generate_mobile_model.py
    ```

2. Generate a mobile model with a specific model file and class labels:
    ```sh
    python generate_mobile_model.py -m my_model.pth -c my_class_labels.txt
    ```

3. Generate a mobile model with a custom name and description:
    ```sh
    python generate_mobile_model.py -n "custom-model" -d "Custom Model Description"
    ```

4. Generate a mobile model and specify the output file name:
    ```sh
    python generate_mobile_model.py -o custom_model.zip
    ```

5. Generate a mobile model with a specific sample input image:
    ```sh
    python generate_mobile_model.py -s ../imgdb/sample_image.png
    ```

6. Generate a mobile model with a specific model architecture:
    ```sh
    python generate_mobile_model.py -t resnet50
    ```

7. Generate a mobile model with all custom parameters:
    ```sh
    python generate_mobile_model.py -m my_model.pth -c my_class_labels.txt -s ../imgdb/sample_image.png -n "custom-model" -d "Custom Model Description" -t resnet50 -o custom_model.zip -p /path/to/files
    ```

This script will generate a zip file containing the optimized model and related metadata, which can be used in mobile applications.

## Transfering the zip file to the device (Android):

Preqrequisites:

- The PhilWoodID or varients thereof is installed in the Android Device.

### Android

#### 1. Transfering the Zip file via Email or 3rd party tools:

- Send the zip file to the email address of the user.
- Use a 3rd party tool like [Dropbox](https://www.dropbox.com/) to transfer the zip file to the device.
- Make sure to save the Zip file in a location that can be accessed by other apps on the device. For example, if you are using Dropbox, you should save it under the "Dropbox" folder. Or if via Gmail, you should save it under the "Google Drive" folder.
- If the device is connected via USB, you may also be able to transfer the zip file directly from the PC to the device.

Note: Do NOT extract the zip file. The zip file contains all the files required for running the model on the device and the mobile app will be the one that extracts them automatically.

#### 2. Loading the Zip file using the PhilWoodID app:

- Open the PhilWoodID app on your Android Device.
- Once on the PhilWoodID main screen, click on the tertiary button (3 dots) located at the top right corner of the screen.
- Tap on the "Updates...". This will open a new screen with the option to "Install a Model". Click on this and a new screen will show up to select the zip file from where you saved.
- After selecting the zip file, the app will load and extract the model immediately in the background, however it will not be active.
- To activate the model, click on the "Activate" button associated to the model you just loaded. Note that your new model should show up in the list alongside other previously installed models.


## IOS

#### Transfering the zip file to the device (iOS):

Various methods are available for transferring the zip file to the device. The following are some of them:

- 1. Send the zip file via email or 3rd party tools like [Dropbox](https://www.dropbox.com/) and [Google Drive](https://drive.google.com/).
- 2. Use AirDrop to transfer the zip file from your PC to the device.
- 3. Connect the device via USB, then use Finder or iTunes to transfer the zip file directly to the device.

Depending on how you transfered the zip file to the device, please note the location on where it was saved. Do not extract the zip file.

#### 2. Loading the Zip file using the PhilWoodID app:

Preqrequisites:
- The PhilWoodID or varients thereof is installed in the iOS Device.


1. Open the PhilWoodID app on your iOS Device.
2. Once on the PhilWoodID main screen, click on the Settings (Cog Wheel Icon) button at the top right of the screen.
3. The Settings Screen will show up but navigate to the bottom and click on the "Update Model" button. This will open the IOS screen to select the zip file from where you transferred it earlier.
4. Select the zip file that was transferred to your iOS Device.
5. Wait for the zip file to be processed by the app and loaded onto the device. Once done, the PhilWoodID app will show a message indicating that the model has been updated successfully. The selected model will be active on the device.




