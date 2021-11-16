import os
import sys
import urllib.request
import zipfile

DRIVE_ROOT_DIR = "/content/gdrive/MyDrive/pysource_object_detection/"
DARKNET_PATH = "/content/darknet"

def is_gpu_enabled():
    # Check if GPU is enabled
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


def download_dnn_model(Model):
    # change makefile to have GPU and OPENCV enabled

    # Download Weights
    if Model == "yolov4-p6":
        urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.conv.289',
                                   os.path.join(DARKNET_PATH, 'yolov4-p6.conv.289'))
    elif Model == "yolov4-tiny":
        urllib.request.urlretrieve(
            'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29',
            os.path.join(DARKNET_PATH, 'yolov4-tiny.conv.29'))
    else:
        urllib.request.urlretrieve(
            'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137',
            'yolov4.conv.137')

# Connect google drive
def connect_google_drive(project_name):
    from google.colab import drive
    drive.mount('/content/gdrive')

    model_dir = os.path.join(DRIVE_ROOT_DIR, project_name)
    model_dir = os.path.join(model_dir, "dnn")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("New project created {}".format(project_name))
        print("You'll find the project on Google Drive, on the folder pysource_object_detection/{} .".format(project_name))
    else:
        print("Project {} already exists. Editing existing project.".format(project_name))
    return model_dir

def unzip_dataset(project_name):
    dataset_path = os.path.join(DRIVE_ROOT_DIR, project_name)
    dataset_path = os.path.join(dataset_path, "dataset.zip")
    output_path = "/content/darknet/data/obj"
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall("/content/darknet/data/obj")
