# This files prepare a custom Object detector for YOLO v4
#
import glob
import errno
import os
import argparse
import re
import zipfile
from pathlib import Path

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--n_classes", type=int, required=True, help="number of classes")
ap.add_argument("-b", "--backup", type=str, required=False, help="backup path")
ap.add_argument("-s", "--subdivisions", type=str, required=False, default="64", choices=["16", "32", "64"],
                help="Subdivisions value (16, 32, 64)")
ap.add_argument("-pn", "--projectname", type=str, required=True, help="Project Name")
ap.add_argument("-m", "--model", type=str, required=False, choices=["yolov4", "yolov4-tiny", "yolov4-p6"],
                default="yolov4", help="YOLO Model")
ap.add_argument("-is", "--imagesize", type=str, required=False,
                help="Image size")
ap.add_argument("-v", "--validation", type=int, required=False, default=10,
                help="Yolo version")
# ap.add_argument("-r", "--radius", type=int, default=3, help="in")
args = vars(ap.parse_args())


class CustomYOLODetector:
    def __init__(self):
        # Files location
        self.cfg_paths = {"yolov4": "cfg/yolov4-custom.cfg",
                     "yolov4-tiny": "cfg/yolov4-tiny-custom.cfg",
                     "yolov4-p6": "cfg/yolov4-p6.cfg"}

        # Argument import
        if args["backup"]:
            self.backup_folder_path = args["backup"]
        if args["subdivisions"]:
            self.subdivisions = args["subdivisions"]
        if args["projectname"]:
            self.projectname = args["projectname"]
        if args["model"]:
            self.model = args["model"]

        self.drive_folder = r"/content/gdrive/MyDrive/pysource_object_detection"
        self.custom_cfg_path = self.cfg_paths.get(self.model)
        self.new_custom_cfg_path = "cfg/temp-custom-detector.cfg"
        self.new_custom_cfg_test_path = "{}/{}/dnn/{}-custom.cfg".format(self.drive_folder, self.projectname, self.model)
        self.obj_data_path = "data/obj.data"
        self.obj_names_path = "data/obj.names"
        self.images_folder_path = "data/obj/"
        self.backup_folder_path = "{}/{}/dnn/".format(self.drive_folder, self.projectname)

        self.n_classes = 0
        self.n_labels = 0

    def count_classes_number(self):
        print("Detecting classes number ...")
        # Detect number of Classes by reading the labels indexes
        # If there are missing indexes, normalize the number of classes by rewriting the indexes starting from 0
        txt_file_paths = glob.glob(self.images_folder_path + "**/*.txt", recursive=True)
        self.n_labels = len(txt_file_paths)
        print("{} label files found".format(self.n_labels))
        # Count number of classes
        class_indexes = set()
        for i, file_path in enumerate(txt_file_paths):
            # get image size
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()
                for line in lines:
                    numbers = re.findall("[0-9.]+", line)
                    if numbers:
                        # Define coordinates
                        class_idx = int(numbers[0])
                        class_indexes.add(class_idx)

        # Update classes number
        self.n_classes = len(class_indexes)
        print("{} classes found".format(self.n_classes))

        # Verify if there are missing indexes
        if max(class_indexes) > len(class_indexes) - 1:
            print("Class indexes missing")
            print("Normalizing and rewriting classes indexes so that they have consecutive index number")
            # Assign consecutive indexes, if there are missing ones
            # for example if labels are 0, 1, 3, 4 so the index 2 is missing
            # rewrite labels with indexes 0, 1, 2, 3
            new_indexes = {}
            classes = sorted(class_indexes)
            for i in range(len(classes)):
                new_indexes[classes[i]] = i

            for i, file_path in enumerate(txt_file_paths):
                # get image size
                with open(file_path, "r") as f_o:
                    lines = f_o.readlines()
                    text_converted = []
                    for line in lines:
                        numbers = re.findall("[0-9.]+", line)
                        if numbers:
                            # Define coordinates
                            class_idx = new_indexes.get(int(numbers[0]))
                            class_indexes.add(class_idx)
                            text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                            text_converted.append(text)
                    # Write file
                    with open(file_path, 'w') as fp:
                        for item in text_converted:
                            fp.writelines("%s\n" % item)

    def generate_yolo_custom_cfg(self):
        """
        This files loads the yolo
        :return:
        """
        # 1) Edit CFG file
        print("Generating .cfg file for {} classes".format(self.n_classes))
        # print("Classes number: {}")

        # Edit subdivision, max_batches, classes and filters on configuration file
        # Max batches 2000*class, but a minimum of 6000
        max_batches = self.n_classes * 2000
        max_batches = 6000 if max_batches < 6000 else max_batches
        max_batches = self.n_labels if max_batches < self.n_labels else max_batches

        print("Settings:")
        print("max batches: {}".format(max_batches))

        with open(args["cfgpath"], "r") as f_o:
            cfg_lines = f_o.readlines()

        lines = []
        for line in cfg_lines:
            if re.search("subdivisions=[0-9]+\n", line) and args["subdivisions"]:
                print("subdivisions: {}".format(args["subdivisions"]))
                new_line = "subdivisions={}\n".format(args["subdivisions"])
                lines.append(new_line)
                continue
            elif re.search("width=[0-9]+\n", line) and args["imagesize"]:
                print("width: {}".format(args["imagesize"]))
                new_line = "width={}\n".format(args["imagesize"])
                lines.append(new_line)
                continue
            elif re.search("height=[0-9]+\n", line) and args["imagesize"]:
                print("height: {}".format(args["imagesize"]))
                new_line = "height={}\n".format(args["imagesize"])
                lines.append(new_line)
                continue
            elif re.search("max_batches[0-9\s=]+\n", line):
                new_line = "max_batches = {}\n".format(max_batches)
                lines.append(new_line)
                continue
            elif re.search("steps[0-9\s=,]+\n", line):
                new_line = "steps={},{}\n".format(int(max_batches * 0.8), int(max_batches * 0.9))
                lines.append(new_line)
                continue
            elif re.search("classes[0-9\s=]+\n", line):
                new_line = "classes={}\n".format(self.n_classes)
                lines.append(new_line)
                continue
            elif re.search("filters=255\n", line):
                filters = (self.n_classes + 5) * 3
                new_line = "filters={}\n".format(filters)
                lines.append(new_line)
                continue

            # Append the same line if different parameters are not found
            lines.append(line)

        # Saving edited file
        with open(self.new_custom_cfg_path, "w") as f_o:
            f_o.writelines(lines)

    def generate_obj_data(self):
        obj_data = 'classes= {}\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = {}' \
            .format(self.n_classes, self.backup_folder_path)

        # Saving Obj data
        with open(self.obj_data_path, "w") as f_o:
            f_o.writelines(obj_data)

        # Saving Obj names
        with open(self.obj_names_path, "w") as f_o:
            for i in range(self.n_classes):
                f_o.writelines("CLASS {}\n".format(i))

    def generate_train_val_files(self):
        print("Generating Train/Validation list")
        images_list = glob.glob(self.images_folder_path + "**/*.jpg", recursive=True)
        print(images_list)
        print("{} Images found".format(len(images_list)))
        if len(images_list) == 0:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "Images list not found. Make sure that the images are '.jpg' "
                                                         "format and inside the directory {}".format(
                    self.images_folder_path))

        # Read labels
        for img_path in images_list:
            img_name_basename = os.path.basename(img_path)
            img_name = os.path.splitext(img_name_basename)[0]

        with open("data/train.txt", "w") as f_o:
            f_o.write("\n".join(images_list))
        print("Train.txt generated")

        # Generate test files, 10% of training
        test_number = len(images_list) // 10
        print("Test images: {}".format(test_number))
        with open("data/test.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                f_o.writelines("{}\n".format(path))
                if i == test_number:
                    break
        print("Test.txt generated")


if "__main__" == __name__:
    cyd = CustomYOLODetector()

    # Extract images
    cyd.count_classes_number()
    cyd.generate_yolo_custom_cfg()
    cyd.generate_yolo_custom_cfg("test")
    cyd.generate_obj_data()
    cyd.generate_train_val_files()
