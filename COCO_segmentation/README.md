# Image segmentation using COCO

## Getting the Dataset

To use the code in this project, you'll need to download and set up the COCO dataset. Follow the steps below to obtain the dataset using the provided script.

### Prerequisites

Make sure you have `wget` and `unzip` installed on your system.

### Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/juliusgraf/segmentation/tree/main/COCO_segmentation
cd COCO_segmentation
```

### Step 2: Download and Extract Images
Run the following commands to create the necessary directory structure and download the COCO images:

```bash
mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip

cd ../
```

### Step 3: Download and Extract Annotations
Run the following commands to download and extract the COCO annotations:

```bash
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
```

### Step 4: Ready to Go
You now have the COCO dataset downloaded and ready to use with the code in this repository.