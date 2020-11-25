# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
LOCAL_PATH=model_emb.sav

# project id
PROJECT_ID=green-mood-tracker-296712

# bucket name
BUCKET_NAME=green-mood-tracker-01

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=models

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=europe-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload:
	# uploading the file
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${UPLOADED_FILE_NAME}
