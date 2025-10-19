import kagglehub
import os

DATASET = "nodoubttome/skin-cancer9-classesisic"
folder = kagglehub.dataset_download(DATASET, force_download=False)  # <— clave
print("Ruta a la carpeta:", folder)


subfolder = os.listdir(folder) # vemos la carpeta principal
subfolder

path = folder + '/' + subfolder[0] + '/' + 'Train'
path


folder_test = os.listdir(folder + '/' + subfolder[0] + '/' + 'Test') # las carpetas que contiene
folder_train = os.listdir(folder + '/' + subfolder[0] + '/' + 'Train')
folder_train.sort()

subfolder[0]

# vamos a leer todos los nombres de los archivos que esten contenidos en la carpeta "Skin cancer ISIC The International Skin Imaging Collaboration/Train/melanoma/"
files = os.listdir(folder + '/' + subfolder[0] + '/' + 'Train' + '/' + 'melanoma')
len(files)

# vamos a leer una imagen de las disponibles en el dataset
# carpeta train/benign/
# para esto importaremos el modulo de opencv
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt

for x in [0,1,2,3,4,5,6,7,8]:
	print(files[x])

files.sort()
print('---------------')
for x in [0,1,2,3,4,5,6,7,8]:
	print(files[x])

# leer imagen con openCV, en este caso el archivo se lee en BGR
im1 = cv2.imread(folder + '/' + subfolder[0] + '/' + 'Train' + '/' + 'melanoma/' + files[5])
# por lo cual debemos convertirla a RGB
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)


# leer imagen con PIL, en este caso el archivo se lee en RGB
im2 = np.array(PIL.Image.open(folder + '/' + subfolder[0] + '/' + 'Train' + '/' + 'melanoma/' + files[5]))



fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('OpenCV')
ax[0].axis(False)
ax[1].imshow(im2)
ax[1].set_title('PIL')
ax[1].axis(False)
plt.show()

print(im1.shape)

# podemos separar los canales en imagenes por separado
im1_R = im1[:,:,0]
im1_G = im1[:,:,1]
im1_B = im1[:,:,2]

fig, ax = plt.subplots(1,4,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('Original')
ax[0].axis(False)
ax[1].imshow(im1_R,cmap='gray')
ax[1].set_title('Canal R')
ax[1].axis(False)
ax[2].imshow(im1_G,cmap='gray')
ax[2].set_title('Canal G')
ax[2].axis(False)
ax[3].imshow(im1_B,cmap='gray')
ax[3].set_title('Canal B')
ax[3].axis(False)
plt.show()


