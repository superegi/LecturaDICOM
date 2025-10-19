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


# extraemos el centro y el tamaño de la imagen
height, width = im1.shape[:2]
centerX, centerY = (width // 2, height // 2)

# obtenemos la matriz de rotacion
M = cv2.getRotationMatrix2D((centerX, centerY), 45, 1.0)

# rotamso la imagen
im1_R = cv2.warpAffine(im1, M, (width, height))

# mostramos la imagen
fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('Original')
ax[0].axis(False)
ax[1].imshow(im1_R)
ax[1].set_title('Imagen rotada')
ax[1].axis(False)
plt.show()

# traslacion de una imagen
# almacenamos el valor de la altura y el ancho de la imagen
height, width = im1.shape[:2]
quarter_height, quarter_width = height / 4, width / 4

# construimos la matriz de traslación
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

# Usamos la funcion warpAffine para trasladar la imagen
img_translation = cv2.warpAffine(im1, T, (width, height))

# mostramos la imagen
fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('Original')
ax[0].axis(False)
ax[1].imshow(img_translation)
ax[1].set_title('Iamgen trasladada')
ax[1].axis(False)


# cambiando el tamaño de la imagen
resized_image = cv2.resize(im1, (240, 240), dst=None, fx=None, fy=None, interpolation=cv2.INTER_CUBIC)
# mostramos la imagen
fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('Original')
ax[0].axis(False)
ax[1].imshow(resized_image)
ax[1].set_title('Imagen resize')
ax[1].axis(False)



# zoom in zoom outz
zoom_factor = 0.5
im1_resize = cv2.resize(im1, None, fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_LINEAR).astype(float)
im1_resize[:,:,0] = (im1_resize[:,:,0]/np.amax(im1_resize[:,:,0]))*255.
im1_resize[:,:,1] = (im1_resize[:,:,1]/np.amax(im1_resize[:,:,1]))*255.
im1_resize[:,:,2] = (im1_resize[:,:,2]/np.amax(im1_resize[:,:,2]))*255.
im1_resize = im1_resize.astype(np.uint8)

difrows = im1.shape[0] - im1_resize.shape[0]
difcol = im1.shape[1] - im1_resize.shape[1]
if difrows<0:
  difrows = abs(difrows)
  difcol = abs(difcol)
  im1_resize_zoom = im1_resize[difrows//2:im1_resize.shape[0]-difrows//2, difcol//2:im1_resize.shape[1]-difcol//2]
else:
  difrows = abs(difrows)
  difcol = abs(difcol)
  im1_resize_zoom = np.zeros_like(im1)
  im1_resize_zoom[difrows//2:im1.shape[0]-difrows//2, difcol//2:im1.shape[1]-difcol//2,:] = im1_resize
# mostramos la imagen
fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].imshow(im1)
ax[0].set_title('Original')
ax[0].axis(False)
ax[1].imshow(im1_resize_zoom)
ax[1].set_title('Imagen zoom')
ax[1].axis(False)
plt.show()


# podemos mostrar cada valor de intensidad en un grafico 3D.
# donde los ejes corresponden a cada uno de los canales.
R = resized_image[:,:,0].flatten()
G = resized_image[:,:,1].flatten()
B = resized_image[:,:,2].flatten()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
ax.scatter(R, G, B)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.view_init(30, 30, 0)
plt.show()



