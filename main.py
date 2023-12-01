import numpy as np
import matplotlib.pyplot as plt
import cv2

# Função para adicionar ruído à imagem
def add_noise(image, noise_factor=100.0):  
    noise = np.random.normal(scale=noise_factor, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

# Função para melhorar a qualidade usando filtragem bilateral
def improve_quality(image):
    # Ajuste dos parâmetros conforme necessário
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Carregando a imagem
imagem = plt.imread('imagem-incrivel.jpg')  

# Adicionando mais ruído à imagem
imagem_ruidosa = add_noise(imagem, noise_factor=10.0)  

# Normalizando os valores para o intervalo [0, 255]
imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255).astype(np.uint8)

# Aplicando filtragem bilateral para melhorar a qualidade da imagem sem ruído
imagem_melhorada = improve_quality(imagem_ruidosa)

# Exibindo as imagens original, ruidosa e melhorada
plt.subplot(131), plt.imshow(imagem, cmap='gray'), plt.title('Imagem Original')
plt.subplot(132), plt.imshow(imagem_ruidosa, cmap='gray'), plt.title('Imagem Ruidosa')
plt.subplot(133), plt.imshow(imagem_melhorada, cmap='gray'), plt.title('Imagem Melhorada')

# Ajustando a resolução da imagem salva
dpi = 300  
plt.savefig('resultado_com_melhoria.png', dpi=dpi, bbox_inches='tight')  

# Adicionando este comando para exibir a figura
# plt.show()
