import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "saidas"


def carregar_imagens_local(pasta=IMAGES_DIR, extensoes=(".jpg", ".png", ".jpeg")):
    pasta = Path(pasta)
    if not pasta.is_absolute():
        pasta = PROJECT_ROOT / pasta

    caminhos = []
    for arquivo in sorted(pasta.iterdir()):
        if arquivo.is_file() and arquivo.name.lower().endswith(extensoes):
            caminhos.append(str(arquivo))
    return caminhos

def carregar_preparar_imagem(caminho_imagem, tamanho=(512, 512)):
    img_rgb = cv2.imread(caminho_imagem)
    if img_rgb is None:
        print(f"Erro ao carregar {caminho_imagem}")
        return None
    img_gray = rgb2gray(img_rgb)
    img_gray_resized = resize(img_gray, tamanho, anti_aliasing=True)
    return (img_gray_resized * 255).astype(np.uint8)


def salvar_imagem_pb_e_cortada(caminho_imagem, pasta_destino, tamanho=(512, 512)):
    img_rgb = cv2.imread(caminho_imagem)
    if img_rgb is None:
        print(f"Erro ao carregar {caminho_imagem}")
        return None

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray_resized = cv2.resize(img_gray, tamanho, interpolation=cv2.INTER_AREA)

    pasta_destino.mkdir(parents=True, exist_ok=True)

    nome_base = Path(caminho_imagem).stem
    caminho = pasta_destino / f"{nome_base}.png"

    cv2.imwrite(str(caminho), img_gray_resized)
    return img_gray_resized

def criar_filtros_textura():
    filtros = []
    nomes = []
    filtros.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])); nomes.append("Sobel X")
    filtros.append(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])); nomes.append("Sobel Y")
    filtros.append(np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])); nomes.append("Diagonal 45°")
    filtros.append(np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])); nomes.append("Diagonal 135°")
    ksize = 9
    sigma = 1.5
    g = cv2.getGaussianKernel(ksize, sigma)
    log_kernel = cv2.Laplacian(g @ g.T, cv2.CV_64F)
    filtros.append(log_kernel); nomes.append("LoG")

    filtros.append(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])); nomes.append("Laplaciano")

    gabor_kernel = cv2.getGaborKernel((9, 9), sigma=3.0, theta=0, lambd=8.0, gamma=0.5, psi=0)
    filtros.append(gabor_kernel); nomes.append("Gabor 0°")
    gabor_kernel_90 = cv2.getGaborKernel((9, 9), sigma=3.0, theta=np.pi / 2, lambd=8.0, gamma=0.5, psi=0)
    filtros.append(gabor_kernel_90); nomes.append("Gabor 90°")


    return filtros, nomes


def aplicar_filtros(imagem, filtros):
    return [cv2.filter2D(imagem, -1, f) for f in filtros]

def extrair_vetores_textura_por_escala(respostas, h, w, tamanho_janela):
    vetores, posicoes = [], []
    for y in range(0, h - tamanho_janela + 1, tamanho_janela):
        for x in range(0, w - tamanho_janela + 1, tamanho_janela):
            vetor = []
            for resposta in respostas:
                janela = resposta[y:y+tamanho_janela, x:x+tamanho_janela]
                vetor.append(np.mean(janela))
                vetor.append(np.std(janela))
            vetores.append(vetor)
            posicoes.append((y, x))
    return np.array(vetores), posicoes

def agrupar_com_kmeans(vetores, k_clusters, escala_id=0):
    kmeans = KMeans(n_clusters=k_clusters, random_state=100, n_init=10)
    rotulos = kmeans.fit_predict(vetores)
    return rotulos, kmeans.cluster_centers_

def reconstruir_imagem_segmentada(rotulos, posicoes, h, w, k_clusters, tamanho_janela):
    import matplotlib
    cores = np.uint8(matplotlib.colormaps["tab10"](np.linspace(0, 1, k_clusters))[:, :3] * 255)
    img_segmentada = np.zeros((h, w, 3), dtype=np.uint8)
    for (y, x), label in zip(posicoes, rotulos):
        img_segmentada[y:y+tamanho_janela, x:x+tamanho_janela] = cores[label]
    return img_segmentada

def processar_imagem(imagem_cinza, filtros, num_escalas=3, tamanho_janela_base=16, k_clusters=3):
    respostas_multi_escala, dimensoes, imagens = [], [], []
    img_atual = imagem_cinza.copy()

    for _ in range(num_escalas):
        respostas = aplicar_filtros(img_atual, filtros)
        respostas_multi_escala.append(respostas)
        dimensoes.append(img_atual.shape)
        imagens.append(img_atual)
        img_atual = cv2.pyrDown(cv2.GaussianBlur(img_atual, (5,5), 1.5))

    todas_segmentacoes = []
    for i in range(num_escalas):
        h, w = dimensoes[i]
        respostas = respostas_multi_escala[i]
        tamanho_janela = max(4, tamanho_janela_base // (2**i))
        vetores, posicoes = extrair_vetores_textura_por_escala(respostas, h, w, tamanho_janela)
        rotulos, _ = agrupar_com_kmeans(vetores, k_clusters, escala_id=i)
        segmentada = reconstruir_imagem_segmentada(rotulos, posicoes, h, w, k_clusters, tamanho_janela)
        todas_segmentacoes.append((segmentada, imagens[i]))
    return todas_segmentacoes, respostas_multi_escala, filtros

def plotar_filtros(filtros, nomes):
    n = len(filtros)
    cols = (n + 1) // 2
    plt.figure(figsize=(3 * cols, 6))
    plt.suptitle("Filtros de Textura (kernels)")
    for i, (filtro, nome) in enumerate(zip(filtros, nomes)):
        plt.subplot(2, cols, i + 1)
        plt.imshow(filtro, cmap="gray")
        plt.title(nome)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "filtrosimg.png", dpi=150)
    print(f"Imagem dos filtros salva em: {OUTPUT_DIR / 'filtrosimg.png'}")
    plt.close('all')

def sobrepor_segmentacao(original, segmentada, alpha=0.5):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(original_rgb, alpha, segmentada, 1 - alpha, 0)


def main():
    caminhos = carregar_imagens_local(IMAGES_DIR)
    if not caminhos:
        print(f"Nenhuma imagem encontrada na pasta '{IMAGES_DIR}'")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filtros, nomes_filtros = criar_filtros_textura()

    for caminho in caminhos:
        imagem = carregar_preparar_imagem(caminho)
        print(f"/n---/nProcessando imagem: {caminho}")
        if imagem is None:
            continue

        base = os.path.splitext(os.path.basename(caminho))[0]
        pasta_imagem = OUTPUT_DIR / base
        pasta_imagem.mkdir(parents=True, exist_ok=True)

        salvar_imagem_pb_e_cortada(caminho, pasta_imagem)

        segmentacoes, respostas_filtros, filtros = processar_imagem(
            imagem, filtros, num_escalas=3, tamanho_janela_base=16, k_clusters=5
        )

        for i, (seg, original) in enumerate(segmentacoes):
            sobreposicao = sobrepor_segmentacao(original, seg)

            plt.figure(figsize=(15, 4))
            plt.suptitle(f"Escala {i+1} - {os.path.basename(caminho)}")
            plt.subplot(1, 3, 1)
            plt.imshow(original, cmap="gray")
            plt.title("Original")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(seg)
            plt.title("Segmentada")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(sobreposicao)
            plt.title("Sobreposição")
            plt.axis("off")
            plt.tight_layout()
            caminho_saida = pasta_imagem / f"segescala{i+1}.png"
            plt.savefig(caminho_saida, dpi=150)
            print(f"Imagem segmentada salva em: {caminho_saida}")
            plt.close()

        for escala, respostas in enumerate(respostas_filtros):
            plt.figure(figsize=(15, 3))
            plt.suptitle(f"Respostas dos Filtros - Escala {escala + 1}")
            for i, resposta in enumerate(respostas):
                plt.subplot(1, len(respostas), i + 1)
                plt.imshow(resposta, cmap='gray')
                plt.title(nomes_filtros[i])
                plt.axis("off")
            plt.tight_layout()
            caminho_saida = pasta_imagem / f"respostas_escala{escala+1}.png"
            plt.savefig(caminho_saida, dpi=150)
            print(f"Imagem das respostas dos filtros salva em: {caminho_saida}")
            plt.close()

        plt.close('all')


if __name__ == "__main__":
    main()

