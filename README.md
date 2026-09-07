# Visão Computacional - UFPR 2026/2

## Autores

- Arthur Dias Viana de Castro
- Iris Mebs Moraes

Trabalho da disciplina de Visão Computacional da UFPR, semestre 2026/2.

## Comandos make

O projeto possui os seguintes comandos do `make`:

- `make install`: cria e prepara o ambiente virtual do projeto e instala as dependências necessárias para a execução.
- `make run`: executa o programa principal dentro do ambiente virtual, sem necessidade de instalar pacotes localmente no sistema.
- `make clean`: remove arquivos temporários e saídas geradas pelo projeto.

Em geral, o comando `make` sozinho deve cuidar da instalação das dependências e da execução do programa no ambiente virtual, mantendo a configuração do projeto isolada do restante do computador.

## Organização das pastas

A pasta `images` fica na raiz do projeto e armazena as imagens originais baixadas do álbum compartilhado. Cada arquivo é mantido localmente para servir como entrada do processamento.

A pasta `saidas` também fica na raiz do projeto e reúne os resultados do processamento. Para cada imagem, o código cria uma subpasta com o nome da foto, contendo as versões já prontas em escala de cinza, imagens cortadas, segmentações por escala, respostas dos filtros e demais arquivos gerados. Essa organização facilita a comparação entre as imagens originais, as versões processadas e os resultados finais de cada etapa.
