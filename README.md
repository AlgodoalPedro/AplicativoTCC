# FEI Vision Studio

Aplicativo de detecção de objetos usando YOLO com interface gráfica PyQt5 e aceleração GPU.

## 📋 Índice

- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Funcionalidades](#funcionalidades)
- [Otimizações](#otimizações)
- [Troubleshooting](#troubleshooting)

## ✨ Características

- **Detecção de objetos** usando modelos YOLO v8
- **Aceleração por GPU** (NVIDIA CUDA 12.4)
- **Suporte a múltiplas fontes**: imagens, vídeos e webcam
- **Interface moderna** com PyQt5
- **Processamento otimizado** com FP16 para GPUs
- **Redimensionamento automático** de vídeos grandes
- **Gestão inteligente de memória** GPU/RAM
- **Exibição de FPS** em tempo real

## 📁 Estrutura do Projeto

```
AplicativoTCC/
├── src/                          # Código fonte modular
│   ├── __init__.py
│   ├── threads/                  # Threads de processamento
│   │   ├── __init__.py
│   │   ├── yolo_thread.py       # Thread para imagens
│   │   └── webcam_thread.py     # Thread para vídeo/câmera
│   ├── ui/                       # Interface gráfica
│   │   ├── __init__.py
│   │   ├── main_window.py       # Janela principal
│   │   └── styles.py            # Estilos CSS
│   └── utils/                    # Utilitários
│       ├── __init__.py
│       └── image_utils.py       # Funções para imagens
├── main.py                       # Ponto de entrada
├── run.bat                       # Script Windows (recomendado)
├── run_yolo_gui.bat             # Script para versão alternativa
├── yolo_gui_pro.py              # Interface alternativa
├── temporeal.py                 # Demo tempo real
├── resultados/                   # Pasta de saída
├── requirements.txt              # Dependências
└── README.md                     # Este arquivo
```

## 💻 Requisitos

### Sistema
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.12 ou 3.13
- **RAM**: 8GB mínimo, 16GB recomendado

### GPU (Opcional mas Recomendado)
- **GPU**: NVIDIA com suporte CUDA
- **VRAM**: 4GB mínimo, 6GB+ recomendado
- **Driver**: NVIDIA 581.15 ou superior
- **CUDA**: 12.4 ou 13.0

### CPU (Fallback)
- O aplicativo funciona em modo CPU, mas será mais lento
- Vídeos grandes podem ser processados lentamente

## 🔧 Instalação

### 1. Clone o repositório
```bash
git clone <repository-url>
cd AplicativoTCC
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

**Windows (Git Bash/MSYS):**
```bash
source venv/Scripts/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Instale as dependências

**Com GPU (NVIDIA CUDA):**
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Apenas CPU:**
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Verificar instalação
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 🚀 Como Executar

### Opção 1: Scripts .bat (Recomendado no Windows)
```bash
# Clique duas vezes ou execute:
run.bat              # Interface principal
```

### Opção 2: Python direto
```bash
# Com ambiente virtual ativado:
python main.py

# Ou com caminho completo:
venv\Scripts\python.exe main.py
```

### Opção 3: Via CMD (se Git Bash der problemas)
```bash
cmd.exe /c "venv\Scripts\python.exe main.py"
```

## Funcionalidades

### Interface Principal

#### 1. Seleção de Modelo
- Suporta múltiplos modelos YOLO (.pt)
- Modelos detectados automaticamente na raiz do projeto
- Troca de modelo em tempo real

#### 2. Tipos de Detecção

**Modo Imagem:**
- Carrega e processa imagens estáticas
- Formatos: JPG, PNG, BMP, TIFF
- Salva resultado em `resultados/saida.jpg`
- Exibe lista de objetos detectados com confiança

**Modo Vídeo:**
- Processa arquivos de vídeo
- Formatos: MP4, AVI, MOV, MKV
- Redimensionamento automático para 1280px (vídeos grandes)
- Exibição de FPS em tempo real

**Modo Câmera:**
- Detecção em tempo real via webcam
- Suporte a múltiplas câmeras
- Controles de iniciar/parar

#### 3. Visualização
- Preview em tempo real
- Zoom e ajuste automático
- Lista de detecções com scores de confiança
- Barra de progresso para processamento

## Otimizações

### Gerenciamento de Memória
- **Redimensionamento automático**: Vídeos > 1280px são reduzidos
- **FP16 (Half Precision)**: Economiza ~50% de VRAM na GPU
- **Limpeza periódica**: Cache GPU limpo a cada 100 frames
- **Thread segura**: Cleanup automático ao parar detecção

### Performance
- **GPU acelerada**: ~10-50x mais rápida que CPU
- **Processamento assíncrono**: UI responsiva durante detecção
- **Tratamento de erros**: Frames individuais com erro não travam app

### Configurações Padrão
```python
# Parâmetros de inferência
conf=0.5         # Confiança mínima 50%
device='0'       # GPU primária (ou 'cpu')
half=True        # FP16 para GPU
max_size=1280    # Tamanho máximo de frame
```

## Troubleshooting

### Problema: Vídeo muito lento
**Soluções:**
1. Use GPU em vez de CPU
2. Reduza resolução do vídeo manualmente
3. Use modelo YOLO menor (yolov8n.pt em vez de yolov8x.pt)

### Problema: Erro "Unknown property content"
**Causa:** Warnings do Qt, podem ser ignorados
**Solução:** Não afeta funcionalidade, é apenas cosmético

### Problema: `python main.py` não usa venv no Git Bash
**Solução:** Use um dos métodos:
```bash
# Opção 1: Script bat
./run.bat

# Opção 2: Caminho completo
./venv/Scripts/python.exe main.py

# Opção 3: Via CMD
cmd.exe /c "venv\Scripts\python.exe main.py"
```

## Módulos

### src/threads/
- **yolo_thread.py**: Processa detecção em imagens estáticas
  - Carrega modelo YOLO
  - Processa imagem com GPU/CPU
  - Salva resultado anotado
  - Emite detecções via signal

- **webcam_thread.py**: Processa detecção em tempo real
  - Suporta webcam e vídeos
  - Redimensionamento automático
  - Gerenciamento de memória GPU
  - Cálculo de FPS
  - Stop seguro com timeout

### src/ui/
- **main_window.py**: Implementação da janela principal
  - Gerenciamento de estado
  - Controle de threads
  - Eventos de UI
  - Cleanup ao fechar

- **styles.py**: Estilos CSS centralizados
  - Tema moderno
  - Cores consistentes
  - Responsividade

### src/utils/
- **image_utils.py**: Funções auxiliares
  - Redimensionamento de imagens
  - Criação de placeholders
  - Conversão de formatos

## Detalhes Técnicos

### PyTorch & CUDA
- **PyTorch**: 2.6.0+cu124
- **torchvision**: 0.21.0+cu124
- **CUDA Compute**: 12.4
- **Precision**: FP16 (half) na GPU

### Dependências
```
PyQt5==5.15.11
opencv-python==4.10.0.84
ultralytics==8.3.34
numpy>=1.24.0
Pillow>=10.0.0
torch>=2.0.0 (+ CUDA variant)
torchvision>=0.15.0 (+ CUDA variant)
```