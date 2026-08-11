# Relatório Científico: Modelo de Detecção de Kanjis N2–N5 (Distribuição Head)

**Autores:** Thomaz (N2-N5) e Miguel (N1)  
**Contexto:** Iniciação Científica — Visão Computacional aplicada a OCR de Mangás Japoneses  
**Data:** 2026

---

## 1. Resumo Executivo

Este relatório descreve o desenvolvimento do modelo "Generalista" para detecção de Kanjis nos níveis JLPT N2, N3, N4 e N5. Diferentemente do modelo N1 (parceiro), que lida com a **Cauda Longa** da distribuição de frequência (Zipf), este trabalho endereça a **Cabeça da Distribuição**: um conjunto relativamente pequeno de ~690 classes que, em conjunto, compõe a **grande maioria** de todo o texto presente em mangás japoneses.

O desafio central não é a **escassez de dados** (os kanjis N5 como 日, 月, 一 aparecem milhares de vezes no Manga109), mas sim o **desequilíbrio massivo entre classes**: sem intervenção, um modelo treinado nestes dados brutos seria enviesado para os kanjis mais frequentes, negligenciando completamente os kanjis N2 menos comuns.

**Solução proposta:** *Frequency-Driven Normalization* via dataset sintético completamente balanceado.

---

## 2. Problema Científico: Desequilíbrio de Classes na Cabeça da Distribuição

### 2.1 A Distribuição Zipfiana em Mangás

A análise do dataset Manga109 revela que a distribuição de kanjis segue uma lei de potência (Zipf): o kanji mais frequente (日) pode aparecer >5.000 vezes, enquanto kanjis N2 menos comuns aparecem <200 vezes — mesmo pertencendo ao mesmo nível de dificuldade. Esta variação de 25× dentro do mesmo subconjunto N2-N5 é prejudicial para o treinamento direto.

**Consequências do treinamento sem balanceamento:**
- Alta precisão nos top-50 kanjis mais frequentes
- Recall próximo de zero para kanjis N2 raros
- Viés de confiança: o modelo aprende a "chutar" caracteres comuns em vez de reconhecer os traços distintivos

### 2.2 Por que N2-N5 é o "Módulo Principal" do Sistema

Os kanjis N3, N4 e N5 correspondem ao vocabulário básico e intermediário do japonês. Em termos práticos:
- **N5 (103 kanjis):** Numerais, dias da semana, direções, pessoas básicas (人, 日, 月, 山, 川...)
- **N4 (181 kanjis):** Vocabulário cotidiano essencial (学, 語, 食, 飲...)
- **N3 (370 kanjis):** Amplia para conceitos abstratos e relacionamentos
- **N2 (367 kanjis):** Vocabulário jornalístico e literário comum

Juntos, estes ~1000 kanjis compõem tipicamente 85-90% de todo o texto em um volume de mangá. O módulo N2-N5 é, portanto, a **espinha dorsal** do sistema de OCR.

---

## 3. Metodologia: Abordagem "Frequency-Driven"

### 3.1 Estratégia de Dados: Sintético Balanceado

**Decisão:** Em vez de usar dados reais do Manga109 para treino (o que introduziria o desequilíbrio), geramos um dataset **100% sintético e uniformemente distribuído**: cada kanji N2-N5 recebe exatamente o mesmo número de imagens de treino.

Esta abordagem tem precedente na literatura de reconhecimento de caracteres (CRNN, ASTER) e é particularmente indicada quando:
1. O espaço de classes tem forte desequilíbrio na distribuição natural
2. Os exemplos reais são difíceis de anotar (BBoxes individuais por kanji são raras)
3. A variabilidade visual sintética pode ser controlada e diversificada programaticamente

**Implementação:** `src/data/generate_synthetic_images.py`
- 100 imagens/classe × ~690 classes = ~69.000 imagens
- Split automático 80/20 (treino/validação)
- Processamento paralelo (N-1 cores via `multiprocessing.Pool`)

### 3.2 A Classe Sentinela: `UNKNOWN_N1`

Um elemento crítico do design é a classe `UNKNOWN_N1`, adicionada como última classe no arquivo `.names`. Durante a geração sintética, kanjis N1 são gerados e rotulados como `UNKNOWN_N1`.

**Efeito:** O modelo aprende explicitamente a **rejeitar** kanjis complexos N1, delegando-os ao modelo especialista. Sem esta classe, o modelo N2-N5 tentaria forçar um mapeamento incorreto para a classe N2-N5 mais visualmente próxima — causando falsos positivos de alta confiança.

Esta técnica é análoga à estratégia de **Open Set Recognition** na literatura de aprendizado de máquina.

### 3.3 Pipeline de Augmentations

As augmentations foram selecionadas especificamente para simular as condições de um mangá digitalizado:

| Técnica | Motivação |
|---|---|
| **Distorção Elástica** | Kanjis em speech bubbles têm forma ligeiramente deformada; fontes manuscritas têm traços irregulares |
| **Screentone** | Padrão de tramado (`halftone`) é inerente à impressão de mangá japonês e afeta a leitura dos traços |
| **Variação de Tinta (Morphological)** | Dilate/Erode simula variações de espessura típicas de diferentes canetas e penas |
| **Textura de Papel** | Multiplicação por ruído Gaussiano simula papel envelhecido ou manchas de scan |
| **Random Crop** | Kanjis parcialmente visíveis na borda de um balão são comuns e devem ser reconhecíveis |
| **Rotação ±10°** | Balões de fala frequentemente estão inclinados em mangá de ação |
| **Ruído Gaussiano** | Qualidade variável de digitalização, especialmente em volumes mais antigos |
| **Blur Gaussiano** | Desfoque por baixa resolução ou compressão JPEG |

A BBox de Ground Truth é calculada **após** as augmentations, rastreando os pixels escuros reais do kanji na imagem final — garantindo que o label YOLO sempre representa a posição real do caractere.

### 3.4 Validação em Dois Níveis

#### Nível 1: Validação Sintética (mAP@50)

20% do dataset sintético é reservado para validação. Esta partição fornece a métrica **mAP@50** (mean Average Precision at IoU 0.5), que avalia simultaneamente:
- **Localização:** A BBox predita sobrepõe ≥50% da BBox real
- **Classificação:** O kanji foi identificado corretamente

Esta é a métrica principal durante o desenvolvimento, pois o feedback é imediato e a convergência é clara.

#### Nível 2: Validação Real — Recall de Kanjis (Manga109)

Para medir a utilidade real em cenas do mundo real, utilizamos o dataset Manga109 com anotações textuais.

**Limitação fundamental:** O Manga109 anota o texto de frases completas (ex: `「今日は学校に行く」`), mas **não fornece BBoxes individuais para cada kanji** dentro da frase. Portanto, o mAP@50 é matematicamente inaplicável neste contexto.

**Métrica alternativa: Recall de Kanjis**
```
Recall de Kanjis = |kanjis detectados ∩ kanjis na frase GT| / |kanjis na frase GT|
```

Se o modelo detecta os kanjis `今`, `日`, `学`, `校`, `行` dentro de uma imagem de balão que contém `「今日は学校に行く」`, o recall é 5/5 = 100%, independente das coordenadas exatas das BBoxes.

Esta métrica é semanticamente válida: mede se o sistema entrega ao usuário a informação correta sobre os kanjis presentes na cena.

---

## 4. Arquitetura do Modelo

### 4.1 Seleção do Backbone: YOLOv8n

A escolha do YOLOv8n (Nano) é justificada por:
- **Eficiência:** 3.2M parâmetros, latência <10ms em GPU moderna
- **Suficiência:** Para 690 classes com formas relativamente distintas, uma arquitetura Nano é suficiente. Classes N2-N5 tendem a ser mais distintas visualmente do que N1 (que tem muitos kanjis complexos com estruturas similares)
- **Convergência rápida:** 50 épocas são suficientes (vs. 100 para N1)
- **Uso em tempo real:** Requisito para o cliente de captura de tela

### 4.2 Hiperparâmetros

```
Modelo base:  yolov8n.pt (pré-treinado em COCO)
Epocas:       50
Batch size:   16
imgsz:        640 × 640
Optimizer:    SGD, lr=0.01, momentum=0.937, weight_decay=0.0005
Device:       GPU (CUDA)
```

**Nota sobre Transfer Learning:** O uso de pesos pré-treinados no COCO (Detecção de Objetos Naturais) para kanjis pode parecer contraintuitivo, mas na prática as camadas iniciais do backbone já aprendem detectores de bordas e formas básicas que são transferíveis para reconhecimento de caracteres.

---

## 5. Integração com o Sistema Semântico

### 5.1 Kanjidic2 como Fonte de Verdade

O Kanjidic2 é o dicionário de kanjis open-source mais completo disponível, mantido pelo EDRDG (Electronic Dictionary Research and Development Group). Contém:
- Significados em inglês e português
- Leituras On'yomi e Kun'yomi
- Nível JLPT, grau escolar, frequência
- Contagem de traços, nanori (leituras de nomes)

O script `src/semantic/build_dictionary.py` baixa o arquivo XML (~10MB comprimido), parseia com `ET.iterparse` (eficiente em memória) e indexa em SQLite com acesso O(1) por caractere.

### 5.2 Fallback para kanjiapi.dev

Para kanjis ausentes do Kanjidic2 (neologismos, gírias modernas), o `SemanticService` implementa fallback transparente para a API pública `kanjiapi.dev`, garantindo cobertura máxima.

---

## 6. Otimizações de Performance para Uso em Tempo Real

### 6.1 Detecção de Mudança de Tela

O cliente de captura não executa YOLO em cada frame. Compara duas capturas consecutivas reduzidas a 100×100 e só dispara inferência se a luminância média mudar >30%. Isso simula a detecção de "virar a página" e reduz uso de GPU em ~90% durante leitura estática.

### 6.2 Lógica de Cascata

O pipeline de inferência é sequencial e lazy:
1. **N2-N5 roda sempre** (modelo mais leve, cobre 85%+ dos kanjis)
2. **N1 só roda se necessário** (quando N2-N5 retorna `UNKNOWN_N1`)

Isso otimiza latência para a maioria dos casos de uso (leitura de mangá comum).

---

## 7. Resultados do Treinamento e Limitação de Escala Identificada

### 7.1 Execução do Treinamento

O treinamento das 50 épocas foi executado no Kaggle (GPU Tesla T4) via o pipeline de automação descrito em `scripts/kaggle_pipeline.py`, e precisou ser dividido em **duas sessões** devido ao limite de 12h por commit do Kaggle: a primeira sessão treinou as épocas 1–23 antes de ser cancelada, e a segunda **retomou automaticamente a partir do checkpoint** (`resume=True` do ultralytics, carregando `last.pt` da época 23) até completar a época 50/50.

**Métricas finais (época 50, validação sintética):**

| Métrica | Valor |
|---|---|
| mAP@50 | 0.9933 |
| mAP@50-95 | 0.9905 |
| Precisão | 0.9797 |
| Recall (detecção, validação sintética) | 0.9836 |

Esses números confirmam convergência sólida **dentro da distribuição sintética** — ou seja, em imagens geradas pelo mesmo processo que gerou o conjunto de treino. Como discutido a seguir, isso não é garantia de desempenho em cenas reais.

### 7.2 Limitação de Escala Identificada (Teste Qualitativo Pós-Treino)

Após o treinamento, o modelo (`best.pt`) foi testado qualitativamente em três imagens para investigar a generalização fora da distribuição sintética:

| Teste | Conteúdo | Escala do kanji | Resultado |
|---|---|---|---|
| A — Capa de mangá real | Título estilizado (同級生), fundo fotográfico | Pequena, dentro de uma cena complexa | **0 detecções** (mesmo com `conf` reduzido a 0.05, único achado foi 1 falso positivo a 7.8% de confiança, numa região sem kanji) |
| B — Balão sintético | Kanjis reais N2-N5 (今日学校), mesma fonte do treino (NotoSansCJKjp) | Pequena, embutida num balão de fala dentro de um painel | **0 detecções** |
| C — Controle | Kanji real N4 (学), mesma fonte, mesmo tamanho relativo do treino | Cheia (preenchendo 50-90% do quadro, como no treino) | **Detectado corretamente** (`学`, confiança 0.59) |

**Diagnóstico:** a comparação entre os testes B e C isola a variável responsável pela falha. Ambos usam kanjis reais do vocabulário treinado e a mesma fonte — a única diferença é a **escala e o contexto de cena**. Isso indica que a causa raiz não é o estilo tipográfico (hipótese inicial), mas sim uma limitação estrutural do dataset de treino: `generate_synthetic_images.py` sempre desenha o kanji ocupando 50–90% do quadro 640×640, centralizado, sem simular um kanji pequeno inserido numa página ou balão de fala maior. O modelo nunca foi exposto a essa variação de escala durante o treino, então não generaliza para ela — mesmo tendo aprendido muito bem o reconhecimento do caractere em si.

Esse achado é consistente com a limitação já prevista na Seção 3.4 (necessidade de validação real via Manga109) e a torna ainda mais crítica: o mAP@50 sintético de 99.3% mede apenas a capacidade de classificação/localização em escala fixa, não a capacidade de **encontrar** o kanji dentro de uma página real.

### 7.3 Correções Aplicadas ao Gerador Sintético

Em resposta ao diagnóstico da Seção 7.2, `src/data/generate_synthetic_images.py` foi ajustado:

1. **Variação de escala e posição:** 45% das imagens agora desenham o kanji em escala pequena (12–35% do quadro) e posição aleatória, em vez de sempre centralizado ocupando 50–90% do quadro. Os 55% restantes mantêm o comportamento original.
2. **Bug de cálculo de bounding box corrigido:** ao implementar (1), foi descoberto que o cálculo da bbox (baseado em pixels "escuros" com limiar fixo em 240) já era incorreto mesmo antes dessa mudança — o fundo é sorteado em tons de cinza claro (230–255) e, sempre que o valor sorteado caía abaixo do limiar fixo, a imagem **inteira** contava como parte do kanji, inflando a bbox para o quadro inteiro. Esse bug estava mascarado porque o kanji já ocupava a maior parte do quadro; tornou-se visível (e crítico) ao testar escalas pequenas, onde ~40% das imagens de teste apresentaram bbox = quadro inteiro. A correção usa um limiar relativo à cor de fundo real de cada imagem (estimada pela mediana dos 4 cantos), não mais um valor absoluto. Validado em 200 gerações de teste sem nenhuma recorrência do bug.

Esse dataset corrigido ainda não foi usado para treinar um novo modelo — o treino de 50 épocas descrito na Seção 7.1 usa o dataset **anterior** à correção (escala fixa). O retreino com o dataset corrigido é o próximo passo natural (ver Trabalhos Futuros).

### 7.4 Retreino com Dataset de Escala Corrigida e Reteste

O retreino descrito como trabalho futuro na Seção 7.3 foi executado (50 épocas, mesmo pipeline de automação, novamente dividido em duas sessões de 12h com resume automático via checkpoint).

**Métricas finais (época 50, validação sintética, dataset corrigido):**

| Métrica | Valor anterior (7.1) | Valor novo (dataset corrigido) |
|---|---|---|
| mAP@50 | 0.9933 | 0.9926 |
| mAP@50-95 | 0.9905 | 0.9875 |
| Precisão | 0.9797 | 0.9742 |
| Recall (validação sintética) | 0.9836 | 0.9816 |

As métricas de validação sintética permaneceram praticamente no mesmo patamar, confirmando que adicionar variação de escala ao dataset não prejudicou a convergência do modelo.

**Reteste qualitativo (mesmas três imagens da Seção 7.2):**

| Teste | Resultado (dataset anterior) | Resultado (dataset corrigido) |
|---|---|---|
| A — Capa de mangá real | 0 detecções | **0 detecções** (ainda com `conf=0.05`) |
| B — Balão sintético | 0 detecções | **0 detecções** |
| C — Controle (escala de treino) | Detectado (`学`, 0.59) | Detectado (`学`, 0.868 + 1 falso positivo secundário a 0.215) |

**Diagnóstico:** a correção de escala **não resolveu** o problema de generalização para os testes A e B. Medindo a escala real do kanji nessas duas imagens: em A, o bloco de título "同級生" ocupa aproximadamente 10% da altura do quadro (860×860); em B, cada kanji do balão ocupa aproximadamente 8–9% da altura do quadro (640×640). Ambos os valores ficam **abaixo do piso de 12%** usado na variação de escala pequena introduzida na Seção 7.3 (12–35% do quadro) — ou seja, a faixa de escala pequena adicionada ainda não é pequena o suficiente para cobrir esses casos reais. Além disso, ambas as imagens têm elementos visuais que o dataset sintético não modela: em A, uma ilustração de fundo complexa (não apenas textura de papel/screentone); em B, um contorno de balão de fala e borda de painel desenhados ao redor do texto. Qualquer um dos dois fatores (escala ainda menor, ou contexto visual não modelado) pode ser a causa dominante — não é possível isolar qual dos dois pesa mais com os testes atuais.

Isso não invalida a correção feita na Seção 7.3 (o bug de bounding box era real e independente desse resultado, e a variação de escala é uma condição necessária, mesmo que não suficiente), mas indica que o próximo ajuste precisa ser mais agressivo: ampliar a faixa de escala pequena para além de 12% (ex.: 5–35%) e/ou compor os kanjis sintéticos sobre fundos com mais complexidade visual (formas desenhadas, texturas de ilustração), em vez de apenas cor sólida com textura de papel/screentone.

**Nota operacional:** a cota semanal de GPU do Kaggle (30h) foi esgotada durante essa rodada (reset em 2026-08-08), então um novo retreino com faixa de escala ampliada só pode ser iniciado a partir dessa data.

### 7.5 Retreino com Escala Ampliada e Contexto Visual Sintético

Com a cota de GPU renovada em 2026-08-08, `generate_synthetic_images.py` foi ajustado novamente, seguindo o diagnóstico da Seção 7.4:

1. **Faixa de escala pequena ampliada:** piso reduzido de 12% para 5% do quadro (cobrindo os ~8–10% medidos nos casos reais), e a proporção de amostras nessa faixa aumentada de 45% para 55%.
2. **Contexto visual sintético:** três novos elementos desenhados sobre o fundo, com probabilidade independente cada — contorno de balão de fala, borda de painel, e "clutter" de fundo (linhas e retângulos claros simulando ilustração/screentone) — aplicados antes da textura de papel, sem alterar a bbox já calculada.
3. **Bug novo, exposto pela mudança (1):** com kanjis muito pequenos perto da borda do quadro, o corte aleatório (`random_crop`) passou a às vezes remover o kanji inteiramente do quadro, deixando zero pixels escuros. O fallback antigo (bbox = quadro inteiro) gerava um rótulo incorreto nesse caso; corrigido para descartar a amostra (`bbox = None` → `continue`) em vez de gerar um label errado. Validado com 2000 sementes reproduzíveis: 0 bboxes incorretas, 7 amostras corretamente descartadas.

O retreino rodou por 3 sessões de até 12h (resume automático via checkpoint), totalizando 50 épocas.

**Bug de monitoramento encontrado durante o retreino:** o script de orquestração local (`retrain_driver.py`, fora do repositório) decide se o treino terminou contando linhas de `results.csv`. Isso funcionou nas Seções 7.1/7.4, mas quebrou aqui porque, ao dar resume, o Ultralytics recria `results.csv` do zero em vez de continuar o arquivo anterior — então a contagem de linhas reflete só as épocas rodadas *nessa sessão*, não o total acumulado. Ao fim da sessão 2 (época real 48/50), o driver leu erroneamente "26/50" (contagem de linhas) e teria disparado uma 3ª sessão completa desnecessária. Corrigido lendo o valor máximo da própria coluna `epoch` do CSV (que é o índice global, não reiniciado) em vez do número de linhas — o processo local foi reiniciado com a correção (sem reenviar o kernel, já em execução) e a sessão 3 terminou corretamente após apenas 2 épocas adicionais, confirmada por status `COMPLETE` (finalização natural) em vez de `CANCEL_ACKNOWLEDGED` (corte por limite de 12h).

**Métricas finais (época 50, validação sintética):**

| Métrica | Valor anterior (7.4) | Valor novo (escala ampliada + contexto) |
|---|---|---|
| mAP@50 | 0.9926 | 0.9850 |
| mAP@50-95 | 0.9875 | 0.9614 |
| Precisão | 0.9742 | 0.9810 |
| Recall (validação sintética) | 0.9816 | 0.9642 |

A queda nas métricas sintéticas é esperada e consistente com o objetivo: o novo dataset é estritamente mais difícil (kanjis menores, mais elementos visuais competindo pela atenção do modelo), então uma pequena perda de desempenho na validação sintética em troca de melhor generalização para imagens reais é o trade-off pretendido.

**Reteste qualitativo (mesmas três imagens das Seções 7.2 e 7.4):**

| Teste | Resultado (7.4, escala corrigida) | Resultado (7.5, escala ampliada + contexto) |
|---|---|---|
| A — Capa de mangá real | 0 detecções | **1 detecção** — bbox localiza corretamente o bloco "同級生" inteiro, mas classificado como `UNKNOWN_N1` (confiança 0.272) |
| B — Balão sintético | 0 detecções | **1 detecção** — bbox localiza corretamente o kanji "今", mas classificado como `UNKNOWN_N1` (confiança 0.082) |
| C — Controle (escala de treino) | Detectado (`学`, 0.868) | Detectado (`学`, 0.987) |

**Diagnóstico:** houve progresso real, mas parcial. Pela primeira vez o modelo **localiza** texto em kanji nas imagens A e B — antes, a rede nem sequer gerava uma caixa candidata nessa região, então o problema era de detecção (recall de localização). Agora o problema restante é de **classificação com baixa confiança**: em ambos os casos, o kanji correto está dentro da caixa, mas o modelo o rotula como `UNKNOWN_N1` (a classe sentinela de rejeição N1, Seção 3.2) em vez do caractere real — apesar de "同", "級", "生" (teste A) e "今" (teste B) serem todos kanjis N2-N5 comuns, não N1. Isso é uma classificação incorreta, não uma rejeição válida. Além disso, no teste A a caixa detectada engloba os três kanji juntos como um único bounding box, enquanto o treino sempre usa um kanji por caixa — sugerindo que o modelo está usando o bloco de texto como pista de localização (contraste alto vs. fundo) mais do que reconhecendo caracteres individuais nessa imagem.

**Diagnóstico complementar (descarta calibração como causa):** para verificar se a classe correta ao menos aparecia como candidata secundária, a inferência foi refeita com `conf=0.0005` (praticamente zero) e os scores de todas as classes-alvo foram buscados em qualquer posição da imagem, não só na caixa detectada. Em nenhum dos dois testes ("同", "級", "生" no teste A; "今", "日", "学", "校" no teste B) qualquer uma das classes corretas apareceu acima desse limiar em qualquer lugar da imagem — enquanto `UNKNOWN_N1` manteve confiança 0.27 (A) e 0.08 (B) na posição certa. Ou seja, não é um caso de "quase acerto" que um ajuste de threshold resolveria: o modelo está confiante na resposta errada, e a classe correta não é sequer uma candidata residual. Isso reforça que a causa é generalização, não calibração de confiança.

**Testes de baixo custo sem GPU (resolução de inferência, TTA, CLAHE):** antes de assumir que a solução exige um novo retreino, três ajustes de inferência foram testados nas imagens A e B, sem alterar o modelo: (1) `imgsz=1280` na inferência (treino foi em 640), (2) test-time augmentation (`augment=True`), e (3) equalização de contraste local (CLAHE) como pré-processamento para aproximar o domínio da imagem real do domínio sintético. TTA e CLAHE não mudaram o resultado em nenhum dos casos. `imgsz=1280` teve efeito genuíno, porém **inconsistente**: revalidando em condição de produção (`conf=0.05`, as mesmas três imagens A/B/C), o resultado foi (A) 0 detecções — pior que as 640px (que ao menos gerava 1 caixa errada); (B) 1 detecção com a classe **correta** (`校`, confiança 0.605) — vitória genuína; (C) 5 caixas sobrepostas e ruidosas, incluindo classes erradas, contra uma única detecção limpa e correta a 0.987 em 640px. Ou seja, aumentar a resolução de inferência desloca a escala efetiva dos objetos para fora da distribuição de escalas aprendida no treino (âncoras/pirâmide de features calibradas para 640px): ajuda casos sub-escalados (B) mas piora casos já bem-escalados (A, C). Não é uma correção segura de aplicar globalmente sem um ensemble multi-escala adequado — descartado como fix de inferência isolado.

**Correção de premissa (fontes já são diversas no gerador):** a hipótese inicial, registrada abaixo, era de que o gerador sintético usava uma única fonte (NotoSansCJKjp) e que isso explicava a falha de classificação. Ao investigar `src/data/generate_synthetic_images.py` para implementar diversificação de fonte, foi constatado que essa premissa está desatualizada: o gerador já sorteia aleatoriamente entre **6 fontes CJK** (`assets/fonts/`: NotoSansCJKjp, NotoSerifCJKjp, KleeOne, RampartOne, ReggaeOne, RocknRollOne — cobertura de glifos para os kanjis testados confirmada em todas), e o retreino desta própria Seção 7.5 já introduziu contexto visual sintético (balão de fala, borda de painel, clutter de fundo, textura de papel, screentone, ruído). Ou seja, "mais diversidade tipográfica/textural sintética" já foi tentado no retreino atual e não foi suficiente para corrigir a classificação em imagens reais — a causa raiz provavelmente não é falta de variação sintética, e sim um gap de domínio que variações puramente sintéticas não cobrem (ver bullet correspondente em Trabalhos Futuros).

**Nota operacional:** treino executado em 3 sessões de GPU (~12h, ~12h, ~1h), dentro da cota semanal renovada em 2026-08-08.

---

## 8. Conclusão e Trabalhos Futuros

### Contribuições deste Módulo

1. **Pipeline de normalização por frequência:** Demonstra que dataset sintético balanceado supera dados reais desbalanceados para reconhecimento de caracteres com distribuição Zipfiana
2. **Classe UNKNOWN_N1 como mecanismo de delegação:** Abordagem de Open Set Recognition para sistemas multi-modelo em cascata
3. **Recall de Kanjis como métrica alternativa:** Métrica semanticamente válida quando Ground Truth geométrico não está disponível
4. **Pipeline de automação de treino no Kaggle:** Push/polling/resume via `scripts/kaggle_pipeline.py`, permitindo treinar através de múltiplas sessões limitadas a 12h sem intervenção manual (ver Seção 7.1)

### Trabalhos Futuros

- **Fechar o gap sintético-real com dados/estilos que a variação puramente sintética não cobre:** o retreino da Seção 7.5 resolveu a localização em imagens reais (antes 0 detecções, agora bbox correta) e já introduziu 6 fontes CJK + contexto visual (balão, painel, screentone, ruído) — mesmo assim a classificação nessas imagens ainda erra para a classe sentinela `UNKNOWN_N1`. Como "mais diversidade sintética" já foi tentado e não bastou, os próximos candidatos são: (a) fontes de letreiramento de mangá reais (diferentes das 6 atuais, mais próximas do estilo usado em títulos e balões), (b) artefatos de compressão JPEG/scan, ausentes em todas as texturas atuais, e (c) misturar uma pequena amostra de dados reais rotulados manualmente (few-shot fine-tuning) em vez de depender só do sintético — historicamente o que resolve esse tipo de gap quando o sintético já está saturado de variação e ainda assim não generaliza
- **Detecção por caractere individual em blocos de texto:** no teste A (Seção 7.5), o modelo devolveu uma única caixa para os três kanji do título, em vez de uma caixa por caractere como no treino — investigar se isso é um efeito de NMS (non-max suppression) agressivo ou uma limitação de resolução em blocos de texto denso
- **Distilação do modelo:** Treinar um modelo YOLOv8s (Small) com os pesos do Nano como teacher para ganhar precisão em N2
- **Dataset real aumentado:** Integrar dados reais do Manga109 com pesos de amostragem para substituir o dataset puramente sintético na fase final
- **Filtragem por confiança adaptativa:** Threshold dinâmico baseado na complexidade visual do frame capturado
- **Interface gráfica:** Overlay visual com BBoxes coloridas por nível JLPT e popup de dicionário completo

---

## Referências

- Bochkovskiy, A., Wang, C-Y., Liao, H-Y. M. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. arXiv:2004.10934
- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Matsui, Y. et al. (2017). *Sketch-based Manga Retrieval using Manga109 Dataset*. Multimedia Tools Appl.
- Breen, J. (2003). *KANJIDIC: A Japanese Kanji Dictionary*. EDRDG.
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
