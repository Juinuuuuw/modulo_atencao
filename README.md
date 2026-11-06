# Observa — Monitor multicâmera na rede local

## Calibrar cada câmera (página de alvo)

Abra `http://IP_DO_SERVIDOR:8765/alvo?display=notebook` (ou `?display=tv` na TV) e informe o participante. Em seguida escolha:

- **O que calibrar:** *Tela do notebook*, *Tela da apresentação (TV)* (9 pontos numa grade 3×3, mais 4 de validação) ou *Robô NAO* (6 partes físicas do robô, mais as 6 repetidas em outra ordem para validação). A tela escolhida aqui define onde o mapa é salvo: escolher a outra tela recarrega a página como aquela tela. Antes, uma janela aberta com `?display=notebook` e arrastada para a TV gravava o mapa como notebook.
- **Câmera:** notebook, NAO ou apresentação. Calibre **uma câmera de cada vez**. Cada câmera tem o próprio mapa, e calibrar a segunda câmera na mesma tela não apaga a primeira.

Clique em **Calibrar e validar**. A validação começa sozinha no fim da calibração. No robô, a página mostra o desenho com a parte destacada e fala "Olhe para a cabeça do robô", "o braço à sua esquerda", e assim por diante. O participante olha para o robô físico, não para o desenho. As partes são cabeça, os dois braços, o peito e as duas pernas. Os lados são os **do participante**: o braço direito do robô fica à esquerda de quem o olha de frente.

Os mapas ficam salvos por participante em `sessions/participants/`. Eles voltam após reiniciar o servidor ou reaplicar as câmeras, desde que cada papel use a mesma câmera física (mesmo computador e índice). Se o papel passar a usar outra câmera, a estimativa some com o motivo, até recalibrar. A lista **Já calibrado** da página mostra o que existe para o participante e o resultado de cada validação.

### Validação

Durante a validação a estimativa **não aparece**: o ponto azul que aparecia antes puxava o olhar do participante, que o seguia em vez do amarelo. O resultado aparece no fim, em três níveis:

- **Tela:** *alta* (os limites originais: erro médio normalizado ≤ 0,09, máximo ≤ 0,14, p90 ≤ 0,16 e melhora ≥ 50% sobre prever sempre o centro); *moderada* (erro médio ≤ 0,16, p90 ≤ 0,30 e melhora ≥ 35%, suficiente para distinguir regiões grandes); *baixa* nos demais casos.
- **Robô:** *alta* com pelo menos 5 de 6 partes certas e erro médio ≤ 0,12 do desenho; *moderada* com pelo menos metade certa; *baixa* nos demais casos.

São limites empíricos, não certificação científica. Uma validação ruim não apaga o mapa: a estimativa continua disponível, identificada com o nível. **Validar de novo** repete só a validação.

### Mapa

### Robô × tela: a zona de cada mapa

Nos dados do P005, pela câmera do notebook, a íris fica quase igual olhando o robô ou a tela do notebook: íris e pálpebras separam os dois em só 63–70% das leituras, enquanto a pose da cabeça separa 99%. Por isso cada calibração guarda também uma **zona**: como ficam íris, cabeça e pálpebras olhando aquela superfície (Mahalanobis com encolhimento, limiar no percentil 97,5 × 1,3 das leituras de ajuste, sem leituras fora da curva). Uma leitura só conta para a superfície se cair na zona dela. Quando a mesma câmera tem mapas de mais de uma superfície, a leitura vai para a zona mais compatível. Reaplicando as leituras de validação do P005: olhando a tela, “robô” caiu de quase sempre para 17% (tela 67%); olhando o robô, 88% robô e 3% tela. **Calibre também uma tela com a mesma câmera usada no robô**; sem essa comparação, a zona do robô sozinha é larga.

No robô, o mapa das partes pode usar a pose da cabeça (candidatos cabeça, cabeça + pálpebras, íris + cabeça), escolhida só se reduzir o erro deixando cada parte de fora. A parte só é mostrada quando a validação daquela câmera acertou ao menos metade das partes; abaixo disso aparece “olhando o robô”, sem parte. Com a câmera do notebook, o P005 acertou 2 de 6 partes: essa câmera diz se a pessoa olha o robô, mas não qual parte. A câmera do NAO, de frente para o participante, tende a ser a melhor para as partes. Mapas salvos antes da zona continuam funcionando com a regra anterior.

O mapa é uma regressão ponderada (cada ponto com o mesmo peso) a partir da posição da íris e da abertura das pálpebras (EAR). A EAR acompanha o olhar vertical melhor que a altura da íris numa webcam acima da tela. O sistema compara candidatos (íris; íris + pálpebras; íris em curva + pálpebras, este só nas telas) e três regularizações, prevendo cada ponto com um mapa ajustado sem ele. Um candidato mais complexo só é escolhido se reduzir esse erro em pelo menos 10% e 5 px. No robô, o mapa leva o olhar a coordenadas do desenho do robô, e a parte é a mais próxima da estimativa (até 0,22 do desenho). Mais longe que isso, a estimativa conta como "fora do robô".

Antes do ajuste, piscadas (EAR abaixo de 75% da mediana do ponto) e leituras fora da curva são descartadas. Durante o uso, EAR abaixo de 60% da referência esconde o ponto como piscada. A pose da cabeça não entra no mapa: em dados reais ela piorou a previsão em pontos de teste. Cada ponto espera uma leitura válida e pode se estender até 10 s para obter 6 quadros distintos. As amostras e o motivo de término de cada calibração ficam em `sessions/calibrations/<id>.json`.

### Janela e desempenho

Se a janela sair da tela cheia ou mudar de monitor no meio da calibração, a coleta **espera** a janela voltar, em vez de falhar no último ponto. Uma segunda aba `/alvo` aberta para a mesma tela é ignorada, e a página avisa. Durante a calibração, a câmera calibrada é analisada a cerca de 30 quadros por segundo e as outras a cerca de 5. Analisar as três a 30 Hz travava o servidor.

Não é um raio ocular físico e não mede atenção ou compreensão. Mantenha a postura e a disposição usadas no ajuste, e refaça após mudar a disposição.

## Olhar ao vivo (operador)

No topo da aba ficam as três câmeras em tempo real (notebook, NAO e apresentação), com as linhas dos olhos, a escolha das origens e o botão **Aplicar câmeras**. É o mesmo bloco da aba **Câmeras e sessão**: ele acompanha a aba aberta.

Escolha o **participante** no seletor. O padrão é o último calibrado. A aba mostra as estimativas de todas as câmeras calibradas para ele, cada câmera com uma cor:

- **Apresentação / TV** e **Notebook**: desenho da tela com um ponto por câmera e o rastro dos últimos 2 segundos.
- **Robô NAO**: o desenho com a estimativa de cada câmera e a parte olhada (cabeça, braços, tronco ou pernas) destacada.
- **"Agora a pessoa está olhando para":** cada câmera vota na tela ou no robô onde a estimativa dela cai e conta contra uma superfície que calibrou, se vê o olhar fora dela. Uma câmera sem mapa do robô não contradiz a câmera que vê o olhar no robô. Empate fica **Indeterminado**. Se todas as câmeras com leitura veem o olhar fora de tudo, o resultado é **Outro local**. Sem nenhuma calibração de posição, a aba usa o classificador de **Calibrar nesta disposição**, se houver.

A aba não precisa da página de alvo aberta e pode ficar aberta enquanto a apresentação roda. Com ela aberta, o servidor analisa cerca de 15 quadros por segundo (em vez de 5). O registro da sessão continua a cerca de 5 Hz. Durante a gravação, `predictions.jsonl` também salva `screen_gaze`, com a estimativa de cada mapa (`superfície:câmera`; no robô, inclui a parte) ou `null`.

Sistema central para observar alvos visuais no ambiente com notebook, TV e robô NAO. O servidor reúne três perspectivas faciais: a câmera do notebook, a câmera interna do NAO e uma webcam na tela de apresentação (TV). A aplicação de apresentação continua separada; seus marcos podem ser registrados no painel ou pela API.

## Iniciar o servidor

Use Python 3.10 de 64 bits (o MediaPipe 0.10.14 está fixado neste projeto). Execute `iniciar_servidor.bat` ou:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py --open
```

O painel abre em `http://localhost:8765`. No notebook remoto, use `http://IP_DO_SERVIDOR:8765`. O terminal mostra o endereço LAN e o código de acesso, também salvo em `.local/access-key.txt`. O código é compartilhado com os operadores e conectores autorizados. Não exponha a porta na Internet. O transporte HTTP não cifra o vídeo; para redes não controladas, configure HTTPS por proxy.

O servidor procura suas próprias webcams nos índices 0–3 e as registra no painel. Para desativar essa busca, use `--no-local-cameras`. O endereço de escuta padrão é `0.0.0.0`; altere com `--host` e `--port`. Se o Windows bloquear acesso na rede privada, o responsável pelo computador precisará configurar a liberação. O programa não modifica o firewall.

## Conectar e selecionar as três câmeras

Abra **Conectar dispositivos** no painel e baixe o ZIP. Consulte [CLIENTE.md](CLIENTE.md) para a instalação.

- **Notebook:** execute `iniciar_cliente.bat` no notebook participante. Informe o servidor e seu código.
- **NAO:** execute `nao_client.py` com o Python/SDK NAOqi do robô. São oferecidas as câmeras superior e inferior; escolha somente uma, normalmente a superior.
- **Apresentação / TV** (papel interno `validation`, mantido por compatibilidade com sessões antigas): webcam junto à tela de apresentação, voltada para o participante, ligada a um computador com o cliente ativo. Ela é analisada como as outras: linhas dos olhos, calibração dos alvos, fusão e calibração de posição (na página `/alvo?display=tv` ela vem selecionada por padrão). É opcional: sem ela, notebook e NAO continuam funcionando.

Em **Câmeras e sessão**, selecione a origem para cada papel e clique em **Aplicar câmeras**. A lista identifica computador e índice do dispositivo. Confira as três prévias. Uma mesma câmera não pode ocupar dois papéis. Trocar a configuração descarta a calibração anterior e é bloqueado durante gravação.

O navegador controla o experimento, mas não captura webcams remotas. Os conectores evitam a exigência de HTTPS do `getUserMedia` em páginas abertas pelo IP do servidor. O cliente mantém apenas o quadro mais recente enquanto transmite, evitando filas antigas quando a rede fica lenta.

## Calibrar a disposição atual pelo painel

O fluxo principal é **Calibrar nesta disposição**, sem depender das janelas de ponto. Informe o participante, leia a indicação, olhe para o alvo físico e clique em **Participante pronto · coletar**. Prepare o olhar antes do clique; o primeiro segundo de cada coleta é descartado. São nove fixações de ajuste (notebook, TV, cabeça e pernas do NAO e locais fora dos alvos) e seis de verificação, iniciada automaticamente após um ajuste válido. São 45 segundos de etapas, além do tempo de preparação e dos cliques. Se uma etapa não conseguir quatro leituras válidas em pelo menos uma câmera, ela permanece no mesmo alvo e pede uma nova coleta. Todos os pontos precisam ter amostras; concluir as etapas sem rosto não gera um modelo.

Calibre a montagem que está sendo testada. Ao mudar posição dos equipamentos, pessoa ou pose do NAO, refaça o ajuste. A fotografia do cenário não fornece coordenadas reais de olhar: são necessárias as fixações de referência da pessoa.

O painel mostra claramente quando não há modelo nem amostras e por que a leitura foi rejeitada (rosto ausente, pálpebras sem leitura, íris pequena ou quadro antigo).

## Calibração automática com pontos (opcional)

O botão **Iniciar automática com pontos** executa **9 fixações de ajuste + 6 de verificação independente**. São 45 segundos de etapas cronometradas, além das instruções e da preparação; reserve aproximadamente um minuto por pessoa. Leituras insuficientes ou janelas ocultas podem aumentar esse tempo. A duração e a precisão precisam ser conferidas no piloto com os participantes reais.

1. Informe o identificador da pessoa. Fixe cadeira, notebook, TV e a pose da cabeça do NAO.
2. Abra **Ponto no notebook** no computador do participante e **Ponto na TV** na apresentação. Se forem dois monitores do mesmo computador, use área de trabalho estendida. Arraste a janela da TV para o monitor correto. A página também pode ser aberta pelo endereço LAN, com o mesmo código de acesso.
3. Em cada janela, clique em **Exibir alvo em tela cheia**. As duas devem continuar visíveis; duas abas no mesmo monitor não substituem duas telas. Somente a janela do notebook fala as instruções. A voz pode ser desativada; nesse caso, o operador deve orientar o participante.
4. No painel, clique em **Calibração rápida guiada**. A sequência só coleta quando as duas telas confirmam a indicação e as câmeras Notebook/NAO têm imagem recente.
5. A pessoa acompanha os pontos amarelos no notebook e na TV. Para o robô, deve olhar o NAO físico: **cabeça, perna esquerda e perna direita**. A perna esquerda do NAO fica à direita de quem o vê de frente; o desenho destaca o lado correto. Os três pontos pertencem à classe `NAO`, com protótipos separados por fixação.
6. São duas fixações no notebook, duas na TV, três no NAO e duas fora dos alvos. Escolha antes uma referência física à esquerda e outra à direita fora das telas/robô. Cada etapa tem 3 segundos, com o primeiro segundo descartado para acomodar o olhar.
7. O sistema passa automaticamente à verificação, com pontos centrais nas telas, cabeça e duas pernas do NAO e uma referência fora dos alvos, em ordem aleatória. Essas amostras não ajustam o modelo. Cada etapa precisa de pelo menos cinco avaliações.
8. Se faltar leitura numa etapa de ajuste, ela é repetida. Após duas tentativas insuficientes, o sistema pausa para ajustar o enquadramento. Se uma janela desaparecer ou perder conexão, a coleta parcial é descartada e aguarda nova confirmação. **Pausar/Retomar** também reinicia a etapa atual.
9. Ao terminar, confira o resultado e volte à apresentação, saindo da tela cheia/fechando as janelas dos pontos. Depois da sessão, **Próximo participante** descarta o ajuste anterior e libera o campo para outra pessoa.

O modo manual permanece em **Ajuste manual / calibração detalhada**, com 12 etapas de ajuste e 8 de verificação, a 4 segundos por etapa (0,5 s descartados). Também é possível usar o perfil rápido manualmente. Use **Refazer uma etapa** para recolher um ponto e **Recalcular resultado**; reajustar o modelo invalida a verificação anterior.

As telas de ponto são `/alvo?display=notebook` e `/alvo?display=tv`. O servidor registra o perfil, os tempos e os alvos físicos da calibração em `calibration_protocol` no arquivo da sessão. As janelas enviam confirmação da indicação exibida, mas o software não identifica automaticamente em qual monitor físico foram colocadas; essa conferência é do operador.

O classificador usa protótipos separados por fixação, por câmera e por participante, com características de olhos e cabeça. Isso evita que os exemplos de “outro local” à esquerda e à direita sejam misturados num alvo artificial no centro. Distâncias altas, ambiguidade, discordância, olhos fechados e falta de imagem podem gerar `INDETERMINADO`.

A fusão aceita concordância de câmeras com escore mínimo 0,35, ou uma câmera com escore mínimo 0,60. O escore é heurístico, não probabilidade. Quadros com idade superior a 0,7 s, incerteza RTT acima de 0,15 s ou diferença superior a 0,25 s em relação à câmera mais recente não são usados na fusão. A outra câmera pode continuar sendo usada com o limiar de câmera única. A pose do NAO é lida por sensores: diferença maior que 0,08 rad em um eixo invalida aquela leitura. Movimento acima de 0,10 rad durante o ajuste exige repetir a calibração. Estes limites são parâmetros empíricos para o piloto, não limiares científicos validados.

A câmera convencional estima o alvo do olhar e a orientação da cabeça. Ela não comprova compreensão, atenção cognitiva ou diagnóstico. O sistema LAN não usa o detector legado de “distração cognitiva”. A precisão entre TV e notebook depende da geometria, enquadramento, iluminação e características do participante; avalie isso no piloto.

## Gravar e marcar eventos

Informe um identificador pseudonimizado e inicie a gravação. As três câmeras precisam ter imagem recente. O sistema exige ajuste e verificação concluídos. Se acurácia ou cobertura da verificação ficar abaixo de 65%, é necessário recalibrar ou preencher a justificativa de uso em piloto, registrada na sessão. Esse limite é apenas um controle operacional.

Os botões **Cena**, **Fala do NAO**, **Decisão** e **Sincronia** registram marcos com notas. Para verificar a sincronização, faça um gesto observável nas três câmeras no início e no fim. A sincronização por RTT alinha relógios, mas não corrige automaticamente buffers internos dos sensores, assimetria de rede ou exposição. Não há sincronização de hardware nem captura de áudio.

A apresentação pode registrar eventos com uma requisição autenticada. O instante corresponde ao recebimento no servidor:

```python
import requests
requests.post('http://IP_DO_SERVIDOR:8765/api/events',
    headers={'Authorization': 'Bearer CODIGO_DO_SERVIDOR'},
    json={'kind': 'INICIO_CENA', 'note': 'cena 03'}, timeout=5).raise_for_status()
```

Encerre pelo painel antes de sair. Os arquivos ficam em `sessions/ID_DA_SESSAO/`:

- `session.json`: participante, equipamentos, modelos, amostras, verificação e justificativa.
- `frames.jsonl`: tempo de cada quadro, chegada, incerteza e arquivo correspondente.
- `notebook/`, `nao/`, `validation/`: sequência JPEG, com tempo real preservado no índice. A reprodução no painel usa esses tempos, sem preencher lacunas como se fossem vídeo válido.
- `predictions.jsonl`: previsões aproximadamente a 5 Hz e resultados de cada câmera.
- `events.jsonl`: marcos do experimento.
- `annotations.json`: intervalos independentes por avaliador.
- `report.json`: último relatório calculado. As anotações dos demais avaliadores continuam preservadas.

O tamanho da gravação depende da resolução, taxa e duração; confira espaço livre antes da coleta. O padrão do cliente USB é até 1280×720 a 30 FPS (JPEG qualidade 90), igual ao módulo isolado; em rede lenta, reduza com `--width 960 --height 540 --fps 20`. Durante as calibrações o extrator processa todos os quadros (~30 Hz); fora delas, ~5 Hz. Na calibração de posição, amostras com EAR abaixo de 0,20 (piscadas, olhos semicerrados) são descartadas. O NAO usa VGA JPEG a até 15 FPS. O ritmo efetivo depende do hardware e da rede. Falhas de conexão ficam registradas como lacunas. Sessões abertas durante uma queda do servidor são marcadas como interrompidas na próxima inicialização. Não se retoma uma coleta automaticamente após reiniciar.

## Validação humana

Em **Validação humana**, selecione uma sessão encerrada. A perspectiva inicial é a da apresentação; você também pode inspecionar as outras imagens. A linha do tempo mostra o horário real e a diferença do quadro exibido. Se não houver imagem em até 0,5 s do instante escolhido, aparece uma lacuna.

Identifique o avaliador, anote intervalos não sobrepostos e salve antes de calcular os resultados. Use `INDETERMINADO` quando a gravação não permitir classificar. O sistema não mostra a previsão automática sobre o vídeo de anotação. Para outro avaliador, informe seu identificador e clique em **Carregar anotações** antes de editar.

O relatório compara previsão com anotação usando a duração de interseção em segundos, em vez de contar quadros com taxas diferentes. Inclui matriz de confusão, precisão, revocação, F1 por alvo, macro F1 dos alvos com suporte, acurácia e kappa. Predições indeterminadas contam como erros na acurácia. A cobertura informa quanto do tempo rotulado recebeu uma previsão determinada. A acurácia seletiva é mostrada separadamente. Trechos humanos indeterminados são excluídos das métricas e têm sua duração informada. Lacunas de previsão acima de 0,5 s viram indeterminadas; a última previsão não se estende indefinidamente. Não há pontuação perfeita quando faltam anotações.

Exporte as previsões CSV e o relatório JSON pelo painel. Para validação científica, use a verificação como diagnóstico do ajuste e a anotação humana de trechos independentes como avaliação; amostras próximas no tempo são correlacionadas. Esta versão calcula concordância sistema–avaliador, não concordância entre dois avaliadores.

## Linhas dos olhos nas prévias

O painel desenha uma seta aproximada saindo de cada íris nas câmeras Notebook e NAO. Quando o vetor fica próximo do eixo frontal, mostra anéis em vez de inventar uma seta longa no plano da imagem. O desenho é 2D, derivado das posições das íris e da orientação aproximada da cabeça; não calcula a interseção física do olhar com uma tela, nem substitui um rastreador ocular.

A cor vem da classificação calibrada e da fusão, não da direção aparente da seta. A referência padrão agora é **Tarefa**: notebook, TV e NAO ficam verdes; fora desses alvos fica vermelho. Em **Telas**, verde corresponde a `NOTEBOOK`/`TV`; vermelho a `NAO`/`OUTRO_LOCAL`. Em **Robô NAO**, apenas `NAO` fica verde. Em **Qualquer alvo da tarefa**, notebook, TV e NAO ficam verdes. A escolha muda somente a apresentação, sem reclassificar ou alterar os registros do experimento.

A região da tarefa é estimada separadamente do alvo exato, usando a distância aos protótipos de dentro/fora. Ambiguidade entre TV, notebook e NAO pode manter o alvo exato indeterminado e ainda produzir verde para a tarefa, desde que haja margem suficiente contra os exemplos de fora. Discordância entre dentro e fora continua indeterminada. Os resultados de região e alvo exato são preservados separadamente nos registros. A verificação mostra as duas métricas; o controle de início de gravação mantém a regra original para a distinção entre alvos exatos.

A cor nas duas prévias representa a fusão para a mesma pessoa. Uma perspectiva auxiliar com os olhos visíveis pode usar a classificação da outra câmera, mesmo que não consiga distinguir todos os alvos sozinha; isso não constitui uma medição independente daquela perspectiva. Sem olhos visíveis, sem modelo ou sem evidência de região, fica cinza. A direção desenhada continua aproximada e pode ser omitida quando a qualidade dos olhos for insuficiente.

Antes da calibração, em caso de conflito entre tarefa e fora, perda de sinal ou evidência insuficiente, a indicação fica cinza. A geometria só é usada quando a observação tem até 0,35 s de idade e diferença de até 0,30 s do quadro exibido. O navegador também limpa indicações antigas quando não chegam novos quadros. As coordenadas respeitam a proporção da imagem e as barras da prévia. O desenho usa os resultados do extrator (~5 Hz), enquanto a prévia mantém a taxa de transmissão da câmera.

Use **Mostrar linhas dos olhos** para ativar/desativar. O processamento dos marcos faciais fica no servidor e o navegador desenha a sobreposição; os JPEG gravados continuam sendo os quadros originais. As três câmeras recebem a sobreposição.

## Testes

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check web\app.js
node --test tests\ray-overlay.test.cjs
```

Os testes cobrem classificação, ambiguidade, câmeras parciais, movimento do NAO, expiração de leitura, separação ajuste/verificação, sincronização por RTT, acesso à API, ingestão JPEG/RGB, anotações, métricas com lacunas e o fluxo de gravação/revisão/exportação com imagens sintéticas. O conector NAO requer confirmação de compatibilidade com a versão real de NAOqi/SDK e teste físico. Testes sintéticos não estimam a precisão do estudo.

## Execução local anterior (uma webcam)

O ponto de entrada `main.py` foi mantido para compatibilidade. O fluxo multicâmera descrito acima usa `server.py`.

## Execução

```powershell
.venv\Scripts\python.exe main.py --participant-id P001
```

A execução abre uma prévia da webcam. Posicione o rosto até aparecer a indicação de detecção e pressione `Espaço` para iniciar a calibração de cinco pontos, com duração de 3 segundos por ponto. Se faltarem leituras válidas, a prévia permite tentar novamente com `Espaço`. Os primeiros 0,5 segundos de cada fixação são descartados. Em seguida, uma janela de diagnóstico apresenta um dos seguintes estados: `EM_FOCO`, `DISTRACTION_GAZE`, `DISTRACTION_COGNITIVE` ou `SLEEPY`.

Para encerrar a execução, pressione `Q` ou `Esc`. Caso exista algum evento em aberto, ele será finalizado automaticamente.

Os arquivos são gravados continuamente no diretório `monitoring_logs/`, nos formatos CSV e JSON. Recomenda-se utilizar um identificador pseudonimizado para cada participante.

### Opções importantes

- `--camera 1`: seleciona outra webcam.
- `--aoi 0.1,0.1,0.9,0.9`: delimita a área de interesse (coordenadas normalizadas).
- `--no-preview`: realiza o monitoramento sem manter a janela de visualização da webcam após a calibração.
- `--cognitive-threshold 0.0004`: ajusta o limiar empírico de variância.
- `--output caminho`: altera o diretório de armazenamento dos registros.

O sistema solicita uma resolução de 1280 × 720 pixels e uma taxa de 30 FPS, além de exibir a latência por quadro na janela de visualização. A taxa efetiva depende das características da webcam e do computador utilizado. A distância recomendada entre o participante e a câmera é de 60 a 75 cm.



O conector NAO usa JPEG com qualidade 80 e solicita 15 FPS por padrão, mantendo 640×480. PIL/Pillow é necessário no robô (já disponível no NAO configurado). `--raw-rgb` restaura o envio sem compressão; `--jpeg-quality 80` ajusta a qualidade entre 60 e 95. A captura ocorre em uma thread separada do envio, com apenas o quadro mais recente em memória. Em teste no NAO 192.168.17.62, foram medidos 14,9–15 FPS no servidor. A taxa pode variar com carga e rede. A identidade persistida pelo script evita perder a seleção ao reiniciar o conector.
