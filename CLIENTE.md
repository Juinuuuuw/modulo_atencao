# Conectar câmeras pela rede local

## Webcam do notebook e câmera traseira USB

1. Conecte o computador à mesma rede do servidor.
2. Extraia o ZIP de conectores baixado no painel.
3. Com Python 3 instalado, execute `iniciar_cliente.bat`.
4. Informe `http://IP_DO_SERVIDOR:8765` e o código exibido pelo servidor.
5. No navegador, abra esse mesmo endereço, entre e associe as câmeras aos papéis.

O cliente procura os índices 0 a 3 e só mantém abertas as webcams selecionadas no painel. Para procurar mais dispositivos, use `--scan 8`. Se o driver não funcionar, tente `--backend msmf`. O nome apresentado contém o nome do computador e o índice da câmera; confira visualmente a prévia.

Execução manual:

```powershell
python -m venv .venv-client
.venv-client\Scripts\python.exe -m pip install -r requirements-client.txt
.venv-client\Scripts\python.exe camera_client.py --server http://IP_DO_SERVIDOR:8765 --name Notebook-participante
```

Para o computador servidor, execute o mesmo cliente usando `--server http://127.0.0.1:8765 --name Servidor`. Não é necessário instalar MediaPipe no cliente.

O código de acesso é solicitado sem eco no terminal. Também pode ser fornecido pela variável de ambiente `MONITOR_ACCESS_KEY`. Não coloque o código em arquivos compartilhados.

## Câmera interna do NAO

Use `nao_client.py` no próprio robô, com Python 2.7 e `naoqi` disponíveis:

```text
python nao_client.py --server http://IP_DO_SERVIDOR:8765
```

Ou execute em uma máquina com o SDK NAOqi compatível configurado:

```text
python nao_client.py --server http://IP_DO_SERVIDOR:8765 --robot IP_DO_NAO
```

Selecione a câmera superior no painel. A inferior também é oferecida, mas somente uma câmera interna é ativada por robô. O conector usa `ALVideoDevice.subscribeCamera`, VGA com envio JPEG (qualidade 80), e lê `ALMotion.getAngles` sem movimentar o robô. O servidor recebe as imagens e executa a análise. O SDK e a conectividade com um NAO real precisam ser verificados no ambiente do estudo.

Mantenha a cabeça do NAO fixa após a calibração. Caso o protocolo exija gestos de cabeça, será necessário ampliar o modelo para as poses usadas; esta versão rejeita a leitura da câmera quando a pose difere da referência em mais de 0,08 rad (~4,6°).

## Rede e horários

HTTP é indicado apenas para uma rede local controlada: não cifra imagens nem código. Use HTTPS por proxy quando precisar de transporte cifrado. Se o Windows solicitar liberação de rede para o Python, o responsável pelo computador deve configurar o acesso na rede privada. Não é preciso alterar configurações de segurança do navegador.

Os conectores medem o tempo de ida e volta ao servidor e renovam o ajuste de relógio a cada 30 segundos. O horário representa a conclusão da leitura do quadro, não a exposição física do sensor. A incerteza informada é uma estimativa por RTT, que não inclui todo o atraso interno da câmera. Use um gesto visível no início e no fim para conferir alinhamento e deriva. Não há captura de áudio.

Encerre o cliente com Ctrl+C. Se reconectá-lo como um novo cliente, reaplique as câmeras e recalibre antes de outra sessão.


## Robô 192.168.17.62 já configurado

Foi instalado o conector em `/home/nao/observa`. O script carrega os caminhos do SDK de `/opt/aldebaran` sem alterar o ambiente global do robô. O código de acesso ao servidor está em `.access-key`, dentro da pasta privada do conector, com permissão 600. Esse arquivo não acompanha o ZIP.

Após ligar o servidor, conecte pelo computador:

```powershell
ssh nao@192.168.17.62
```

No terminal do robô, inicie o conector quando ele ainda não estiver rodando:

```sh
~/observa/iniciar_nao.sh
```

O servidor padrão é `http://192.168.16.162:8765`. Para outro endereço:

```sh
~/observa/iniciar_nao.sh http://NOVO_IP:8765
```

O processo iniciado manualmente fica no terminal; Ctrl+C encerra a captura. Depois de iniciar, no painel do servidor escolha `NAO / câmera superior` e aplique. A inicialização automática após ligar o robô não foi configurada. Não inicie duas cópias do conector ao mesmo tempo.

Na instalação desta conversa, o conector já foi iniciado em segundo plano. Seu PID está em `/home/nao/observa/observa.pid`; o log está em `/home/nao/observa/observa.log`. A conexão pelo SDK é feita internamente em `127.0.0.1:9559`, pois a porta 9559 externa desse robô não está acessível pelo computador.


O conector NAO usa JPEG com qualidade 80 e solicita 15 FPS por padrão, mantendo 640×480. PIL/Pillow é necessário no robô (já disponível no NAO configurado). `--raw-rgb` restaura o envio sem compressão; `--jpeg-quality 80` ajusta a qualidade entre 60 e 95. A captura ocorre em uma thread separada do envio, com apenas o quadro mais recente em memória. Em teste no NAO 192.168.17.62, foram medidos 14,9–15 FPS no servidor. A taxa pode variar com carga e rede. A identidade persistida pelo script evita perder a seleção ao reiniciar o conector.
