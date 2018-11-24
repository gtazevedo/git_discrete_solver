##Log

_Note: This log will be in portuguese since its for my own control of what I'm doing. If you are reading this repository, you don't really need to read or try to understand this_

Deseja-se fazer:
- [ ] Testar o RMSPropOptimizer como otimizador
- [ ] Testar learning rate decrescente
- [ ] Tentar fazer um colormap de episódios x steps
- [ ] Separar as funções em arquivos diferentes
- [ ] Isolar as funções de gráfico
- [ ] Implementar este código para visão computacional por análise de pixels(?)
- [ ] Para o gráfico implementar dois vetores, um que adiciona 0 para valores antes do treinamento e outro que adiciona 0 para valores depois do treinamento e ambos não plotam valores zerados. Outra opção é fazer esta diferença para o caso de objetivo atingido ou não
- [ ] Rodar o CartPole-v0 com os pesos salvos do CartPole-v1 e vice versa e observar o desempenho
- [ ] Adicionar um gráfico com a pontuação média dos ultimos 100 episódios. Fazer isso criando um deque de 100 unidades

**As pontuações buscadas em cada um dos ambientes são mostradas na tabela abaixo**
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>200 pontos (máxima pontuação permitida)</td>
      <td>500 pontos (máxima pontuação permitida)</td>
			<td>Qualquer valor acima de -110 pontos (Pontuação que simboliza que o veículo atingiu o ponto desejado)</td>
			<td>Qualquer valor acima de -80 pontos (Obtenção de uma capacidade considerável de equilibrar o braço)</td>
			<td>200 pontos (máxima pontuação permitida)</td>
    </tr>
  </tbody>
</table>

**Este é um log dos casos simulados para controle das simulações. O padrão é a utilização do Otimizador Adam com lr = 0.001 e gamma =0.99, simulações com valores fora do padrão terão esta informação destacada, caso contrário, as mesmas serão omitidas.**

1. Para o caso simulado utilizando uma HL de 24 com 1000 episodios, utilizando recompensa clipada e e 1k de memória de replay. Este caso ainda utilizava uma única DQN
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apresentou muitas oscilações de pontuação mas solucionou o ambiente</td>
      <td>Apresentou muitas oscilações de pontuação mas solucionou o ambiente</td>
			<td>Falhou em solucionar o ambiente, não atingiu o objetivo nenhuma vez</td>
			<td>Não simulado</td>
			<td>Oscilou muito mas conseguiu alguma pontuação positiva, mas insuficiente para solucionar o ambiente</td>
    </tr>
  </tbody>
</table>

2. Para o caso simulado utilizando uma HL de 50 com 1000 episodios, utilizando recompensa clipada e e 1k de memória de replay. Este caso ainda utilizava uma única DQN
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apresentou muitas oscilações de pontuação, solucionou o ambiente, mas quando parecia ter aprendido, se perdeu</td>
      <td>Conseguiu solucionar o ambiente a partir de 250 episódios, mais rápido que com HL 24</td>
			<td>Em torno de 200 episódios parecia ter aprendido a politica mas depois começou a oscilar e se perder</td>
			<td>Não simulado</td>
			<td>Não recebeu punições absurdas mas ainda incapaz de solucionar o ambiente</td>
    </tr>
  </tbody>
</table>

3. Para o caso simulado utilizando uma HL de 50 com 1000 episodios, sem recompensa clipada e 1k de memória de replay. Este caso ainda utilizava uma única DQN
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apresentou muitas oscilações de pontuação, solucionou o ambiente, mas quando parecia ter aprendido, se perdeu, desempenho muito similar ao com recompensa clipada, porém parece apresentar menor estabilidade e maior tempo para aprender</td>
      <td>O caso com recompensa clipada parece ter obtido um desempenho mais satisfatório. Este apresentou vários picos de high score antes de ser capaz de realizar várias pontuações elevadas consecutivamente</td>
			<td>Em torno de 200 episódios parecia ter aprendido a politica mas depois começou a oscilar e se perder</td>
			<td>Não simulado</td>
			<td>Sem o clipping os resultados apresentados parecem similares mas houve um pico de melhor desempenho significativo sem o clip</td>
    </tr>
  </tbody>
</table>
*Este experimento fez com que fosse adotado o clipping de recompensa*

4. Para o caso simulado utilizando uma HL de 50 com 1k de episódios e 10k de memória de replay e com treinamento iniciado após 4k de memória ser preenchida. Análise realizada apenas para o CartPole-v0
<table>
  <thead>
    <tr>
      <th>Adam com lr 0.001</th>
      <th>Adamax com lr 0.001</th>
      <th>Adamax com lr 0.002</th>
			<th>Nadam com lr 0.001</th>
			<th>Nadam com lr 0.002</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apresentou muitas oscilações até próximo de 600 episódios e próximo ao final (em torno do episódio 900) apresentou pontuação apresentou pontuação de 20, porém, a partir do episódio 50 já havia começado a apresentar pontuação de 200</td>
      <td>Atingiu 200 pontos pela primeira vez depois do episódio 500 e depois disso oscilou bastante, não apresentando nenhuma continuidade de pontuação máxima como o Adam</td>
			<td>Começou a atingir 200 pontos depois do episódio 400. Apresentou grande estabilidade mas por volta do episódio 950 quando parecia ter aprendido bem a política teve um vale de baixa pontuação</td>
			<td>Começou a atingir os 200 pontos depois do episódio 400 mas apresentou vários vales longos de baixa pontuação</td>
			<td>Apresentou grande instabilidade e começou a atingir 200 pontos depois do episódio 400</td>
    </tr>
  </tbody>
</table>

5. Para o caso simulado utilizando uma HL de 50 com Adam e lr 0.001 para 1000 episódios com 10k de memória de replay e 1000 de memória para se iniciar o treinamento.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Oscilou bastante, entre o episódio 500 e 600 teve desempenho perfeito mas depois começou a perder desempenho novamente</td>
      <td>Oscilou bastante</td>
			<td>Oscilou muito. Desempenho insatisfatório.</td>
			<td>Não simulado</td>
			<td>Oscilou muito mas depois do episódio 200 as recompensas negativas diminuiram muito embora não tenha solucionado o ambiente</td>
    </tr>
  </tbody>
</table>

6. Para o caso simulado utilizando uma HL de 50 com Adam e lr 0.001 para 1000 episódios com 10k de memória de replay e 4000 de memória para se iniciar o treinamento.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apresentou maior estabilidade quando comparado ao caso anterior</td>
      <td>Sem diferenças significativas, se mostra necessário mais episódios para uma melhor análise</td>
			<td>O ambiente parece que está começando a ser resolvido</td>
			<td>Não simulado</td>
			<td>Melhora de desempenho</td>
    </tr>
  </tbody>
</table>

7. Para o caso simulado utilizando uma HL de 50 com Adam e lr 0.001 para 2000 episódios com 10k e treinamento iniciado após o uso de 5k de memória
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Oscilou bastante mas depois de 1k parecia ter tido uma ótima estabilização, mas então voltou a oscilar.</td>
      <td>O ambiente foi solucionado e teve bons resultados mas a instabilidade ainda está elevada</td>
			<td>Teve muitas oscilações mas solucionou o ambiente diversas vezes embora tenha apresentado vales de falhas, aparenta ser o caminho para a solução</td>
			<td>Não simulado</td>
			<td>O desempenho está melhorando mas o ambiente ainda não foi solucionado</td>
    </tr>
  </tbody>
</table>

8. Para o caso simulado utilizando uma HL de 50 com recompensa clipada com 2k de episódios e 10k de memória com treinamento iniciado após 5k de memória ser preenchido
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Conseguiu atingir o objetivo mas ainda apresenta alguma instabilidade, que parece melhorar significativamente depois do episódio 1750. Desempenho melhor que o CartPole-v1</td>
      <td>Conseguiu atingir o objetivo mas a instabilidade ainda está elevada, aparenta ter melhorado bastante depois do episódio 1900</td>
			<td>Parece ter aprendido a politica, não teve uma pontuação constante mas depois do episódio 1900 não deixou mais de atingir o objetivo. Todos seus pontos foram acima de > -100 e depois do episódio 500 deixou de apresentar falhas</td>
			<td>O desempenho está bom mas aparenta ser um pouco abaixo do desejado, é muito instável, no invervalo de 500 a 750 episódios apresentou o melhor desempenho, que então apresenta queda de pontuação, pequena mas significativa, que simboliza solucionar ou não o ambiente</td>
			<td>Não foi nem capaz de se aproximar dos 200 pontos, apenas adicionar mais episódios não parece ser capaz de solucionar este caso</td>
    </tr>
  </tbody>
</table>

9. Para o caso simulado utilizando uma HL de 50 com recompensa clipada com 2k de episódios e treinamento iniciado após 1k de memória ser utilizado com 10k de memória máxima
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Da mesma forma que a versão 1 solucionou o ambiente mais rapidamente (devido ao menor tempo de observação) mas ao custo de uma mais baixa estabilidade</td>
      <td>O ambiente foi solucionado mais rapidamente mas apresentou uma estabilidade muito menor</td>
			<td>Parece ter aprendido a politica, não teve uma pontuação constante mas depois do episódio 1900 não deixou mais de atingir o objetivo. Todos seus pontos foram acima de > -100 e depois do episódio 500 deixou de apresentar falhas</td>
			<td>Conseguiu solucionar o ambiente mas obteve uma instabilidade muito alta. Apesar de alcançar o objetivo mais rapidamente</td>
			<td>Não conseguiu solucionar o ambiente mas os resultados aparentam ser melhores, mas novamente muito instáveis. Ainda assim continua muito abaixo do desejado</td>
    </tr>
  </tbody>
</table>
No geral, iniciar o treinamento após a memória atingir 5k aparenta aumentar a estabilidade de maneira significativa. Nenhum dos dois testes foram capazes de trazer resultados significativos para o LunarLander-v2

10. Para o caso simulado utilizando otimizador Adamax com lr 0.002, uma HL de 50 com recompensa clipada com 2k de episódios e treinamento iniciado após 1k de memória ser utilizado com 10k de memória máxima
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Bons resultados, instabilidade razoável mas quando ocorre não ocorre em vales e a queda de pontuação a leva para algo em torno de 100 pontos</td>
      <td>Alta instabilidade mas soluciona o ambiente</td>
			<td>Alta instabilidade, foi capaz de resolver o ambiente, mas a maioria dos pontos em todos os momentos estão na pontuação mínima</td>
			<td>Alta instabilidade, não foi capaz de solucionar o ambiente</td>
			<td>Desempenho padrão, pontuação ao redor de 0 bem instável</td>
    </tr>
  </tbody>
</table>

11. Para o caso simulado utilizando duas HL de 24 e recompensa clipada com 10k de memória máxima e treinamento iniciado em 5k de memória  com 2k de episódios.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Atingiu o objetivo e apresentou estabilidade crescente, desempenho significativamente bom</td>
      <td>Atingiu o objetivo mas apresentou uma instabilidade considerável</td>
			<td>Apresentou instabilidade considerável, porém, foi capaz de solucionar o ambiente</td>
			<td>Instabilidade considerável, pode-se considerar que solucionou o ambiente apesar da queda de desemepnho apresentada no final</td>
			<td>Desempenho ruim</td>
    </tr>
  </tbody>
</table>

12. Para o caso simulado utilizando duas HL de 24 e lr 0.0001 e gamma = 0.999 com recompensa clipada, 10k de memória máxima e treinamento iniciado em 5k de memória para 2k de episódios.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Instabilidade significativa mas depois de 1500 episódios apenas fez max score</td>
      <td>Atingiu a pontuação máxima por um tempo, mas a queda de pontuação sem recuperação leva ao questionamento do aprendizado</td>
			<td>Não aprendeu nada, permaneceu em -200 pontos</td>
			<td>Se aproximou muito do objetivo mas foi incapaz de alcançá-lo</td>
			<td>Desempenho muito próximo do 0 pontos. Insatisfatório</td>
    </tr>
  </tbody>
</table>

13. Para o caso simulado utilizando duas HL, uma de 128 e outra de 64 nesta respectiva ordem, com 50k de memória de replay e treinamento iniciado em 10k de memória, com 2,5k de episódios
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Solucionou o problema mas com uma instabilidade considerável</td>
      <td>Solucionou o problema mas com elevada instabilidade</td>
			<td>Teve uma certa instabilidade mas na maioria dos casos conseguiu fazer mais de -110 pontos. Desempenho pode ser considerado bom</td>
			<td>Apresentou certa instabilidade mas na maioria das vezes solucionou o ambiente, e diferente do CartPole, a istabilidade se concentrou ao redor da solução</td>
			<td>Foi o caso que até o momento desta simulação, mais se aproximou da solução do problema</td>
    </tr>
  </tbody>
</table>

14. Para o caso simulado utilizando duas Hl de 50, com recompensa clipada e uma recompensa bonus caso o objetivo proposto seja atingido para 2,5k de episódios com 50 de memória de replay com treinamento iniciado em 10k de memória.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Teve um bom desempenho, solucionou o ambiente e apresentou instabilidade decrescente</td>
      <td>Solucionou o ambiente com certa instabilidade que aparenta ser decrescente</td>
			<td>Instabilidade de solução muito elevada apesar da pontuação desejada ter sido alcançada</td>
			<td>O ambiente foi solucionada com sucesso, porém apresentou algumas oscilações abaixo do desejado, mas próximo</td>
			<td>Foi o caso que até o momento desta simulação, mais se aproximou da solução do problema</td>
    </tr>
  </tbody>
</table>

15. Para o caso de uma HL com 50, 2,5k episódios com 50k de memória e treinamento iniciado em 5k com recompensa clipada e bonus por se alcançar o objetivo
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>As oscilações e a instabilidade são muito menores que para o CarPole-v1, porém, ainda é existente</td>
      <td>Solucionou o ambiente mas com muitas oscilações e grande instabilidade</td>
			<td>Solucionou o ambiente mas teve muitas oscilações grandes e com grande frequência</td>
			<td>Solucionou o ambiente porém ficou oscilando ao redor da solução. Desempenho pode ser considerado ok mas foi inferior ao enterior com duas HL</td>
			<td>Ainda incapaz de solucionar o objetivo mas a pontuação média está acima de 100</td>
    </tr>
  </tbody>
</table>

16. Para o caso do otimizador Adam com duas HL de 24 e 2,5k episodios com 50k de memória e treinaento inciado em 10k de memória.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Solucionou o ambiente com poucas oscilações que aparentam ser decrescentes ao longo do tempo</td>
      <td>Muitas oscilações de grande amplitude e que ocorrem muito frequentemente</td>
			<td>Conseguiu solucionar o ambiente porém apresentou uma quantidade muito elevada de oscilações</td>
			<td>Conseguiu solucionar o ambiente, mas oscila com frequência próxima do resultado desejado, embora estas oscilações de pontuação sejam de baixa amplitude</td>
			<td>Desempenho instável, incapaz de solucionar o ambiente, pontuação oscilante próximo ao 0</td>
    </tr>
  </tbody>
</table>

17. Para o caso do otimizador Adam com uma HL de 50 e 2,5k episodios com 50k de memória e treinaento inciado em 10k de memória.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Conseguiu solucionar o ambiente, mas quando parecia estar apresentando uma estabilidade crescente, teve uma queda de desempenho considerável</td>
      <td>Conseguiu apresentar ponto de solução do ambiente, mas com alta instabilidade</td>
			<td>Conseguiu apresentar pontos de solução do ambiente, mas apresenta alta instabilidade</td>
			<td>Oscilou com muita frequência em torno do objetivo estabelecido</td>
			<td>Conseguiu atingir os 200 pontos uma única vez e então apresentou uma instabilidade muito grande entre 0 a 150 pontos</td>
    </tr>
  </tbody>
</table>

18. Para o caso do otimizador Adam com duas HL de 50 e 3k episodios com a memória de replay dada pela expressão 40k * action e o treinaento inciado em action * 5k de memória.
<table>
  <thead>
    <tr>
      <th>CartPole-v0</th>
      <th>CartPole-v1</th>
      <th>MountainCar-v0</th>
			<th>Acrobot-v1</th>
			<th>LunarLander-v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Conseguiu solucionar o ambiente e apresentou uma estabilidade crescente com o número de episódios</td>
      <td>Apresentou grande instabilidade para solucionar o ambiente</td>
			<td>Conseguiu solucionar o ambiente mas apresentou certa instabilidade</td>
			<td>Oscilou com determinada frequência ao redor da pontuação desejada</td>
			<td>Inicialmente parecia estar aprendendo sobre o ambiente, mas então começou a rapidamente ter uma queda de desempenho e terminou com uma pontuação negativa</td>
    </tr>
  </tbody>
</table>

18. Para o caso do otimizador Adam com duas HL de 50 e 3k episodios com a memória de replay dada pela expressão 40k * action e o treinaento inciado em action * 5k de memória. Foi variado o lr para o LunarLander v2.
<table>
  <thead>
    <tr>
      <th>rl = 5e-5</th>
      <th>rl = 0.0001</th>
      <th>rl = 0.0001 e gamma = 0.999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Teve um desempenho crescente, aprendeu a pousar a nave, mas não da forma desejada. Não foi capaz de atingir 200 pontos ou mais uma única vez</td>
      <td>Atingiu 200 pontos ou mais algumas vezes, mas não conseguiu manter esta pontuação, aprendeu a pousar a nave mas não da forma desejada</td>
			<td>Péssimo desempenho</td>
    </tr>
  </tbody>
</table>


