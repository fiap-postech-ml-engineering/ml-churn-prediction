# ML Canvas
Neste documento trouxemos um levantamento de informações focado nos conceitos de _"Bussiness Undestanding"_ e _"Data Understanding"_

## Qual problema queremos resolver?
Desacelerar a perda de clientes da empresa através de métricas que direcionem a atenção dos stakeholders para clientes com padrões e comportamentos que indicam um possível cancelamento (Churn)

## Métrica que consideraremos para a resolução do problema
- **Churn:** Quantidade de clientes que tiveram o cancelamento efetivado através de solicitação própria, comparado com a base total
$$
churn = \frac{clientes\ que\ cancelaram}{total\ de\ clientes}
$$

## Qual a métrica de negócio esperamos ser afetada?
**Faturamento mensal.** Queremos diminuir o impacto dos cancelamentos no faturamento final trazendo mais previsibilidade para o negócio

## O que queremos atingir
Com um modelo com **Recall no Top 20% de pelo menos 65%** para priorizar clientes com maior risco de cancelamento esperamos reduzir o churn mensal em 10 p.p.
- **Top 20:** Clientes com maior risco
- **65%:** Percentual de churners reais capturados dentro desse Top 20% 


## Recursos necessários
- **Base de dados** para treinamento dos modelos cruzando os dados analíticos do negócio a nível de usuário com informações demográficas e de comportamento do usuário
- **Apoio das equipes** de Marketing, CRM e Negócios para que possamos traduzir o comportamento dos usuários em dados assertivos
- **Infraestrutura** em nuvem para CI/CD/CT do modelo e hospedagem da API onde os dados vão ser servidos


## Dados e variáveis relevantes

