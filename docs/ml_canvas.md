# ML Canvas

Neste documento trouxemos um levantamento de informações focado nos conceitos de _"Bussiness Undestanding"_ e _"Data Understanding"_

## Qual problema queremos resolver?

Desacelerar a perda de clientes da empresa através de métricas que direcionem a atenção dos stakeholders para clientes com padrões e comportamentos que indicam um possível cancelamento (Churn)

## Métrica do negócio que consideraremos para a resolução do problema

- **Weighted Recall:** A proporção de CLTV de churners recuperada com o modelo

  $
  \text{Weighted Recall} = \frac{\sum_{i \in TP} CLTV_i}{\sum_{i \in \text{churners reais}} CLTV_i}
  $

## Qual a métrica de negócio esperamos ser afetada?

**CLTV** - Queremos recuperar a maior quantidade de valor de CLTV possível

## O que queremos atingir

Esperamos atingir uma recuperação de +80% do CLTV total dos churners nos testes, exemplo:
- De uma lista de 1000 churners, temos um CLTV somado de R$100k
- O objetivo é recuperar em valor de CLTV o equivalente a R$80k desses churners, independente da quantidade de churners 

## Recursos necessários

- **Base de dados** para treinamento dos modelos cruzando os dados analíticos do negócio a nível de usuário com informações demográficas e de comportamento do usuário
- **Apoio das equipes** de Marketing, CRM e Negócios para que possamos traduzir o comportamento dos usuários em dados assertivos
- **Infraestrutura** em nuvem para CI/CD/CT do modelo e hospedagem da API onde os dados vão ser servidos

## Dados e variáveis relevantes
- **CLTV:** - O quanto o cliente vale para a empresa em todo o seu ciclo de vida
- **Tenure Months** - Total de meses em que a pessoa é cliente da empresa
- **Monthly Charge** - Valor mensal cobrado do cliente somando todos os serviçoes
- **Total Charges** - Valor total acumulado já cobrado ao cliente
- **Aderência aos Serviços** - Quantos serviços o cliente tem contratado e quais são eles?
