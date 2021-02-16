# Testing Mediation Effects Using LOGic of BooleAN Matrices (LOGAN)

This repository contains the implementation for the paper ["Testing Mediation Effects Using Logic of Boolean Matrices"](https://arxiv.org/abs/2005.04584) in Python.

## Summary of the paper

Mediation analysis is becoming an increasingly important tool in scientific studies. A central question in high-dimensional mediation analysis is to infer the significance of individual mediators. The main challenge is the sheer number of possible paths that go through all combinations of mediators. Most existing mediation inference solutions either explicitly impose that the mediators are conditionally independent given the exposure, or ignore any potential directed paths among the mediators. In this article, we propose a novel hypothesis testing procedure to evaluate individual mediation effects, while taking into account potential interactions among the mediators. Our proposal thus fills a crucial gap, and greatly extends the scope of existing mediation tests. Our key idea is to construct the test statistic using the logic of Boolean matrices, which enables us to establish the proper limiting distribution under the null hypothesis. We further employ screening, data splitting, and decorrelated estimation to reduce the bias and increase the power of the test. We show our test can control both the size
and false discovery rate asymptotically, and the power of the test approaches one, meanwhile allowing the number of mediators to diverge to infinity with the sample
size. We demonstrate the efficacy of our method through both simulations and a neuroimaging study of Alzheimer’s disease.

<img align="center" src="ERRAB.png" alt="drawing" width="600">

**Figure**: Empirical rejection rate and ROC curve of the proposed test, LOGAN, and the test of Chakrabortty et al. (2018), MIDA, when the dimension of the mediators d = 50. The upper panels: the sample size n = 100, and the bottom panels: n = 200. The left panels: under $H_0$, the middles panels: under $H_1$, where the horizontal axis is the mediator index, and the right panels: the average ROC curve. 
