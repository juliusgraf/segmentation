**Compte-rendu de la réunion du 12 décembre 2023**

Lors de la réunion, les points suivants ont été reléves, et seront résolus prochainement.
- Il faut reécrire la perte du réseau $\mathcal L(\varphi (Y), \hat Y)$ plus clairement, sans la notation du batch de taille $B$.
- Lors de la soutenance, il faudra plus clairement savoir expliquer ce qui est relatif à la classification, et ce qui est relatif à la segmentation. En particulier, le fait que l'image soit représentée en HSV sert uniquement à la visualisation.
- Appliquer un $\mathrm{softmax}$ à l'image de sortie du réseau au lieu de l'application sigmoïde $\sigma$ (plutôt utilisée pour les problèmes de classification binaire) afin de pouvoir comparer les sorties, ce qui permet d'examiner plus facilement les comportements de la constante de Lipschitz.
- Lors de la soutenance, il faudra être plus clair sur la régularisation, c'est-à-dire comment elle est implémentée.
- Transposer le code écrit sur $\verb|Colab|$ vers un dossier utilisable pour le DCE Metz, puis refaire les simulations jusqu'au bout afin de pouvoir comparer avec les autres approches.
- Chercher à optimiser le poids accordé à la régularisation, c'est-à-dire $\lambda$ afin d'étudier comment $\lambda$ influe sur les performances du modèle, mesurer l'impact de la régularisation.
- Partager avec M. Terris le code écrit pour U-Net et Carvana et COCO afin de faire les simulations pour des images non rabattues à 32*48 pixels. Ensuite, réaliser des simulations sur un nombre d'époques plus importants (pour COCO).
- Ici, ajouter la mesure de la constante de Lipschitz au fur et à mesure des époques, afin de pouvoir ici aussi jouer sur le paramètre $\lambda$ et mesurer l'impact sur les performances du modèle.
- Enfin, pour COCO, revenir à un problème de classification multi-classe (contrairement à la classification binaire de Carvana), ce qui va nécessiter de revoir la fonction créant les masques à partir des segmentations.
- Ensuite, comparer les résultats de classification multi-classes entre U-Net sur COCO et ResNet-18 sur CIFAR-10 et tirer des conclusions sur l'impact de la régularisation lipschitzienne.
- Pour la soutenance, présenter clairement l'objectif du problème de manière à réunir les différentes approches utilisées.
- Aussi, penser à normaliser les images d'entrée comme pour CIFAR-10 afin de pouvoir comparer les résultats obtenus sur la régularisation.
