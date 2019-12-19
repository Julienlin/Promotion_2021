# Optimisation Par Colonie De Fourmi D'un Chemin dans un labyrinthe

Deux types de projets : un où on répartit les fourmis par processus, l'autre où on répartit le labyrinthe parmi les processus.


## Configuration, Compilation

Installer freeglut3 : 

- soit (si on est root de sa machine) avec :

    sudo apt-get install freeglut3-dev

- soit en local avec : 

    make thirdparty



Copier `Make.inc.[gnu ou osx]` dans `Make.inc`, puis
modifier la ligne `PROJECT_ROOT` du fichier `Make.inc` : comme son nom l'indique, elle doit contenir le nom du répertoire de base du projet (`pwd`).
Éventuellement ajuster les autres variables de ce `Make.inc` si besoin ou envie.



Compiler le projet : 

    make all



Tester le programme : `cd test ; ./ant_simu.exe`
