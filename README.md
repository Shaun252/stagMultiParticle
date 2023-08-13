# Tools for staggered multi-particle states 

Tools I developed and used for the work contained in my [thesis](To be added) on staggered two-pion states. It is primarily based off the work in this [paper](https://www.sciencedirect.com/science/article/abs/pii/0550321387902859) .

### Module Descriptions

###### gfunctions.py.py 

Set of tools for performing standard group theory operations. The functions operate on python dictionaries with keys labelling the group elements and the corresponding values being the represenatation matrix or characters. Examples of the functionality: restrict irreps to subgroups; compute conjucacy classes & associated characters; check orthogonality between irreps; check irreducibility; generate cosets under a subgroup; compute direct products and decompose into irreps.

###### translationGroup.py

Generates the lattice translation group, $T$, and its representations.

###### W3RotationGroup.py 

Generates the regular and projective representations of the cubic group W3.

###### naiveGroup.py

Generates the bosonic representations of the group $T \rtimes W3$, the (non-staggered) lattice group. Irreps are saved using one representative element from each conjucacy class + one representative element from each coset in TxW3 / W3.

###### cliff41Group.py

Generates the bosonic and fermionic representations of the group $\Gamma_{4,1}$, taste+charge conjugation.

###### stagGroup.py

Generates the bosonic and fermionic representations of the group $T \rtimes W3 \rtimes \Gamma_{4,1}$, the (non-staggered) lattice group. Irreps are saved using one representative element from each conjucacy class +  one representative element from each coset in TxW3xGamma / W3xGamma + one representative element from each coset in W3xGamma / W3.

###### continuumDecomp.py

Performs the decomposition from the continuum states of $SU(4)_T \times SU(2)_S$ to the states of $T \rtimes W3 \rtimes \Gamma_{4,1}$.

###### restframeIsoGroup.py

Generates the irreducible represenations of the "rest-frame group" $SW_4 \times \Gamma_{2,2}$ which is used in the continuum decomposition, including the isometry between the rest frame group and the group $W3 \rtimes \Gamma_{4,1}$.

###### cgcoeff.py

Functions to compute the Clebsch-Gordan coefficients, i.e the matrix which relates the direct product matrix indexed by products of single particle states to its block-diagonal irreducible form for the naive and staggered group case.

###### operatorsTXYZ.py

Generate staggered operators that transform under specific irreps.

###### wickContractions.py

Perform wick contractions from correlation functions, including taking the iso-symmetric limit.