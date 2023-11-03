# DeepEST
This is the repository with the implementation of DeepEST, a multimodal method to perform protein function prediction for bacterial species. Given the characteristics of their genome, the functional characterization of bacteria cannot rely solely on protein sequences. In fact, it requires the use of data sources capable of capturing different dimensions of the protein functionality, i.e., protein structures and expression-location patterns.

The architecture of DeepEST is visualized in the following figure:

![method](https://github.com/BorgwardtLab/DeepEST/assets/56036317/bb421134-1ad6-4bc9-8220-440db949b624)

## Repository organization


## Data
We study the following 25 bacterial species:
![Species](https://github.com/BorgwardtLab/DeepEST/assets/56036317/7a24e712-8b1d-41c1-8990-a1272a8094a2)

### Expression and location data

### Protein structures

### GO terms annotations
GO annotations are retrieved from the [UniProt database](https://www.uniprot.org/) (accessed on July 12, 2023) using the RefSeq protein identifier of every known protein and the taxonomic reference code of a given pathogen's strain at hand. 
To retrieve a particular GO term's children or ancestors we use the GO ontology released on October 7, 2022.


## Contacts
For queries on the implementation and data, please contact:
- giulia.muzio@bsse.ethz.ch
- leyden.fernandez@umu.se


## Funding
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement (No 813533). Swedish Research Council (No. 2021-02466), Kempes-tiftelserna (JCK22-0017), Insamlingsstiftelsen, Medical Faculty at Ume ̊a University to K Avican and Icelab Multidisciplinary Postdoctoral Fellowship to L Fernandez.
