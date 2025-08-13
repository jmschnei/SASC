# Scientific Article Section Classification

This service is responsible for the classification of scientific article sections in differnet predefined section labels. 

Actually it uses a set of 9 labels: 'introduction', 'background', 'case', 'method', 'result', 'discussion', 'conclusion' and 'additional'.



## Installation
For the moment there are no installation instructions. The will follow.

## Usage 

The article segmentation tool can be used in different ways, each one fulfilling a different purpose: (i) preprocessing, (ii) OTHER; and (iii) REST API Controller, which can be used to segment new articles based on the trained models.


### (i) Preprocessing 

The preprocessinig of datasets or documents ir accomplished by using the preprocess.py file through command line. For example, you can use the following command to process the PubMed Central dataset:

```
python ScientificArticleSectionClassification/src/preprocessing/preprocess.py -mode generateLabeledDataFromPMC -lower_percentage 1
```

The -mode determines the method of the file preprocess.py that is going to be executed, and the rest of the parameters are passed as arguments to the method.


### (ii) Training

TBD

### (iii) Run the REST API Controller using FLASK and Python

To use directly the flask controller, the repository has to be cloned. Once the code is available, we have to install the requirements. Then, the flask controller must be run. The follolwing commands perform all these tasks:

```
git clone 
pip3 install -r requirements.txt
python3 
```

Once the controller is running, we can execute it using the following command:

```
curl ..........
```

## Support
If you need any support or help by using this code, please contact us at [julian.moreno_schneider@dfki.de](mailto:julian.moreno_schneider@dfki.de) or [raia.abu_ahmad@dfki.de](mailto:raia.abu_ahmad@dfki.de).

## Roadmap
The plan is to finish the first minimum viable prototype in Q2 2024. After that, there is no concrete planning  for further developments.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to the project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
The authors of the technology developed in this project are:

- Julián Moreno Schneider
- Raia Abu Ahmad
- Ekaterina Borisova

We would like to thank:

- DFKI Team
- Scilake and NFDI4DS Projects

## License
The concrete license has not been discussed yeet, but we are planning to use an open-source license.

## Project status
The project is currently under active development. The DFKI SLT department is further developing this technology.
