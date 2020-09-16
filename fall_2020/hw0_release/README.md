# Homework 0

Follow the instructions in `hw0.ipynb` to complete the assignment. Follow the instructions in [this webpage](http://vision.stanford.edu/teaching/cs131_fall1819/assignments.html) to learn how to complete the assignments and how to turn it in.


If you have not yet cloned (copied this assignment) into your local machine, follow the following steps:

1. In your terminal, navigate to a directory where you want to save your work in progress.
2. Clone the assignments repository by running:
```
git clone https://github.com/StanfordVL/CS131_release.git
```
3. Navigate into the homework 0 directory:
```
cd cs131/hw0
```
4. Setup a virtual environment that will install all your dependencies.
```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```
If you find that virtualenv is not installed in your local machine, install it by running:
```
sudo pip install virtualenv
```
5. Remember that everytime you take a break and then decide to go back to working on this assignment, you will need to activate your virtualenv to use the dependencies. You can activate your virtualenv by calling:
```
source .env/bin/activate
```
6. Now you are ready to start `hw0.ipynb` and start working on your assignment. Start the notebook by calling:
```
jupyter notebook
```
Now go to your web browser (ex, Google Chrome, Safari, Firefox) and go the this url: `localhost:8888`. Click on `hw0.ipynb` to start working on the assignment.
