Download Link: https://assignmentchef.com/product/solved-ee270-large-scale-matrix-computation-optimization-and-learning-hw-2
<br>
<h1>1.   Singular Value Decomposition (SVD) for Compression</h1>

Singular value decomposition (SVD) factorizes a <em>m </em>× <em>n </em>matrix <em>X </em>as <em>X </em>= <em>U</em>Σ<em>V </em><sup>&gt;</sup>, where <em>U </em>∈ R<em><sup>m</sup></em><sup>×<em>m </em></sup>and <em>U</em><sup>&gt;</sup><em>U </em>= <em>UU</em><sup>&gt; </sup>= <em>I</em>, Σ ∈ R<em><sup>m</sup></em><sup>×<em>n </em></sup>contains non-increasing non-negative values along its diagonal and zeros elsewhere, and <em>V </em>∈ R<em><sup>n</sup></em><sup>×<em>n </em></sup>and <em>V </em><sup>&gt;</sup><em>V </em>= <em>V V </em><sup>&gt; </sup>= <em>I</em>. Hence matrix can be represented aswhere <em>u<sub>i </sub></em>denotes the <em>i<sup>th </sup></em>column of U and <em>v<sub>i </sub></em>denotes the <em>i<sup>th </sup></em>column of <em>V </em>. Download the <a href="https://www.nasa.gov/sites/default/files/thumbnails/image/harvey-saturday-goes7am.jpg"><strong>image</strong></a><a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> and load it as the matrix X (in greyscale).

<ul>

 <li>Perform SVD on this matrix and zero out all but top <em>k </em>singular values to form an approximation <em>X</em>˜. Specifically, compute, display the resulting approximation</li>

</ul>

<em>X</em>˜ as an image, and report .

<ul>

 <li>How many numbers do you need to describe the approximation for <em>k </em>∈ {2<em>,</em>10<em>,</em>40} ?</li>

</ul>

<strong>Hint: </strong>Matlab can load the greyscale image using

X=double(rgb2gray(imread(’harvey-saturday-goes7am.jpg’))) and display the image using imagesc()

In Python 2.7 you can use: import numpy, Image

X=numpy.asarray(Image.open(’harvey.jpg’).convert(’L’))

<h1>2.   Singular Value Decomposition and Subspaces   Let <em>A </em>be an <em>m </em>× <em>n </em>singular matrix of rank <em>r </em>with SVD</h1>

 <em>σ</em><sub>1                                                      </sub>



               …                                                                    <em>~v</em>



                                                 <em>σ<sub>r                                                        </sub>~v</em>

<em>A </em>= <em>U</em>Σ<em>V <sup>T </sup></em>= <sup></sup><sub></sub><em>~u<sub>m </sub></em>

                                                           0



                                                            

                                                            

0

<em>σ</em><sub>1                                                    </sub>

…                                        

<em>σ                                   </em>

=                                                           <em><sub>r                             </sub></em>

0                      <sub></sub>

…         <sub></sub>

0




<em>~u</em><sub>1</sub><em>~u</em><sub>2</sub>

…









<em>U</em>ˆ     <em>U</em>˜                                                                      <em>V</em>ˆ˜<em>TT</em>

                                                                             <em>V</em>










where <em>σ</em><sub>1 </sub>≥ <em>… </em>≥ <em>σ<sub>r </sub>&gt; </em>0, <em>U</em>ˆ consists of the first <em>r </em>columns of <em>U</em>, <em>U</em>˜ consists of the remaining <em>m </em>− <em>r </em>columns of <em>U</em>, <em>V</em>ˆ consists of the first <em>r </em>columns of <em>V </em>, and <em>V</em>˜ consists of the remaining <em>n </em>− <em>r </em>columns of <em>V </em>. Give bases for the spaces range(<em>A</em>), null(<em>A</em>), range(<em>A<sup>T</sup></em>) and null(<em>A<sup>T</sup></em>) in terms of the components of the SVD of <em>A</em>, and a brief justification.

<h1>3.   Least Squares</h1>

Consider the least squares problem min<em><sub>x </sub></em>||<em>b </em>− <em>Ax</em>||<sub>2</sub>. Which of the following statements are necessarily true?

<ul>

 <li>If <em>x </em>is a solution to the least squares problem, then <em>Ax </em>= <em>b</em>.</li>

 <li>If <em>x </em>is a solution to the least squares problem, then the residual vector <em>r </em>= <em>b </em>− <em>Ax </em>is in the nullspace of <em>A<sup>T</sup></em>.</li>

 <li>The solution is unique.</li>

 <li>A solution may not exist.</li>

 <li>None of the above.</li>

</ul>

<h1>4.   Binary Classification via Regression</h1>

Download the MNIST dataset from <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist/</a><a href="http://yann.lecun.com/exdb/mnist/">.</a>

In binary classification, we restrict <em>Y </em>to take on only two values. Suppose <em>Y </em>∈ 0<em>,</em>1. Now let us use least squares linear regression for classification. Let us consider the classification problem of recognizing if a digit is 2 or not using linear regression. Here, let <em>Y </em>= 1 for all the 2’s digits in the training set, and use <em>Y </em>= 0 for all other digits.

Build a linear classifier by minimizing:

using the training set {(<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>)}. Use appropriate regularization (<em>λ &gt; </em>0) as needed. Let <em>x<sub>i </sub></em>be a vector of the pixel values of the image.

For the purpose of classification, we can label a digit as a 2 if <em>w<sup>T</sup>z </em>is larger than some threshold.

<ul>

 <li>Based on your training set, choose a reasonable threshold for classification. What is your 0/1 loss on the training set? What is your square loss on the training set?</li>

 <li>On the test set, what is the 0/1 loss? What is the square loss?</li>

</ul>

<h1>5.   Leave-One-Out Cross Validation (LOOCV)</h1>

Assume that there are n training examples, (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,</em>(<em>x</em><sub>2</sub><em>,y</em><sub>2</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>), where each input data point <em>x<sub>i </sub></em>, has d real valued features. The goal of regression is to learn to predict <em>y<sub>i </sub></em>from <em>x<sub>i </sub></em>. The linear regression model assumes that the output y is a linear combination of the input features plus Gaussian noise with weights given by <em>w</em>.

The maximum likelihood estimate of the model parameters <em>w </em>(which also happens to minimize the sum of squared prediction errors) is given by the normal equations we saw in class: (<em>X<sup>T</sup>X</em>)<sup>−1</sup><em>X<sup>T</sup>y</em>. Here, the rows of the matrix <em>X </em>are the training examples. Define <em>Y</em>ˆ to be the vector predictions using ˆ<em>w </em>if we were to plug in the original training set <em>X</em>:

<em>y</em>ˆ = <em>Xw</em>ˆ

= <em>X</em>(<em>X<sup>T</sup>X</em>)<sup>−1</sup><em>X<sup>T</sup>y </em>= <em>Hy</em>

where we define <em>H </em>= <em>X</em>(<em>X<sup>T</sup>X</em>)<sup>−1</sup><em>X<sup>T </sup></em>(<em>H </em>is often called the Hat Matrix). Show that <em>H </em>is a projection matrix.

As mentioned above, ˆ<em>w</em>, also minimizes the sum of squared errors:

<em>n</em>

<em>SSE </em>= <sup>X</sup>(<em>y<sub>i </sub></em>− <em>y</em>ˆ<em><sub>i</sub></em>)<sup>2</sup>

<em>i</em>=1

Leave-One-Out Cross Validation score is defined as:

<em>n</em>

<em>LOOCV </em>= X(<em>y</em><em>i </em>− <em>y</em>ˆ<em>i</em>(−<em>i</em>))2

<em>i</em>=1

where ˆ<em>y</em><sup>(−<em>i</em>) </sup>is the estimator of <em>y </em>after removing the <em>i<sup>th </sup></em>observation from the training set, i.e., it minimizes X                  − <em>y</em>ˆ<em><sub>j</sub></em>(−<em>i</em>))2<em>.</em>

(<em>y<sub>j</sub></em>

<em>j</em>6=<em>i</em>

<ul>

 <li>What is the time complexity of computing the LOOCV score naively? (The naive algorithm is to loop through each point, performing a regression on the <em>n</em>−1 remaining points at each iteration.)</li>

</ul>

<strong>Hint: </strong>The complexity of matrix inversion using classical methods is <em>O</em>(<em>k</em><sup>3</sup>) for a <em>k </em>× <em>k </em>matrix.

<ul>

 <li>Write ˆ<em>y<sub>i </sub></em>in terms of <em>H </em>and <em>y</em>.</li>

 <li>Show that ˆ<em>y</em><sup>(−<em>i</em>) </sup>is also the estimator which minimizes SSE for Z where</li>

 <li>Write ˆ in terms of <em>H </em>and <em>Z</em>. By definition, ˆ , but give an answer that is analogous to (b).</li>

 <li>Show that ˆ , where <em>H<sub>ii </sub></em>denotes the <em>i<sup>th </sup></em>element along the diagonal of <em>H</em>.</li>

 <li>Show that</li>

</ul>

What is the algorithmic complexity of computing the LOOCV score using this formula? <strong>Note: </strong>We see from this formula that the diagonal elements of <em>H </em>somehow indicate the impact that each particular observation has on the result of the regression.

<ol start="6">

 <li><strong>Sketching Least Squares  </strong>Here, we will consider the empirical performance of random sampling and random projection algorithms for approximating least-squares.</li>

</ol>

Let <em>A </em>be an <em>n </em>× <em>d </em>matrix, with  be an <em>n</em>-vector, and consider approximating the solution to min<em><sub>x </sub></em>k<em>Ax </em>− <em>b</em>k<sub>2</sub>. Generate the matrices <em>A </em>from one of three different classes of distributions introduced below

<ul>

 <li>Generate a matrix <em>A </em>from multivariate normal <em>N</em>(1<em><sub>d</sub>,</em>Σ), where the (i, j)th element of Σ<em><sub>ij </sub></em>= 2 × 0<em>.</em>5<sup>|<em>i</em>−<em>j</em>|</sup>.(Refer to as GA data.)</li>

 <li>Generate a matrix <em>A </em>from multivariate t-distribution with 3 degree of freedom and covariance matrix Σ as before. (Refer to as T3 data.)</li>

 <li>Generate a matrix <em>A </em>from multivariate t-distribution with 1 degree of freedom and covariance matrix Σ as before. (Refer to as T1 data.) To start, consider matrices of size <em>n </em>× <em>d </em>equal to 500 × 50.</li>

</ul>

<ul>

 <li>First, for each matrix, consider approximating the solution by randomly sampling a small number <em>r </em>of rows/elements (i.e., constraints of the overconstrained least-squares problem) in one of three ways: uniformly at random; according to an importance sampling distribution that is proportional to the Euclidean norms squared of the rows of <em>A</em>; and according to an importance sampling distribution that is proportional to the leverage scores of <em>A</em>. In each case, plot the error as a function of the number <em>r </em>of samples, paying particular attention to two regimes: <em>r </em>= <em>d,d</em>+1<em>,d</em>+2<em>,…,</em>2<em>d</em>; and <em>r </em>= 2<em>d,</em>3<em>d,…</em>. Show that the behavior of these three procedures is most similar for GA data, intermediate for T3 data, and most different for T1 data; and explain why, and explain similarities and differences.</li>

 <li>Next, for each matrix, consider approximating the solution by randomly projecting rows/elements (i.e., constraints of the overconstrained least-squares problem) in one of two ways: a random projection matrix in which each entry is i.i.d. {±1}, with appropriate variance; and a random projection matrix, in which each entry is i.i.d. Gaussian, with appropriate variance. In each case, plot the error as a function of the number of samples, i.e., dimensions on which the data are projected, paying particular attention to two regimes: <em>r </em>= <em>d,d </em>+ 1<em>,d </em>+ 2<em>,…,</em>2<em>d</em>; and <em>r </em>= 2<em>d,</em>3<em>d,…</em>. Describe and explain similarities and differences between these two procedures for GA data, T1 data, and T3 data.</li>

 <li>Finally, for each matrix, consider approximating the solution by randomly projecting rows/elements (i.e., constraints of the overconstrained least-squares problem) with sparse projection matrices. In particular, consider a random projection matrix, in which each entry is i.i.d. either 0 or Gaussian, where the probability of 0 is <em>q </em>and the probability of Gaussian is 1 − <em>q</em>. (Remember to rescale the variance of the Gaussian appropriately, depending on <em>q</em>) For <em>q </em>varying from 0 to 1, in increments sufficiently small, plot the error for solving the least-squares problem. Describe and explain how this varies as a function of the number of samples, i.e., dimensions on which the data are projected, paying particular attention to two regimes: <em>r </em>= <em>d,d</em>+1<em>,d</em>+2<em>,…,</em>2<em>d</em>; and <em>r </em>= 2<em>d,</em>3<em>d,…</em>. Describe and explain similarities and differences between these three procedures for GA data, T1 data, and T3 data.</li>

</ul>

Next, we describe how these behave for larger problems. To do so, we will work with dense random projection algorithms in which the projection matrix consists of i.i.d. {±1} random variables, with appropriate variance. Fix a value of <em>d</em>, and let <em>n </em>increase from roughly 2<em>d </em>to roughly 100<em>d</em>. The exact value of <em>d </em>and <em>n </em>will depend on your machine, your computational environment, etc., so that you get reasonably good low-precision approximate solutions to the original least-squares problem. (You should expect <em>d </em>≈ 500 should work; and if you can’t do the full plot to 100<em>d</em>, don’t worry, since the point is to get large enough to illustrate the phenomena below.)

<ul>

 <li>Plot the running time of the random projection algorithm versus the running time of solving the problem with a call to a QR decomposition routine provided by your system as well as the running time of solving the problem with a call to an SVD routine provided by your system. Illustrate that, for smaller problems the random projection methods are not faster, but that for larger problems, the random projection methods are slightly faster and/or can be used to solver larger problems than QR or SVD. Explain why is this the case.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://www.nasa.gov/sites/default/files/thumbnails/image/harvey-saturday-goes7am.jpg">https://www.nasa.gov/sites/default/files/thumbnails/image/harvey-saturday-goes7am.jpg</a>

<a href="https://www.nasa.gov/sites/default/files/thumbnails/image/harvey-saturday-goes7am.jpg">(NOAA’s GOES-East satellite’s image of Hurricane Harvey in the western Gulf of Mexico on Aug. 26, 2017. Credit:</a>

<a href="https://www.nasa.gov/sites/default/files/thumbnails/image/harvey-saturday-goes7am.jpg">NASA/NOAA GOES Project)</a>