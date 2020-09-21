//
// Created by Han Lulu on 2020/9/18.
//
#include<iostream>
#include<armadillo>
#include"project1Func.hpp"

using namespace arma;

int main() {
    // load data
    std::cout << "Loading data..." << std::endl;
    colvec X = {2000, 2001, 2002, 2003, 2004, 2005, 2006,
                2007, 2008, 2009, 2010, 2011, 2012, 2013};
    colvec y = {2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704,
                6.853, 7.971, 8.561, 10.000, 11.280, 12.900};

    int m = y.n_elem;
    X.print("X:");
    y.print("y:");
    std::cout << "Normalizing feature..." << std::endl;
    //feature normalize
    normFeature tmp = featureNormalize(X);
    colvec X_norm = tmp.first;
    double X_mu = tmp.second.first;
    double X_sigma = tmp.second.second;

    tmp = featureNormalize(y);
    colvec y_train = tmp.first;
    double y_mu = tmp.second.first;
    double y_sigma = tmp.second.second;
    X_norm.print("X_norm:");
    y_train.print("y_norm:");
    std::cout << "Adding bias to X..." << std::endl;
    // add bias
    colvec v(m, fill::ones);
    mat X_trian = join_horiz(v, X_norm);

    X_trian.print("X_trian:");

    // random initialization theta
    std::cout << "Initializing theta randomly..." << std::endl;
    colvec theta = randInitializeTheta(2, 1);
    theta.print("initialization theta:");
    int iterations = 3000;
    double alpha = 0.01;
    double lambda = 1;

    std::cout << "Running Gradient Descent ..." << std::endl;
    // run gradient descent
    gradientNcost result = gradientDescent(X_trian, y_train, theta, alpha, iterations, lambda);
    std::cout << "Theta found by gradient descent:" << std::endl;
    theta = result.first;
    theta.print("theta:");

    //predict X=2014
    double Xt = 2014;
    double Xt_norm = Xt - X_mu;
    Xt_norm = Xt_norm / X_sigma;
    colvec Xp = {1, Xt_norm};
    colvec yt_norm = Xp.t() * theta;
    std::cout << "Predict the Nanjing housing price in 2014..." << std::endl;
    double yp = reFeature(yt_norm(0), y_mu, y_sigma);
    std::cout << "For X=2014,we predict that price is " << yp << std::endl;

    //Using normal equations to solve theta
    std::cout << "Solving normal equations..." << std::endl;
    theta = normalEqn(X_trian, y_train);
    std::cout << "Theta found by  close-form solution:" << std::endl;
    theta.print("theta:");

    //predict X=2014
    yt_norm = Xp.t() * theta;
    yp = reFeature(yt_norm(0), y_mu, y_sigma);
    std::cout << "Predict the Nanjing housing price in 2014..." << std::endl;
    std::cout << "For X=2014,we predict that price is " << yp << std::endl;
    return 0;


}
