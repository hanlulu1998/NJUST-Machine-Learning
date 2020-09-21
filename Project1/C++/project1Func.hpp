//
// Created by Han Lulu on 2020/9/19.
//

#ifndef PROJECT1_PROJECT1FUNC_HPP
#define PROJECT1_PROJECT1FUNC_HPP

#include<iostream>
#include<armadillo>

using namespace arma;

typedef std::pair<colvec, std::pair<double, double>> normFeature;
typedef std::pair<double, colvec> costGradient;
typedef std::pair<colvec, colvec> gradientNcost;

normFeature featureNormalize(const colvec &X);

colvec randInitializeTheta(int m, int n);

costGradient costFunction(const mat &X, const colvec &y,
                          const colvec &theta, double lambda);

gradientNcost gradientDescent(const mat &X, const colvec &y, const colvec &theta,
                              double alpha, int num_iters, double lambda);

double reFeature(double X_norm, double mu, double sigma);

colvec normalEqn(const mat &X, const colvec &y);

normFeature featureNormalize(const colvec &X) {
    double mu = mean(X);
    colvec X_norm = X - mu;
    double sigma = stddev(X_norm);
    X_norm = X_norm / sigma;
    std::pair<double, double> pram(mu, sigma);
    normFeature ret(X_norm, pram);
    return ret;
}


colvec randInitializeTheta(int m, int n) {
    double epsilon_init = 0.1;
    colvec theta = randu(m, n) * 2 * epsilon_init - epsilon_init;
    return theta;
}


costGradient costFunction(const mat &X, const colvec &y,
                          const colvec &theta, double lambda) {
    int m = y.n_elem;
    colvec h = X * theta;
    colvec t = theta;
    t(0) = 0;
    colvec temp = h - y;
    double J = sum(temp % temp) * 1.0 / (2.0 * m) + lambda / (2.0 * m) * accu(t % t);
    colvec grad = 1.0 / m * X.t() * (h - y) + lambda / m * t;
    costGradient ret = std::make_pair(J, grad);
    return ret;


}

gradientNcost gradientDescent(const mat &X, const colvec &y, const colvec &theta,
                              double alpha, int num_iters, double lambda) {
    int m = y.n_elem;
    colvec theta_i = theta;
    colvec J_history(num_iters, fill::zeros);
    for (int i = 0; i < num_iters; i++) {
        colvec h = X * theta_i;
        colvec temp = h - y;
        double temp0 = theta_i(0) - alpha * (1.0 / m) * accu(temp);
//        std::cout << "temp0=" << temp0 << std::endl;
        double temp1 = theta_i(1) - alpha * (1.0 / m) * accu(temp % X.col(1));
//        std::cout << "temp1=" << temp1 << std::endl;
        theta_i(0) = temp0;
        theta_i(1) = temp1;
//        std::cout << "theta_i=" << theta_i << std::endl;
        J_history(i) = costFunction(X, y, theta_i, lambda).first;
    }
    gradientNcost ret = std::make_pair(theta_i, J_history);
    return ret;
}

double reFeature(double X_norm, double mu, double sigma) {
    double X = X_norm * sigma;
    X = X + mu;
    return X;
}


colvec normalEqn(const mat &X, const colvec &y) {
    return pinv(X.t() * X) * X.t() * y;
}

#endif //PROJECT1_PROJECT1FUNC_HPP
