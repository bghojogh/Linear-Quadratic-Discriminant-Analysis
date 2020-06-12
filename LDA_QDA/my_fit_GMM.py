import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal


# fit Gaussian Mixture Model (GMM):
class My_fit_GMM:

    def __init__(self, X, n_models):
        # X: rows are features and columns are samples
        self.X = X
        self.n_models = n_models
        self.n_samples = X.shape[1]
        self.n_dimensions = X.shape[0]
        self.means_of_mixture = None
        self.covariances_of_mixture = None
        self.weights_of_mixture = None

    def fit(self, n_iterations=1000, initial_mean=None, initial_cov=None):
        # initializations:
        gamma = np.random.rand(self.n_samples, self.n_models)
        weight = np.random.rand(self.n_models)
        weight = weight / np.sum(weight)
        if initial_mean is not None:
            mean_ = initial_mean
        else:
            # mean_ = np.random.rand(self.n_models, self.n_dimensions)
            mean_ = np.zeros((self.n_models, self.n_dimensions))
            mean_[0, :] = [-5, 5]
            mean_[1, :] = [5, -5]
        if initial_cov is not None:
            covariance_ = initial_cov
        else:
            # covariance_ = np.random.rand(self.n_models, self.n_dimensions, self.n_dimensions)
            covariance_ = np.zeros((self.n_models, self.n_dimensions, self.n_dimensions)) * 3
            for model_index in range(self.n_models):
                # covariance_[model_index, :, :] = np.eye(self.n_dimensions)
                covariance_[model_index, :, :] = self.generate_random_positive_definite_matrix(matrix_size=self.n_dimensions)
        for iteration_index in range(n_iterations):
            print("iteration " + str(iteration_index) + ":")
            # E step:
            for sample_index in range(self.n_samples):
                sample = self.X[:, sample_index].reshape((-1, 1))
                denominator_of_gamma = self.calculate_denominator_gamma(mean_, covariance_, weight, sample)
                for model_index in range(self.n_models):
                    mean_of_model = mean_[model_index, :].reshape((-1, 1))
                    covariance_of_model = covariance_[model_index, :, :]
                    # numerator_of_gamma = weight[model_index] * self.Gaussian_PDF(point=sample, mean_=mean_of_model, covariance_=covariance_of_model)
                    numerator_of_gamma = np.exp( np.log(weight[model_index]) * self.log_of_Gaussian_PDF(point=sample, mean_=mean_of_model, covariance_=covariance_of_model) )
                    gamma[sample_index, model_index] = numerator_of_gamma / denominator_of_gamma
            # M step:
            for model_index in range(self.n_models):
                # mean:
                numerator_of_mean = np.zeros((self.n_dimensions, 1))
                for sample_index in range(self.n_samples):
                    sample = self.X[:, sample_index].reshape((-1, 1))
                    numerator_of_mean = numerator_of_mean + (gamma[sample_index, model_index] * sample)
                denominator_of_meanOrCovariance = np.sum(gamma[:, model_index])
                mean_[model_index, :] = numerator_of_mean.ravel() / denominator_of_meanOrCovariance
                # covariance:
                numerator_of_covariance = np.zeros((self.n_dimensions, self.n_dimensions))
                for sample_index in range(self.n_samples):
                    sample = self.X[:, sample_index].reshape((-1, 1))
                    temp = sample - mean_[model_index, :]
                    numerator_of_covariance = numerator_of_covariance + (gamma[sample_index, model_index] * temp.dot(temp.T))
                covariance_[model_index, :, :] = numerator_of_covariance / denominator_of_meanOrCovariance
                # weight:
                numerator_of_weight = np.sum(gamma[:, model_index])
                denominator_of_weight = np.sum(gamma[:, :])
                weight[model_index] = numerator_of_weight / denominator_of_weight
            print("mean:")
            print(mean_[0, :])
            print(mean_[1, :])
            print("weight:")
            print(weight)
            print("covariance:")
            print(covariance_[0, :, :])
            print(covariance_[1, :, :])
        self.means_of_mixture = mean_
        self.covariances_of_mixture = covariance_
        self.weights_of_mixture = weight
        return mean_, covariance_, weight

    def get_PDF_of_mixture_at_a_point(self, point):
        PDF_mixture = 0
        for model_index in range(self.n_models):
            PDF_of_one_model = self.Gaussian_PDF(point=point, mean_=self.means_of_mixture, covariance_=self.covariances_of_mixture)
            weight_of_the_model = self.weights_of_mixture[model_index]
            PDF_mixture = PDF_mixture + (weight_of_the_model * PDF_of_one_model)
        return PDF_mixture

    def Gaussian_PDF(self, point, mean_, covariance_):
        # point = np.array(point).reshape((-1, 1))
        # mean_ = np.array(mean_).reshape((-1, 1))
        # temp1 = 1 / (( ((2 * np.pi)**self.n_dimensions) * np.linalg.det(covariance_) )**0.5)
        # temp2 = point - mean_
        # temp3 = (temp2.T).dot(LA.inv(covariance_)).dot(temp2)
        # temp4 = np.exp(-0.5 * temp3)
        # PDF = temp1 * temp4

        # point = np.array(point).reshape((-1, 1))
        # mean_ = np.array(mean_).reshape((-1, 1))
        # temp5 = -0.5 * self.n_dimensions * np.log(2 * np.pi)
        # temp6 = -0.5 * np.log(np.linalg.det(covariance_))
        # temp2 = point - mean_
        # temp3 = (temp2.T).dot(LA.inv(covariance_)).dot(temp2)
        # temp7 = -0.5 * temp3
        # log_of_PDF = temp5 + temp6 + temp7
        # PDF = np.exp(log_of_PDF)

        rv = multivariate_normal(mean_.ravel(), covariance_)
        PDF = rv.pdf(point.ravel())
        return PDF

    def log_of_Gaussian_PDF(self, point, mean_, covariance_):
        # point = np.array(point).reshape((-1, 1))
        # mean_ = np.array(mean_).reshape((-1, 1))
        # temp5 = -0.5 * self.n_dimensions * np.log(2 * np.pi)
        # temp6 = -0.5 * np.log(np.linalg.det(covariance_))
        # temp2 = point - mean_
        # temp3 = (temp2.T).dot(LA.inv(covariance_)).dot(temp2)
        # temp7 = -0.5 * temp3
        # log_of_PDF = temp5 + temp6 + temp7

        rv = multivariate_normal(mean_.ravel(), covariance_)
        PDF = rv.pdf(point.ravel())
        log_of_PDF = np.exp(PDF)

        return log_of_PDF

    def calculate_denominator_gamma(self, mean_, covariance_, weight, sample):
        denominator_of_gamma = 0
        for model_index in range(self.n_models):
            mean_of_model = mean_[model_index, :].reshape((-1, 1))
            covariance_of_model = covariance_[model_index, :, :]
            # ttt = weight[model_index] * self.Gaussian_PDF(point=sample, mean_=mean_of_model, covariance_=covariance_of_model)
            ttt = np.exp( np.log(weight[model_index]) * self.log_of_Gaussian_PDF(point=sample, mean_=mean_of_model, covariance_=covariance_of_model) )
            denominator_of_gamma = denominator_of_gamma + ttt
        return denominator_of_gamma

    def generate_random_positive_definite_matrix(self, matrix_size):
        # https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
        A = np.random.rand(matrix_size, matrix_size)
        B = np.dot(A, A.T)
        return B
