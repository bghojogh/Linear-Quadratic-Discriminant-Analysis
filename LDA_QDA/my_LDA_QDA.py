import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from my_fit_GMM import My_fit_GMM
from sklearn import mixture #--> https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py    and    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html


class My_LDA_QDA:

    def __init__(self, X, y):
        # X: rows are features and columns are samples
        # y: a list
        self.X = X
        y = np.asarray(y)
        y = np.reshape(y, (-1, 1))
        self.y = y
        self.y_unique = np.unique(y)
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_classes = len(self.y_unique)
        self.priors = None
        self.unified_covariance = None
        self.means = None
        self.covariances = None

    def get_color_map(self, auto=True):
        if auto is True:
            # https://matplotlib.org/tutorials/colors/colormaps.html
            # cm = plt.cm.RdBu
            # cm = plt.cm.Set1
            cm = plt.cm.jet
        else:
            # Create the colormap
            # http://valdez.seos.uvic.ca/~jklymak/matplotlibhtml/gallery/images_contours_and_fields/custom_cmap.html
            # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
            # google it: python create discrete colormap
            cmap_name = 'my_list'
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
        return cm

    def plot_data(self, color_of_classes=["blue", "green", "red"]):
        # --- plot the data points:
        color_list = color_of_classes
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            color = color_list[class_index]
            # plt.plot(X_in_class[0, :], X_in_class[1, :], 'o', color=color)
            plt.scatter(X_in_class[0, :], X_in_class[1, :], color=color, edgecolors='k')
        plt.axis('equal')
        plt.show()

    def plot_data_and_estimationOfSpace(self, means, covariances, LDA_or_QDA_or_Bayes_or_naiveBayes="LDA", color_of_classes=["blue", "green", "red"]):
        # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        # --- mesh grid:
        h = 0.09  # step size in the mesh
        cm = self.get_color_map(auto=True)
        x_min, x_max = self.X[0, :].min() - .5, self.X[0, :].max() + .5
        y_min, y_max = self.X[1, :].min() - .5, self.X[1, :].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # --- classify the space:
        meshGrid_n_points = xx.shape[0] * xx.shape[1]
        mesh_grid_data = np.zeros((2, meshGrid_n_points))
        for index1 in range(xx.shape[0]):
            for index2 in range(xx.shape[1]):
                x_ = xx[index1, index2]
                y_ = yy[index1, index2]
                mesh_grid_data[:, (index1 * xx.shape[1]) + index2] = np.asarray([x_, y_])
        print("number of points in the mesh grid: " + str(mesh_grid_data.shape[1]))
        if LDA_or_QDA_or_Bayes_or_naiveBayes == "LDA":
            self.fit_LDA_or_QDA()
            estimated_class_for_mesh_grid = self.LDA_classify(X=mesh_grid_data)
        elif LDA_or_QDA_or_Bayes_or_naiveBayes == "QDA":
            self.fit_LDA_or_QDA()
            estimated_class_for_mesh_grid = self.QDA_classify(X=mesh_grid_data)
        elif LDA_or_QDA_or_Bayes_or_naiveBayes == "Bayes":
            self.fit_LDA_or_QDA()
            estimated_class_for_mesh_grid = self.Bayes_classify_for_Gaussians(X=mesh_grid_data, means=means, covariances=covariances)
        elif LDA_or_QDA_or_Bayes_or_naiveBayes == "naive_Bayes":
            self.fit_naive_Bayes()
            estimated_class_for_mesh_grid = self.naive_Bayes_classify(X=mesh_grid_data)
        elif LDA_or_QDA_or_Bayes_or_naiveBayes == "Bayes_mixtureModel":
            self.fit_LDA_or_QDA()
            whichClassIsMultiModal = 0
            if whichClassIsMultiModal == 0:
                mask = np.argwhere(np.array(self.y.ravel()) == 0).ravel()
                mask = mask.astype(int)
            elif whichClassIsMultiModal == 1:
                mask = np.argwhere(np.array(self.y) == 1).ravel()
                mask = mask.astype(int)
            X_multiModal = self.X[:, mask]
            # my_fit_GMM = My_fit_GMM(X=X_multiModal, n_models=2)
            # mean_multiModal, covariance_multiModal, weight_multiModal = my_fit_GMM.fit(n_iterations=1000, initial_mean=None, initial_cov=None)
            fit_GMM = mixture.GaussianMixture(n_components=2, covariance_type='full')
            fit_GMM.fit(X=X_multiModal.T)
            mean_multiModal = fit_GMM.means_
            covariance_multiModal = fit_GMM.covariances_
            weight_multiModal = fit_GMM.weights_
            print("Means of multi-modal class:")
            print(mean_multiModal)
            print("Covariances of multi-modal class:")
            print(covariance_multiModal)
            print("Weights of multi-modal class:")
            print(weight_multiModal)
            estimated_class_for_mesh_grid = self.Bayes_classify_for_Gaussians_multiModal(X=mesh_grid_data, means=means, covariances=covariances, object_my_fit_GMM=fit_GMM)
        # --- plot the data points:
        color_list = color_of_classes
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            color = color_list[class_index]
            # plt.plot(X_in_class[0, :], X_in_class[1, :], 'o', color=color)
            plt.scatter(X_in_class[0, :], X_in_class[1, :], color=color, edgecolors='k')
            # estimated_class_for_mesh_grid = np.ones((xx.shape[0] * xx.shape[1])) * 4  #----> just for test
        # --- plot the estimation of class of space (mesh grid) points:
        estimated_class_for_mesh_grid = estimated_class_for_mesh_grid.reshape(xx.shape)
        plt.contourf(xx, yy, estimated_class_for_mesh_grid, cmap=cm, alpha=.3)
        plt.axis('equal')
        plt.show()

    def fit_LDA_or_QDA(self):
        self.means = self.estimate_mean()
        self.covariances = self.estimate_covariance()
        self.unified_covariance = self.unify_covariances(self.covariances)
        self.priors = self.estimate_prior()

    def fit_naive_Bayes(self):
        self.means = [None] * self.n_classes
        self.covariances = [None] * self.n_classes
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            mean_of_dimensions = np.zeros((self.n_dimensions))
            variance_of_dimensions = np.zeros((self.n_dimensions))
            for dimension_index in range(self.n_dimensions):
                dimension_of_samples_of_class = X_in_class[dimension_index, :].ravel()
                mean_of_dimension_of_samples_of_class = np.mean(dimension_of_samples_of_class)
                variance_of_dimension_of_samples_of_class = np.var(dimension_of_samples_of_class, ddof=1)  #--> ddof=1 --> for unbiased estimation
                mean_of_dimensions[dimension_index] = mean_of_dimension_of_samples_of_class
                variance_of_dimensions[dimension_index] = variance_of_dimension_of_samples_of_class
            self.means[class_index] = mean_of_dimensions
            self.covariances[class_index] = variance_of_dimensions
        self.priors = self.estimate_prior()

    def LDA_classify(self, X):
        # X: rows are features and columns are samples
        estimated_class = np.zeros((X.shape[1]))
        n_samples = X.shape[1]
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            delta = np.zeros((self.n_classes))
            for class_index in range(self.n_classes):
                prior_of_class = self.priors[class_index]
                mean_of_class = self.means[class_index]
                delta[class_index] = (mean_of_class.T).dot(LA.inv(self.unified_covariance)).dot(sample) \
                                     - 0.5 * (mean_of_class.T).dot(LA.inv(self.unified_covariance)).dot(mean_of_class) \
                                     + np.log(prior_of_class)
            estimated_class[sample_index] = np.argmax(delta)
        return estimated_class

    def QDA_classify(self, X):
        # X: rows are features and columns are samples
        estimated_class = np.zeros((X.shape[1]))
        n_samples = X.shape[1]
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            delta = np.zeros((self.n_classes))
            for class_index in range(self.n_classes):
                prior_of_class = self.priors[class_index]
                mean_of_class = self.means[class_index]
                covariance_of_class = self.covariances[class_index]
                temp = (sample - mean_of_class).reshape((-1, 1))
                delta[class_index] = - 0.5 * np.log(np.linalg.det(covariance_of_class)) \
                                     - 0.5 * (temp.T).dot(LA.inv(covariance_of_class)).dot(temp) \
                                     + np.log(prior_of_class)
            estimated_class[sample_index] = np.argmax(delta)
        return estimated_class

    def Bayes_classify_for_Gaussians(self, X, means, covariances):
        # X: rows are features and columns are samples
        estimated_class = np.zeros((X.shape[1]))
        n_samples = X.shape[1]
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            delta = np.zeros((self.n_classes))
            for class_index in range(self.n_classes):
                prior_of_class = self.priors[class_index]
                mean_of_class = means[class_index]
                mean_of_class = np.asarray(mean_of_class).reshape((-1, 1))
                covariance_of_class = covariances[class_index]
                temp = (sample - mean_of_class).reshape((-1, 1))
                delta[class_index] = - 0.5 * np.log(np.linalg.det(covariance_of_class)) \
                                     - 0.5 * (temp.T).dot(LA.inv(covariance_of_class)).dot(temp) \
                                     + np.log(prior_of_class)
            estimated_class[sample_index] = np.argmax(delta)
        return estimated_class

    def Bayes_classify_for_Gaussians_multiModal(self, X, means, covariances, object_my_fit_GMM):
        # X: rows are features and columns are samples
        estimated_class = np.zeros((X.shape[1]))
        n_samples = X.shape[1]
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            delta = np.zeros((self.n_classes))
            for class_index in range(self.n_classes):
                prior_of_class = self.priors[class_index]
                if class_index == 0:
                    # PDF_mixture = object_my_fit_GMM.get_PDF_of_mixture_at_a_point(point=sample)
                    # delta[class_index] = prior_of_class * PDF_mixture
                    log_PDF_mixture = object_my_fit_GMM.score_samples(X=sample.T)
                    delta[class_index] = np.log(prior_of_class) + log_PDF_mixture
                elif class_index == 1:
                    mean_of_class = means[2] #--> the third class is now the second class
                    mean_of_class = np.asarray(mean_of_class).reshape((-1, 1))
                    covariance_of_class = covariances[2]  #--> the third class is now the second class
                    # delta[class_index] = prior_of_class * self.Gaussian_PDF(point=sample, mean_=mean_of_class, covariance_=covariance_of_class)
                    temp = (sample - mean_of_class).reshape((-1, 1))
                    delta[class_index] = - 0.5 * np.log(np.linalg.det(covariance_of_class)) \
                                         - 0.5 * (temp.T).dot(LA.inv(covariance_of_class)).dot(temp) \
                                         + np.log(prior_of_class)
            estimated_class[sample_index] = np.argmax(delta)
        return estimated_class

    def Gaussian_PDF(self, point, mean_, covariance_):
        point = np.array(point).reshape((-1, 1))
        mean_ = np.array(mean_).reshape((-1, 1))
        temp1 = 1 / (( ((2 * np.pi)**self.n_dimensions) * np.linalg.det(covariance_) )**0.5)
        temp2 = point - mean_
        temp3 = (temp2.T).dot(LA.inv(covariance_)).dot(temp2)
        temp4 = np.exp(-0.5 * temp3)
        PDF = temp1 * temp4
        return PDF

    def naive_Bayes_classify(self, X):
        # X: rows are features and columns are samples
        estimated_class = np.zeros((X.shape[1]))
        n_samples = X.shape[1]
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            delta = np.zeros((self.n_classes))
            for class_index in range(self.n_classes):
                prior_of_class = self.priors[class_index]
                label = self.y_unique[class_index]
                mask = (self.y == label).ravel()
                X_in_class = self.X[:, mask]
                posterior_in_class = 1
                for dimension_index in range(self.n_dimensions):
                    mean_ofDimensionInClass = self.means[class_index][dimension_index]
                    variance_ofDimensionInClass = self.covariances[class_index][dimension_index]
                    dimension_of_sample = sample[dimension_index, :]
                    posterior_of_dimension_in_class = (1 / (2 * np.pi * variance_ofDimensionInClass)**0.5) * \
                                                       np.exp(-1 * (dimension_of_sample - mean_ofDimensionInClass)**2 / (2 * variance_ofDimensionInClass))
                    posterior_in_class = posterior_in_class * posterior_of_dimension_in_class
                delta[class_index] = posterior_in_class * prior_of_class
            estimated_class[sample_index] = np.argmax(delta)
        return estimated_class

    def estimate_prior(self):
        priors = [None] * self.n_classes
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            n_samples_in_class = X_in_class.shape[1]
            prior_of_class = n_samples_in_class / self.n_samples
            priors[class_index] = prior_of_class
        return priors

    def estimate_mean(self):
        means = [None] * self.n_classes
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            mean_of_class = X_in_class.mean(axis=1).reshape((-1, 1))
            means[class_index] = mean_of_class
        return means

    def estimate_covariance(self):
        covariances = [None] * self.n_classes
        means = self.estimate_mean()
        for class_index in range(self.n_classes):
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            covariance_of_class = np.cov(X_in_class, bias=False)
            # mean_of_class = means[class_index].reshape((-1, 1))
            # cov = np.zeros((self.n_dimensions, self.n_dimensions))
            # n_samples_in_class = X_in_class.shape[1]
            # for sample_index in range(n_samples_in_class):
            #     sample = X_in_class[:, sample_index].reshape((-1, 1))
            #     temp = sample - mean_of_class
            #     cov = cov + temp.dot(temp.T)
            # covariance_of_class = (1 / (n_samples_in_class - 1)) * cov
            covariances[class_index] = covariance_of_class
        return covariances

    def unify_covariances(self, covariances):
        temp = np.zeros((self.n_dimensions, self.n_dimensions))
        for class_index in range(self.n_classes):
            covariance_of_class = covariances[class_index]
            label = self.y_unique[class_index]
            mask = (self.y == label).ravel()
            X_in_class = self.X[:, mask]
            n_samples_in_class = X_in_class.shape[1]
            temp = temp + (n_samples_in_class * covariance_of_class)
        unified_covariance = (1 / self.n_samples) * temp
        return unified_covariance
