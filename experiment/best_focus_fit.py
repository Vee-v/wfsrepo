import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
from pathlib import Path
from scipy.optimize import curve_fit
from torchvision.transforms import v2
from uncertainties import ufloat, unumpy

def parabola(x, a, p, c):
    return a*(x - p)**2 + c

def split_wfs_image(img):
    """
    Splits an image into its subapertures.
    Creates an view of a the input tensor. (121, 28, 28)

    Returns:
        Tensor: images of each subaperture.
    
    """

    img = v2.CenterCrop(308)(img)
    subaps = img.unfold(0, 28, 28).unfold(1, 28, 28)
    return subaps.contiguous().view(-1, 28, 28)

data = Path("measurements")

depth = np.array([0.56, 0.85, 1.30, 1.67, 2.08, 2.53, 2.75, 3.15, 3.71, 4.00, 4.49, 4.80])
depth_offset = np.mean(depth)
depth_err = 0.05
pixels = np.zeros([depth.size], dtype=np.dtype(object))

threshold = 1
for i, d in enumerate(depth):
    img = np.load(data/f"focus_img_{d:.2f}.npy")
    subaps = split_wfs_image(torch.from_numpy(img))
    pixelsum = torch.sum(subaps>=threshold, axis=(1,2), dtype=torch.float16)

    pixels[i] = np.mean(ufloat(pixelsum.mean(), pixelsum.std()))




# # We will use the simplest form of GP model, exact inference
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2, offset_prior=gpytorch.priors.NormalPrior(0.0, 1.0)))
    
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # initialize likelihood and model
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(depth, counts, likelihood)


# # Find optimal model hyperparameters
# model.train()
# likelihood.train()

# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# training_iter = 200000

# for i in tqdm(range(training_iter)):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     output = model(depth)
#     # Calc loss and backprop gradients
#     loss = -mll(output, counts)
#     loss.backward()
    
#     optimizer.step()

# print('Loss: %.3f   offset: %.3f   noise: %.3f' % (
#          loss.item(),
#         model.covar_module.base_kernel.offset.item(),
#         model.likelihood.noise.item()
#     ))

# test_x = torch.linspace(0, 6, 1000)
# # f_preds = model(test_x)
# # y_preds = likelihood(model(test_x))

# # f_mean = f_preds.mean
# # f_var = f_preds.variance
# # f_covar = f_preds.covariance_matrix
# # f_samples = f_preds.sample(sample_shape=torch.Size(1000,))


# # Get into evaluation (predictive posterior) mode
# model.eval()
# likelihood.eval()

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))


# with torch.no_grad():
#     # Initialize plot

#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black stars
#     plt.plot(depth.numpy(), counts.numpy(), 'k*')
#     # Plot predictive means as blue line
#     plt.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     # Shade between the lower and upper confidence bounds
#     plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     # ax.set_ylim([-3, 3])
#     plt.legend(['Observed Data', 'Mean', 'Confidence'])
#     plt.ylabel("Counts")
#     plt.xlabel("Housing Depth [mm]")
#     focus = test_x[torch.argmin(observed_pred.mean)]
#     plt.title(f"focus @ {focus:.3f}mm")
#     plt.show()


popt, pcov = curve_fit(parabola, depth, unumpy.nominal_values(pixels), sigma=unumpy.std_devs(pixels), p0=(1, 2.8, 10))
sigma = np.sqrt(np.diagonal(pcov))


domain = np.linspace(0, 5, 1000)
# 
# an easy way to properly format parameter errors
from uncertainties import ufloat
a = ufloat(popt[0], sigma[0])
p = ufloat(popt[1], sigma[1])
c = ufloat(popt[2], sigma[2])

text_res = f"Best fit parameters:\na = {a}\np = {p}\nc = {c}"
print(text_res)
bound_upper = parabola(domain, *(popt + sigma))
bound_lower = parabola(domain, *(popt - sigma))
# plotting the confidence intervals
plt.fill_between(domain, bound_lower, bound_upper,
                 color = 'black', alpha = 0.15, label="Confidence")

plt.plot(domain, parabola(domain, *popt), label="Off-center parabola fit", zorder=0)
plt.errorbar(depth, unumpy.nominal_values(pixels), yerr=unumpy.std_devs(pixels), capsize=3, fmt="k.", zorder=1)
plt.scatter(depth, unumpy.nominal_values(pixels), edgecolor="k", facecolor="silver", label="Data",zorder=2)
plt.axvline(p.n, ls="--", color='k', linewidth=0.8)
plt.fill_betweenx([-1, 35], p.n-p.s, p.n+p.s, alpha=0.2, color='k')
plt.ylabel("Non-zero pixel count")
plt.xlabel("Housing Depth [mm]")
plt.ylim((0,35))
plt.xlim((domain.min(), domain.max()))
plt.text(1.1, 20, f"MLA focal length\n@ {p:.3f}mm")
plt.legend()
plt.show()
