import numpy as np

from dataclasses import dataclass
from scipy.stats import multivariate_normal, norm

from ....settings.settings import Settings
from .....io.database_hr import DatabaseHR


class ModelUncertainty:
    """
    Model uncertainties class. Containing all model uncertainties for each closing situation.

    Attributes
    ----------
    model_uncertainties : dict
        A dictionary with
    """

    # Init attributes
    model_uncertainties = {}
    correlations = {}

    def __init__(self, settings: Settings):
        """
        Read the model uncertainties from the database and add them to this class

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Save the discretisation step size
        self.step_size = {
            "h": settings.model_uncertainty_water_level_steps,
            "hs": settings.model_uncertainty_wave_height_steps,
            "tspec": settings.model_uncertainty_wave_period_steps,
        }

        # Obtain the model uncertainties and correlation between model uncertainties
        with DatabaseHR(settings.database_path) as database:
            mu = database.get_model_uncertainties(settings)
            cu = database.get_correlation_uncertainties(settings)

        # Iterate over the model uncertainties and add them to the class
        for comb, uncertainty in mu.groupby(["k", "rvid"]):
            self.model_uncertainties[comb] = DistributionUncertainty(
                uncertainty.to_numpy()[0]
            )

        # Iterate over the correlation between model uncertainties and add them to the class
        if cu is not None:
            for comb, correlation in cu.groupby(["k", "rvid", "rvid2"]):
                self.correlations[comb] = CorrelationUncertainty(
                    correlation.to_numpy()[0]
                )

    def iterate_model_uncertainty_wave_conditions(
        self, closing_situation: int = 1, wave_period: str = "tspec"
    ):
        """
        Iterate over all model uncertainty combinations for the significant wave height and wave period

        Parameters
        ----------
        closing_situation : int, optional
            The closing situation id (default : 1)
        wave_period : str, optional
            Whether to iterate over peak period (tp) or spectral wave period (tspec) (default : 'tspec')

        Returns
        -------
        iterator
            (hs, t, probability of (hs, t))
        """
        # Check if tp or tspec
        wave_period = wave_period.lower()
        if wave_period not in ["tp", "tspec"]:
            raise KeyError(f"[ERROR] Wave period '{wave_period}' unknown.")

        # Distributions
        mu_hs = self.get_model_uncertainty("hs", closing_situation)
        mu_t = self.get_model_uncertainty(wave_period, closing_situation)

        # Significant wave height
        if mu_hs is not None:
            hs, hsedges = mu_hs.discretise(self.step_size["hs"])
            tmp = norm.cdf(hsedges)
            p_hs = tmp[1:] - tmp[:-1]
        else:
            hs, p_hs = [1.0], np.array([1.0])

        # Wave period
        if mu_t is not None:
            t, tedges = mu_t.discretise(self.step_size["tspec"])
            tmp = norm.cdf(tedges)
            p_t = tmp[1:] - tmp[:-1]
        else:
            t, p_t = [1.0], np.array([1.0])

        # Multiply the probabilities (assuming independence)
        combined_probability = p_hs[:, None] * p_t[None, :]

        # Obtain the correlation between wave height and period (optional, otherwise None)
        corr_hs_t = self.get_correlation("hs", wave_period)

        # If rho is defined, apply correlation
        if corr_hs_t is not None:
            # Correlation can only be applied when both the wave height and period are defined
            if (mu_hs is not None) and (mu_t is not None):
                # Apply correlation
                exc_probs = np.zeros((len(hsedges), len(tedges)))
                for i, x in enumerate(hsedges):
                    for j, y in enumerate(tedges):
                        exc_probs[i, j] = multivariate_normal.cdf(
                            [x, y],
                            mean=(0, 0),
                            cov=[[1, corr_hs_t.rho], [corr_hs_t.rho, 1]],
                        )

                # Take the difference in both directions
                combined_probability = exc_probs[1:] - exc_probs[:-1]
                combined_probability = (
                    combined_probability[:, 1:] - combined_probability[:, :-1]
                )

            # Otherwise give a warning
            else:
                print(
                    "[WARNING] Correlation between wave height and period defined. However can not be applied because no model uncertainty is defined for either the wave height, period or both."
                )

        # Check
        assert abs(combined_probability.sum() - 1) < 1e-6

        # Iterator
        for i, fh in enumerate(hs):
            for j, ft in enumerate(t):
                yield fh, ft, combined_probability[i, j]

    def get_model_uncertainty(self, result_variable: str, closing_situation: int = 1):
        """
        Return the model uncertainty object for a result variable and closing situation id

        Parameters
        ----------
        result_variable : str
            Result variable (e.g. h, hs)
        closing_situation : int
            Closing situation ID (default: 1)

        Returns
        -------
        DistributionUncertainty or None
            The DistributionUncertainty object if it exists, otherwise None
        """
        # To lower
        rv = result_variable.lower()

        # Try to return the model uncertainty, otherwise return None
        return self.model_uncertainties.get((closing_situation, rv))

    def get_correlation(
        self, result_variable1: str, result_variable2: str, closing_situation: int = 1
    ):
        """
        Return the correlation object between two result variables given a closing situation id

        Parameters
        ----------
        result_variable1 : str
            Result variable (e.g. h, hs)
        result_variable2 : str
            Result variable (e.g. h, hs)
        closing_situation : int
            Closing situation ID (default: 1)

        Returns
        -------
        Correlation or None
            The Correlation object if it exists, otherwise None
        """
        # To lower
        rv1 = result_variable1.lower()
        rv2 = result_variable2.lower()

        # Check both orders [ccid, rvid1, rvid2]
        return self.correlations.get(
            (closing_situation, rv1, rv2),
            self.correlations.get((closing_situation, rv2, rv1)),
        )

    def process_model_uncertainty(
        self,
        closing_situation: int,
        result_variable: str,
        levels: np.ndarray,
        exceedance_probability: np.ndarray,
        haxis: int,
    ):
        """
        Verwerk modelonzekerheid in gegeven stochast. Afhankelijk van de stochast wordt de onzekerheid
        opgeteld (additief) of vermenigvuldigd (multiplicatief).

        Args:
            stochast (str): Stochastnaam. h, hs, tp of tspec
            niveaus (np.ndarray): Niveaus waarin de onzekerheden worden ingedeeld.
            ovkansen (np.ndarray): Overschrijdingskansen van de stochast, met eventueel meerdere dimensies
            die meegeïntegreerd
            haxis (int): as waarop de te integreren stochast zit
            sluitsituatie (str): Keringsituatie, in sommige gevallen is de onzekerheid hiervan afhankelijk

        Returns:
            np.ndarray: Overschrijdingskansen met geïntegreerde onzekerheid
        """
        # Obtain distribution
        dis = self.get_model_uncertainty(result_variable, closing_situation)

        # Is the model uncertainty defined?
        if dis is None:
            return exceedance_probability

        # Additive (h) or Multiplicative (hs, tspec, tp)
        if result_variable in ["h"]:
            klassekansen = self.bepaal_klassekansen_additief(levels, dis.mu, dis.sigma)

        elif result_variable in ["hs", "tp", "tspec"]:
            klassekansen = self.bepaal_klassekansen_multiplicatief(
                levels, dis.mu, dis.sigma
            )

        else:
            raise KeyError(result_variable)

        # Calculate the exceedance probability
        exceedance_probability = np.tensordot(
            klassekansen, exceedance_probability, axes=([0], [haxis])
        )

        return exceedance_probability

    def bepaal_klassekansen_additief(self, niveaus, mu, sigma):
        # Bepaal klassegrenzen en klassekansen
        hgrens = np.concatenate(
            [[-np.inf], (niveaus[1:] + niveaus[:-1]) / 2.0, [np.inf]]
        )
        klassekansen = []

        # Bereken per niveau (waterstand) de kans dat de waterstand door onzekerheid in een andere klasse valt
        for niveau in niveaus:
            grenskansen = norm.cdf(x=hgrens - mu, loc=niveau, scale=sigma)
            klassekansen.append(grenskansen[1:] - grenskansen[:-1])
        klassekansen = np.array(klassekansen)

        return klassekansen

    def bepaal_klassekansen_multiplicatief(self, niveaus, mu, sigma):
        # Bepaal klassegrenzen en klassekansen
        hgrens = np.concatenate(
            [[-np.inf], (niveaus[1:] + niveaus[:-1]) / 2.0, [np.inf]]
        )
        klassekansen = []

        # Bereken per niveau (waterstand) de kans dat de waterstand door onzekerheid in een andere klasse valt
        for niveau in niveaus:
            grenskansen = norm.cdf(hgrens / niveau, loc=mu, scale=sigma)
            klassekansen.append(grenskansen[1:] - grenskansen[:-1])
        klassekansen = np.array(klassekansen)

        return klassekansen


class DistributionUncertainty:
    """
    Model uncertainty class for a closing situation.
    """

    def __init__(self, uncertainty: list):
        """
        Initialise the model uncertainty (Normal Distribution)
        """
        # Save information
        self.k, self.rvid, self.mu, self.sigma = uncertainty
        self.k = int(self.k)
        self.mu = float(self.mu)
        self.sigma = float(self.sigma)

    def discretise(self, nsteps: int):
        """
        Discretise the model uncertainty

        Parameters
        ----------
        nsteps : int
            Number of steps

        Returns
        -------
        probabilities : list
            List of the probabilities for each bin
        edges : list
            List with the edges of each bin
        """
        # If nsteps is 1, there is no need to discretise
        if nsteps == 1:
            return self.mu, [self.mu - 100 * self.sigma, self.mu + 100 * self.sigma]

        # Determine the residual probabilities
        keuzerestkans = 0.05
        restkans = keuzerestkans / (nsteps**1.5)
        afstand = -norm.ppf(q=0.5 * restkans, loc=0.0, scale=1.0) * self.sigma
        ondergrens = self.mu - afstand
        bovengrens = self.mu + afstand

        # Calculate the probability at the center of the bin
        probabilities = (
            ondergrens + np.arange(0.5, nsteps, 1) * (bovengrens - ondergrens) / nsteps
        )

        # Determine the edges of the bins
        edges = (
            np.concatenate(
                [
                    [self.mu - 100 * self.sigma],
                    (probabilities[1:] + probabilities[:-1]) / 2,
                    [self.mu + 100 * self.sigma],
                ]
            )
            - self.mu
        ) / self.sigma

        # Return probabilities and edges
        return probabilities, edges


@dataclass
class CorrelationUncertainty:
    """
    Class om de correlaties tussen uitvoervariabelen op te slaan.
    Bijv. de correlatie tussen Hs en Tspec
    """

    def __init__(self, correlation: list):
        """
        Initialise the correlation between two result variables
        """
        # Save information
        self.k, self.rvid, self.rvid2, self.rho = correlation
        self.k = int(self.k)
        self.rho = float(self.rho)

        # Only allow correlation between Hs and Tp and Hs and Tspec
        if [self.rvid, self.rvid2] not in [
            ["hs", "tp"],
            ["tp", "hs"],
            ["hs", "tspec"],
            ["tspec", "hs"],
        ]:
            raise ValueError(
                f"Not Implemented: Correlation between ({self.rvid}) and ({self.rvid2})"
            )
