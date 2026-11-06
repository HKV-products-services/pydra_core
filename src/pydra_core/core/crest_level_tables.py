import ast
import multiprocessing as mp
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# From this many crest level points (all model uncertainty combinations together), the crest levels are calculated
# in worker processes: a point takes about 15 microseconds, starting the worker processes about a second
MIN_POINTS_PARALLEL = 100_000

# Model uncertainty combinations per task: enough work per task, and enough tasks to keep the workers equally busy
COMBINATIONS_PER_TASK = 25


class CrestLevelTables:
    """
    The crest level tables of an HBN calculation with model uncertainty, calculated in worker processes.

    Every model uncertainty combination needs the crest level of every point of every loading model, one call of
    the overtopping library per point, and the combinations do not depend on each other. All tasks are submitted
    at once, in the order of the loop in 'HBN.calculate_location' (wave factors outside, water level shifts inside),
    so the worker processes start while the water level probabilities are calculated. A worker calculates with 'crest_level_table', the same function as
    'LoadingModel.calculate_hbn', on the same arguments, so the crest levels are the same.
    """

    def __init__(self, profile, qcrit: float, h_org: dict, loading, dh_values: list, wave_factors: np.ndarray, processes: int):
        """
        Start the worker processes and submit the crest level tables of all model uncertainty combinations.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        h_org : dict
            The water levels per loading model key, before the shift with the water level model uncertainty
        loading : LoadingOvertopping
            The overtopping loading
        dh_values : list
            The shifts of the water level, in the order in which the loop takes them for each wave factor
        wave_factors : np.ndarray
            The factors for the significant wave height and the spectral wave period, one row per combination
        processes : int
            The number of worker processes
        """
        self.n_dh = len(dh_values)
        self.n_combinations = len(wave_factors) * self.n_dh
        self._executor = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=mp.get_context("spawn"),
            initializer=CrestLevelTables.init_worker,
            initargs=(profile,),
        )
        combinations = [(dh, factor_hs, factor_tspec) for factor_hs, factor_tspec in wave_factors for dh in dh_values]
        self._futures = {}
        for start in range(0, self.n_combinations, COMBINATIONS_PER_TASK):
            for key, m in loading.iter_models():
                self._futures[(start, key)] = self._executor.submit(
                    CrestLevelTables.calculate_shifted,
                    qcrit,
                    h_org[key],
                    m.hs,
                    m.tspec,
                    m.dir,
                    combinations[start : start + COMBINATIONS_PER_TASK],
                )
        self._keys = [key for key, _ in loading.iter_models()]

    @staticmethod
    def worthwhile(loading, n_combinations: int, processes: int) -> bool:
        """
        Determine whether calculating the crest levels in worker processes is worth starting them. A daemon process
        can not start worker processes.

        Parameters
        ----------
        loading : LoadingOvertopping
            The overtopping loading
        n_combinations : int
            The number of model uncertainty combinations
        processes : int
            The number of worker processes

        Returns
        -------
        bool
            True if the crest levels are calculated in worker processes
        """
        n_points = sum(m.h.size for _, m in loading.iter_models()) * n_combinations
        if n_combinations < 2:
            return False
        return CrestLevelTables.worth_starting_workers(n_points, processes, "use HBN(..., processes=1)")

    def get(self, i_dh: int, i_wave: int) -> dict:
        """
        Return the crest levels per loading model key for a model uncertainty combination. A task is released after
        its last combination, so the tables are only kept until they are used.

        Parameters
        ----------
        i_dh : int
            The index in the water level shifts
        i_wave : int
            The index of the wave factors

        Returns
        -------
        dict
            The crest level table per loading model key
        """
        index = i_wave * self.n_dh + i_dh
        start = index - index % COMBINATIONS_PER_TASK
        last = index == min(start + COMBINATIONS_PER_TASK, self.n_combinations) - 1
        hbn = {}
        for key in self._keys:
            hbn[key] = self._futures[(start, key)].result()[index - start]
            if last:
                del self._futures[(start, key)]
        return hbn

    def close(self) -> None:
        """
        Stop the worker processes, also when the calculation stopped before all tables were used.
        """
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def default_processes() -> int:
        """
        Return the number of worker processes when none is given: the number of cores minus 2, at least 1.

        Returns
        -------
        int
            The number of worker processes
        """
        return max(1, (os.cpu_count() or 2) - 2)

    @staticmethod
    def crest_level_table(
        profile,
        qcrit: float,
        h: np.ndarray,
        hs: np.ndarray,
        tspec: np.ndarray,
        direction: np.ndarray,
        factor_hs: float,
        factor_tspec: float,
    ) -> np.ndarray:
        """
        Calculate the crest level for every point of a loading grid.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        h, hs, tspec, direction : np.ndarray
            The water level, significant wave height, spectral wave period and wave direction
        factor_hs : float
            Factor for the significant wave height, used for model uncertainty
        factor_tspec : float
            Factor for the spectral wave period, used for model uncertainty

        Returns
        -------
        np.ndarray
            The crest levels, with the shape of 'h'
        """
        _h = h.ravel()
        _hs = np.array(hs * factor_hs).ravel()
        _tspec = np.array(tspec * factor_tspec).ravel()
        _dir = np.array(direction).ravel()
        return np.reshape([profile.calculate_crest_level(qcrit, _h, _hs, _tspec, _dir)], h.shape)

    @staticmethod
    def worth_starting_workers(n_points: int, processes: int, hint: str) -> bool:
        """
        Determine whether a number of crest level points is worth starting worker processes for. A daemon process can
        not start worker processes.

        Parameters
        ----------
        n_points : int
            The number of crest level points
        processes : int
            The number of worker processes
        hint : str
            Completes the warning for a main script without 'if __name__ == "__main__":' with how to calculate in this
            process without the warning

        Returns
        -------
        bool
            True if the crest levels are calculated in worker processes
        """
        if processes < 2 or n_points < MIN_POINTS_PARALLEL or mp.current_process().daemon:
            return False
        if not CrestLevelTables.main_module_is_guarded():
            print(
                "[WARNING] The main script has no 'if __name__ == \"__main__\":', so worker processes would run it "
                f"again; the crest levels are calculated in this process. Put the calculation under that line, or "
                f"{hint} to hide this warning."
            )
            return False
        return True

    @staticmethod
    def calculate_in_workers(
        profile, qcrit: float, work: list[tuple[tuple, list[tuple]]], processes: int
    ) -> list[list]:
        """
        Calculate crest level tables in worker processes. A worker calculates
        'crest_level_table(profile, qcrit, h + delta_h, hs, tspec, direction, factor_hs, factor_tspec)', the call of the
        ports, so the tables are the same.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        work : list[tuple[tuple, list[tuple]]]
            Per entry a loading (h, hs, tspec, direction) and its combinations (delta_h, factor_hs, factor_tspec)
        processes : int
            The number of worker processes

        Returns
        -------
        list[list]
            Per entry of 'work' the crest level table of each combination, in the same order
        """
        tasks = [
            (index, loading, combinations[start : start + COMBINATIONS_PER_TASK])
            for index, (loading, combinations) in enumerate(work)
            for start in range(0, len(combinations), COMBINATIONS_PER_TASK)
        ]
        tables = [[] for _ in work]
        with ProcessPoolExecutor(
            max_workers=processes,
            mp_context=mp.get_context("spawn"),
            initializer=CrestLevelTables.init_worker,
            initargs=(profile,),
        ) as executor:
            futures = [
                (index, executor.submit(CrestLevelTables.calculate_shifted, qcrit, *loading, part))
                for index, loading, part in tasks
            ]
            for index, future in futures:
                tables[index].extend(future.result())
        return tables

    @staticmethod
    def main_module_is_guarded() -> bool:
        """
        Determine whether a worker process can load the main script again. A worker process is started with 'spawn',
        which runs the main script again except the part under 'if __name__ == "__main__":'; without that line, every
        worker would start the whole script again. Without a main script (an interactive session, a notebook) nothing
        is run again.

        Returns
        -------
        bool
            True if there is no main script or it has an 'if __name__ == "__main__":' block
        """
        path = getattr(sys.modules.get("__main__"), "__file__", None)
        if path is None:
            return True
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return False
        for node in tree.body:
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if "__name__" in test and "__main__" in test:
                    return True
        return False

    @staticmethod
    def init_worker(profile) -> None:
        """
        Keep the profile in the worker process.

        Parameters
        ----------
        profile : Profile
            The profile
        """
        global _profile
        _profile = profile


    @staticmethod
    def calculate_shifted(
        qcrit: float, h: np.ndarray, hs: np.ndarray, tspec: np.ndarray, direction: np.ndarray, combinations: list
    ) -> list:
        """
        Calculate the crest level tables of one loading for a series of combinations, in a worker process.

        Parameters
        ----------
        qcrit : float
            The critical discharge
        h, hs, tspec, direction : np.ndarray
            The water level, significant wave height, spectral wave period and wave direction
        combinations : list
            The water level shift and the factors for the significant wave height and the spectral wave period, one
            tuple per combination

        Returns
        -------
        list
            The crest level table per combination
        """
        return [
            CrestLevelTables.crest_level_table(_profile, qcrit, h + dh, hs, tspec, direction, factor_hs, factor_tspec)
            for dh, factor_hs, factor_tspec in combinations
        ]


_profile = None
